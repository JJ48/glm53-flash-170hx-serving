#!/usr/bin/env python3
"""Add weight-only INT8 (or INT4) to the BF16 remainder of a compressed-tensors checkpoint, streaming shard by shard on CPU.

Built for wtdcode/GLM-5.3-Flash-AWQ-W4A16 (experts already INT4 pack-quantized; attention, dense MLPs, shared experts,
lm_head kept BF16). Uses compressed-tensors' own quantize/pack routines so vLLM's loader sees exactly the format it expects.

  python3 requant_remainder.py selftest
  python3 requant_remainder.py plan   --src /data/glm53-awq [--profile kda]
  python3 requant_remainder.py run    --src /data/glm53-awq --dst /data/glm53-awq-w8rem --profile kda --bits 8 --shard-gb 8

Profiles (what gets quantized; everything else is copied unchanged):
  kda      KDA (linear-attention) layers: q/k/v/b/f_a/g_a input projections + o_proj; dense MLP layers 0-2; shared experts.
           Needs fork patch 0002 (kda.py: keep self.quant_config after the parent init) — see ../fork-patches.
  kda_mla  kda + the MLA layers' q_b_proj / kv_b_proj / o_proj. REQUIRES a one-line fork patch: Glm5NextDecoderLayer passes
           quant_config=None to Glm5NextMLAAttention; patch 0001 passes the real quant_config (the ignore list protects the rest).
  aggr     kda_mla + lm_head.
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time
from collections import OrderedDict
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
try:
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32   # >= 0.18
except ImportError:  # older layouts
    try:
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import pack_to_int32, unpack_from_int32
    except ImportError:
        from compressed_tensors.compressors.quantized_compressors.helpers import pack_to_int32, unpack_from_int32

KDA_IN = ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"]   # fused into in_proj_qkvbfg_a by the fork
KDA_KEEP_BF16 = ["f_b_proj", "g_b_proj"]                                     # small separate linears, left BF16
MLA_BIG = ["q_b_proj", "kv_b_proj", "o_proj"]
MLA_KEEP_BF16 = ["q_a_proj", "kv_a_proj_with_mqa"]                           # latent projections (fused_qkv_a_proj), left BF16

def layer_kinds(names):
    """layer_idx -> 'kda' | 'mla' from tensor names (config-free)."""
    kinds = {}
    for n in names:
        m = re.search(r"\.layers\.(\d+)\.self_attn\.(\w+)", n)
        if not m: continue
        i, sub = int(m.group(1)), m.group(2)
        if sub == "kv_b_proj": kinds[i] = "mla"
        elif sub in ("k_conv1d", "A_log"): kinds[i] = "kda"
    return kinds

def build_plan(names, profile, mtp_layer):
    kinds = layer_kinds(names)
    kda = sorted(i for i, k in kinds.items() if k == "kda" and (i != mtp_layer or MTP_QUANT))
    mla = sorted(i for i, k in kinds.items() if k == "mla" and (i != mtp_layer or MTP_QUANT))
    tq = set()  # tensor names to quantize (the .weight tensors)
    for n in names:
        if not n.endswith(".weight"): continue
        m = re.search(r"\.layers\.(\d+)\.", n); li = int(m.group(1)) if m else None
        if "experts." in n and "shared_experts" not in n:
            if MTP_QUANT and li == mtp_layer and re.search(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$", n): tq.add(n)
            continue
        if li == mtp_layer and not MTP_QUANT: continue
        if re.search(r"\.self_attn\.(" + "|".join(KDA_IN) + r"|o_proj)\.weight$", n) and li in kda: tq.add(n)
        elif re.search(r"\.shared_experts\.(gate_proj|up_proj|down_proj)\.weight$", n): tq.add(n)
        elif re.search(r"\.layers\.(0|1|2)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", n): tq.add(n)
        elif profile in ("kda_mla", "aggr") and li in mla and re.search(r"\.self_attn\.(" + "|".join(MLA_BIG) + r")\.weight$", n): tq.add(n)
        elif profile == "aggr" and n == "lm_head.weight": tq.add(n)
    return tq, kda, mla

INT4_CATS = None; INT4_GROUP = 32
MTP_QUANT = False   # --mtp-quant: quantize the MTP head (layer 45) too; its routed experts use group_0's INT4 g128 symmetric format by default
MTP_EXPERT_GROUP = 128
MTP_EXPERT_BITS = 4   # --mtp-expert-bits 8: INT8 g128 experts in their own config group (vLLM resolves the MoE scheme per layer by name: layer_name + '.0.gate_proj' etc.)
def profile_cats(profile):
    base = ["kda_in", "kda_o", "shared", "dense"]
    if profile in ("kda_mla", "aggr"): base.append("mla")
    if profile == "aggr": base.append("lm_head")
    return base

def new_quant_config(qc, profile, bits, group_size, kda, mla, mtp_layer):
    qc = json.loads(json.dumps(qc))
    strategy = "channel" if group_size in (None, 0, -1) else "group"
    lay = lambda idxs: "(" + "|".join(str(i) for i in idxs) + ")"
    targets = [rf"re:.*\.layers\.{lay(kda)}\.self_attn\.({'|'.join(KDA_IN)}|o_proj)$",
               rf"re:.*\.layers\.{lay(kda)}\.self_attn\.in_proj_qkvbfg_a$",     # the fork's fused module name
               r"re:.*\.shared_experts\.(gate_proj|up_proj|down_proj|gate_up_proj)$",
               r"re:.*\.layers\.(0|1|2)\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)$"]
    if profile in ("kda_mla", "aggr"):
        targets.append(rf"re:.*\.layers\.{lay(mla)}\.self_attn\.({'|'.join(MLA_BIG)})$")
    if profile == "aggr":
        targets.append("re:.*lm_head$")  # the fork registers the head as language_model.lm_head; exact "lm_head" never matches
    def group(tgts, b, gs):
        st = "channel" if gs in (None, 0, -1) else "group"
        return {"format": "pack-quantized", "input_activations": None, "output_activations": None, "targets": tgts,
                "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": (None if st == "channel" else gs),
                            "num_bits": b, "observer": "memoryless_minmax", "observer_kwargs": {}, "scale_dtype": None,
                            "strategy": st, "symmetric": True, "type": "int", "zp_dtype": None}}
    int4_cats = [c for c in (INT4_CATS or []) if c in profile_cats(profile)]
    if int4_cats:
        base_targets = [t for c in profile_cats(profile) if c not in int4_cats for t in cat_targets(c, kda, mla)]
        int4_targets = [t for c in int4_cats for t in cat_targets(c, kda, mla)]
        qc["config_groups"]["group_1"] = group(base_targets, bits, group_size)
        qc["config_groups"]["group_2"] = group(int4_targets, 4, INT4_GROUP)
    else:
        qc["config_groups"]["group_1"] = group(targets, bits, group_size)
    # rebuild the ignore list: keep every Linear we did NOT quantize protected from group_0's class-name target "Linear"
    ign = [x for x in qc["ignore"] if x not in (r"re:.*self_attn\..*", r"re:.*shared_experts.*", r"re:.*layers\.(0|1|2)\.mlp\..*")]
    if profile == "aggr": ign = [x for x in ign if x != "lm_head"]
    ign += [rf"re:.*\.layers\.{lay(kda)}\.self_attn\.({'|'.join(KDA_KEEP_BF16)}|.*conv1d.*|.*norm.*|indexer.*)$"]
    if profile == "kda":
        ign += [rf"re:.*\.layers\.{lay(mla)}\.self_attn\..*"]
    else:
        ign += [rf"re:.*\.layers\.{lay(mla)}\.self_attn\.({'|'.join(MLA_KEEP_BF16)}|fused_qkv_a_proj|indexer.*|.*norm.*|wk_weights_proj)$"]
    if MTP_QUANT:
        ign = [x for x in ign if not re.search(rf"layers\\\.{mtp_layer}\\\.", x)]   # drop the base's blanket layer-45 ignores (both spellings)
        if (MTP_EXPERT_BITS, MTP_EXPERT_GROUP) != (4, 128):   # a dedicated group for the head's experts; a regex target beats group_0's class-name target
            qc["config_groups"]["group_mtp"] = group([rf"re:.*\.layers\.{mtp_layer}\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"], MTP_EXPERT_BITS, MTP_EXPERT_GROUP)
            if qc["config_groups"]["group_mtp"]["weights"]["strategy"] == "channel":
                # vLLM's 8-bit WNA16 MoE method asserts weight_quant.group_size == -1 (it reads the raw value; only the Linear scheme
                # normalises None -> -1). compressed-tensors accepts -1 for the channel strategy. Checked on 487ecf187, 2026-09-10.
                qc["config_groups"]["group_mtp"]["weights"]["group_size"] = -1
    else:
        ign += [rf"re:.*\.layers\.{mtp_layer}\..*"]
    qc["ignore"] = list(OrderedDict.fromkeys(ign))
    return qc

CATS = ["kda_in", "kda_o", "mla", "shared", "dense", "lm_head"]  # kda_in = fused q/k/v/b/f_a/g_a (one scheme in the fork)
def category(n, kda, mla):
    m = re.search(r"\.layers\.(\d+)\.", n); li = int(m.group(1)) if m else None
    if n == "lm_head.weight": return "lm_head"
    if MTP_QUANT and ".mlp.experts." in n and ".shared_experts." not in n: return "mtp_experts"
    if ".shared_experts." in n: return "shared"
    if li in (0, 1, 2) and ".mlp." in n: return "dense"
    if li in kda and ".self_attn." in n: return "kda_o" if ".o_proj." in n else "kda_in"
    if li in mla and ".self_attn." in n: return "mla"
    return None

def cat_targets(cat, kda, mla):
    lay = lambda idxs: "(" + "|".join(str(i) for i in idxs) + ")"
    return {"kda_in": [rf"re:.*\.layers\.{lay(kda)}\.self_attn\.({'|'.join(KDA_IN)})$", rf"re:.*\.layers\.{lay(kda)}\.self_attn\.in_proj_qkvbfg_a$"],
            "kda_o": [rf"re:.*\.layers\.{lay(kda)}\.self_attn\.o_proj$"],
            "mla": [rf"re:.*\.layers\.{lay(mla)}\.self_attn\.({'|'.join(MLA_BIG)})$"],
            "shared": [r"re:.*\.shared_experts\.(gate_proj|up_proj|down_proj|gate_up_proj)$"],
            "dense": [r"re:.*\.layers\.(0|1|2)\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)$"],
            "lm_head": ["re:.*lm_head$"]}[cat]

def qargs(bits, group_size):
    if group_size in (None, 0, -1):
        return QuantizationArgs(num_bits=bits, type="int", symmetric=True, strategy="channel")
    return QuantizationArgs(num_bits=bits, type="int", symmetric=True, strategy="group", group_size=group_size)

def quantize_weight(w: torch.Tensor, args: QuantizationArgs):
    """w: [out, in] -> (weight_packed int32 [out, ceil(in*bits/32)], weight_scale bf16, weight_shape int64[2])"""
    w32 = w.to(torch.float32)
    out_f, in_f = w32.shape
    if str(getattr(args.strategy, "value", args.strategy)) == "channel":
        mn, mx = w32.amin(dim=1, keepdim=True), w32.amax(dim=1, keepdim=True)
        scale, zp = calculate_qparams(mn, mx, args)
    else:
        g = args.group_size; assert in_f % g == 0, f"in_features {in_f} not divisible by group {g}"
        wg = w32.reshape(out_f, in_f // g, g)
        scale, zp = calculate_qparams(wg.amin(dim=2), wg.amax(dim=2), args)   # [out, in/g]
    q = quantize(w32, scale, zp, args, dtype=torch.int8)
    packed = pack_to_int32(q, args.num_bits)
    return packed, scale.to(torch.bfloat16), torch.tensor([out_f, in_f], dtype=torch.int64), q, scale


# ----------------------------------------------------------------------------- calibrated (GPTQ) path: Hessians from calib/calibrate.py
CALIB_DIR = None; CALIB_ALL = False; _HCACHE = {}; CALIB_STATS = {"gptq": 0, "rtn_fallback": 0}
def find_hessian(tensor_name):
    """Map a checkpoint tensor to the vLLM module whose input Hessian was saved (fused KDA in-proj, fused gate_up, etc.)."""
    if CALIB_DIR is None: return None
    m = re.search(r"\.layers\.(\d+)\.", tensor_name); li = m.group(1) if m else None; base = tensor_name[:-len(".weight")]
    leaf = base.split(".")[-1]
    if tensor_name == "lm_head.weight": cands = ["lm_head"]
    elif ".shared_experts." in base: cands = [f"layers.{li}.mlp.shared_experts.{'gate_up_proj' if leaf in ('gate_proj','up_proj') else leaf}", f"layers.{li}.mlp.shared_experts.{leaf}"]
    elif ".mlp." in base: cands = [f"layers.{li}.mlp.{'gate_up_proj' if leaf in ('gate_proj','up_proj') else leaf}", f"layers.{li}.mlp.{leaf}"]
    elif leaf in KDA_IN: cands = [f"layers.{li}.self_attn.in_proj_qkvbfg_a", f"layers.{li}.self_attn.{leaf}"]
    else: cands = [f"layers.{li}.self_attn.{leaf}"]
    for c in cands:
        if c in _HCACHE: return _HCACHE[c]
        for f in os.listdir(CALIB_DIR):
            if f.endswith(".pt") and (f[:-3].endswith("." + c) or f[:-3] == c):
                _HCACHE[c] = torch.load(os.path.join(CALIB_DIR, f), map_location="cpu")["H"]; return _HCACHE[c]
    return None

def quantize_weight_gptq(w, H, bits, group_size):
    """GPTQ-quantized q/scales packed exactly like quantize_weight (symmetric, group or channel)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gptq", next((c for c in (os.environ.get("GPTQ_PY", ""), os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib", "gptq.py")) if c and os.path.exists(c)), os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib", "gptq.py"))); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    out_f, in_f = w.shape; gs = in_f if group_size in (None, 0, -1) else group_size
    if H.shape[0] != in_f: return None
    q, scales = g.gptq_quantize(w.to(torch.float32), H, bits=bits, group_size=gs)
    packed = pack_to_int32(q.to(torch.int8), bits)
    return packed, scales.to(torch.bfloat16), torch.tensor([out_f, in_f], dtype=torch.int64), q, scales

def selftest(bits=8, group_size=None):
    torch.manual_seed(0); args = qargs(bits, group_size)
    w = (torch.randn(256, 512) * 0.02).to(torch.bfloat16)
    packed, scale_bf16, shape, q, scale = quantize_weight(w, args)
    q2 = unpack_from_int32(packed, bits, torch.Size(shape.tolist()))
    assert torch.equal(q.to(torch.int8), q2), "pack/unpack mismatch"
    deq = dequantize(q2, scale, torch.zeros_like(scale), args)
    err = (deq - w.float()).abs()
    print(f"selftest bits={bits} group={group_size}: packed {tuple(packed.shape)} {packed.dtype}, scale {tuple(scale_bf16.shape)}, "
          f"max|err| {err.max():.3e} (half-step {(scale.max()/2).item():.3e}), mean|err| {err.mean():.3e}, rel {err.mean()/w.float().abs().mean():.3%}")
    assert err.max() <= scale.max() / 2 + 1e-6
    print("OK")

def load_index(src):
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = idx["weight_map"]; shards = OrderedDict()
    for n, f in wm.items(): shards.setdefault(f, []).append(n)
    return wm, shards

def plan(a):
    if a.index_json:
        ti = json.load(open(a.index_json)); names = list(ti.keys()); shards = {"(index)": names}
        class _Sl:
            def __init__(self, sh): self.sh = sh
            def get_shape(self): return self.sh
        class _SF:
            def __init__(self): pass
            def __enter__(self): return self
            def __exit__(self, *x): pass
            def get_slice(self, n): return _Sl(ti[n]["shape"])
        safe_open_ = lambda path, fmt: _SF()
    else:
        wm, shards = load_index(a.src); names = list(wm.keys()); safe_open_ = safe_open
    mtp = a.mtp_layer
    tq, kda, mla = build_plan(names, a.profile, mtp)
    print(f"layers: kda={len(kda)} {kda[:6]}... mla={len(mla)} {mla} mtp={mtp}")
    tot = 0; cats = {}
    for f, ns in shards.items():
        with safe_open_(os.path.join(a.src or ".", f), "pt") as sf:
            for n in ns:
                if n in tq:
                    sh = sf.get_slice(n).get_shape(); b = 2 * sh[0] * sh[1]; tot += b
                    c = "lm_head" if n == "lm_head.weight" else ("mtp_experts" if (".mlp.experts." in n and "shared_experts" not in n) else ("shared_experts" if "shared_experts" in n else ("dense_mlp" if ".mlp." in n else ("mla_attn" if int(re.search(r"layers\.(\d+)", n).group(1)) in mla else "kda_attn"))))
                    cats[c] = cats.get(c, 0) + b
    print(f"tensors to quantize: {len(tq)}; BF16 bytes {tot/1e9:.2f} GB -> ~{tot/1e9*a.bits/16:.2f} GB at {a.bits} bit (saves {tot/1e9*(1-a.bits/16):.2f} GB per full pass)")
    for c, b in sorted(cats.items(), key=lambda kv: -kv[1]): print(f"  {c:16s} {b/1e9:6.2f} GB")
    cfg_path = a.config_json or os.path.join(a.src, "config.json")
    qc = new_quant_config(json.load(open(cfg_path))["quantization_config"], a.profile, a.bits, a.group_size, kda, mla, mtp)
    from compressed_tensors.quantization import QuantizationConfig
    QuantizationConfig.model_validate(qc); print("quantization_config validates with compressed-tensors", __import__("compressed_tensors").__version__)
    print("new group_1 targets:"); [print("   ", t) for t in qc["config_groups"]["group_1"]["targets"]]
    print("ignore:"); [print("   ", t) for t in qc["ignore"]]
    return tq, kda, mla, qc

def run(a):
    os.makedirs(a.dst, exist_ok=True)
    tq, kda, mla, qc = plan(a)
    args = qargs(a.bits, a.group_size); args4 = qargs(4, INT4_GROUP); int4 = set(INT4_CATS or [])
    wm, shards = load_index(a.src)
    out_map = {}; buf = OrderedDict(); buf_bytes = 0; shard_no = 0; written = []
    limit = int(a.shard_gb * 1e9)
    def flush():
        nonlocal buf, buf_bytes, shard_no
        if not buf: return
        shard_no += 1; fn = f"model-{shard_no:05d}.safetensors"
        save_file(buf, os.path.join(a.dst, fn), metadata={"format": "pt"})
        for k in buf: out_map[k] = fn
        written.append(fn); print(f"  wrote {fn} ({buf_bytes/1e9:.2f} GB, {len(buf)} tensors)", flush=True)
        buf = OrderedDict(); buf_bytes = 0
    t0 = time.time(); nq = 0; saved = 0
    for f, ns in shards.items():
        print(f"shard {f}: {len(ns)} tensors", flush=True)
        with safe_open(os.path.join(a.src, f), "pt") as sf:
            for n in ns:
                t = sf.get_tensor(n)
                if n in tq:
                    cat_ = category(n, kda, mla); use4 = cat_ in int4; a_ = qargs(MTP_EXPERT_BITS, MTP_EXPERT_GROUP) if cat_ == "mtp_experts" else (args4 if use4 else args)
                    res = None
                    if CALIB_DIR is not None and (use4 or CALIB_ALL):
                        H = find_hessian(n); res = quantize_weight_gptq(t, H, a_.num_bits, a_.group_size if str(getattr(a_.strategy, "value", a_.strategy)) == "group" else -1) if H is not None else None
                        CALIB_STATS["gptq" if res is not None else "rtn_fallback"] += 1
                    packed, scale, shape, _, _ = res if res is not None else quantize_weight(t, a_)
                    base = n[:-len(".weight")]
                    items = [(base + ".weight_packed", packed), (base + ".weight_scale", scale), (base + ".weight_shape", shape)]
                    nq += 1; saved += t.numel() * 2 - packed.numel() * 4 - scale.numel() * 2
                else:
                    items = [(n, t)]
                for k, v in items:
                    buf[k] = v.contiguous(); buf_bytes += v.numel() * v.element_size()
                if buf_bytes >= limit: flush()
    flush()
    # index + config + side files
    total = sum(os.path.getsize(os.path.join(a.dst, fn)) for fn in written)
    json.dump({"metadata": {"total_size": total}, "weight_map": out_map}, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)
    cfg = json.load(open(os.path.join(a.src, "config.json"))); cfg["quantization_config"] = qc
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
    for side in os.listdir(a.src):
        if side.endswith((".json", ".jinja", ".yaml", ".md", ".txt")) and side not in ("config.json", "model.safetensors.index.json"):
            import shutil; shutil.copy2(os.path.join(a.src, side), os.path.join(a.dst, side))
    json.dump({"tool": "requant_remainder.py", "profile": a.profile, "bits": a.bits, "group_size": a.group_size, "int4_cats": sorted(INT4_CATS or []), "int4_group": INT4_GROUP, "quantized_tensors": sorted(tq),
               "src": a.src, "seconds": round(time.time() - t0, 1)}, open(os.path.join(a.dst, "requant_info.json"), "w"), indent=1)
    print(f"done: {nq} tensors quantized, {saved/1e9:.2f} GB saved, output {total/1e9:.2f} GB in {len(written)} shards, {time.time()-t0:.0f} s" + (f"; calib: {CALIB_STATS}" if CALIB_DIR else ""))

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["selftest", "plan", "run"])
    p.add_argument("--src"); p.add_argument("--dst"); p.add_argument("--profile", default="kda", choices=["kda", "kda_mla", "aggr"])
    p.add_argument("--bits", type=int, default=8, choices=[4, 8]); p.add_argument("--group-size", type=int, default=None, help="None/-1 = per-channel (default for INT8); 128 recommended for INT4")
    p.add_argument("--shard-gb", type=float, default=8.0); p.add_argument("--mtp-layer", type=int, default=45)
    p.add_argument("--int4-cats", default=None, help="comma list of categories to quantize at INT4 (group --int4-group) inside an INT8 profile: " + ",".join(CATS))
    p.add_argument("--int4-group", type=int, default=32)
    p.add_argument("--mtp-expert-bits", type=int, default=4, help="bits for the MTP head's routed experts (4 = group_0's format; 8 = INT8 in a dedicated config group)")
    p.add_argument("--mtp-expert-group", type=int, default=128)
    p.add_argument("--mtp-quant", action="store_true", help="also quantize the MTP head (layer --mtp-layer): routed experts INT4 g128 (the base's format), attention/shared with the profile's scheme; norms, gate, indexer, eh_proj, shared_head stay BF16")
    p.add_argument("--calib-dir", default=None, help="directory of per-module Hessians from calib/calibrate.py: GPTQ instead of RTN for INT4 categories")
    p.add_argument("--calib-all", action="store_true", help="also GPTQ the INT8 tensors")
    p.add_argument("--index-json", help="plan only: tensor index {name:{dtype,shape,bytes}} (from shard headers) instead of local shards")
    p.add_argument("--config-json", help="plan only: path to config.json when --src is not available")
    a = p.parse_args()
    global INT4_CATS, INT4_GROUP, MTP_QUANT
    INT4_CATS = [c for c in a.int4_cats.split(',') if c] if a.int4_cats else None; INT4_GROUP = a.int4_group; MTP_QUANT = bool(a.mtp_quant)
    global MTP_EXPERT_BITS, MTP_EXPERT_GROUP; MTP_EXPERT_BITS = a.mtp_expert_bits; MTP_EXPERT_GROUP = a.mtp_expert_group
    global CALIB_DIR, CALIB_ALL
    CALIB_DIR = a.calib_dir; CALIB_ALL = a.calib_all
    if INT4_CATS and (bad := [c for c in INT4_CATS if c not in CATS]): sys.exit(f'unknown --int4-cats {bad}')
    if a.group_size in (0, -1): a.group_size = None
    if a.cmd == "selftest": selftest(a.bits, a.group_size); selftest(4, 128); return
    if a.cmd == "plan":
        if not (a.src or a.index_json): sys.exit("--src or --index-json required")
        plan(a); return
    if not a.src: sys.exit("--src required")
    else:
        if not a.dst: sys.exit("--dst required")
        run(a)

if __name__ == "__main__":
    main()
