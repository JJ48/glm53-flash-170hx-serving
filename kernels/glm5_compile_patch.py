#!/usr/bin/env python3
"""Fork patch 0014: opt-in torch.compile (inductor fusion) for Glm5Next — the "fusing" probe.

The fork auto-enables VLLM_USE_BREAKABLE_CUDAGRAPH for Glm5Next (config/vllm.py:1321-1346), which forces compilation
mode NONE: every small kernel between the custom ops (elementwise, norms, activations, copies: ~4.4 ms of the 42.5 ms
step at k=7) runs unfused. Upstream Qwen3-Next compiles because its model class carries @support_torch_compile and
its GDN core is a custom op listed in CompilationConfig._attention_ops (a splitting op: runs eagerly between
piecewise graphs). This patch gives Glm5Next the same shape, all behind VLLM_GLM5_COMPILE=1 (default off):
  kda.py          registers torch.ops.vllm.glm5next_kda_core (mutates core_attn_out, fake impl no-op) that resolves
                  the layer through forward_context.no_compile_layers[prefix] and runs the existing _forward; the
                  layer's forward routes through it only under the flag (else the eager-break call as before)
  compilation.py  adds "vllm::glm5next_kda_core" to _attention_ops (default splitting op). The other op this model
                  ran through the eager-break mechanism, "vllm::sparse_attn_indexer_kpool", is already in that list
                  in this fork (compilation.py:776), so it stays outside the captured graphs without a change.
  model.py        applies support_torch_compile(Glm5NextModel) under the flag; hoists the aux-dump env read out of
                  forward (dynamo-safe constant); pre-computes the indexer's lazy fp32 head-gate after load_weights
                  under the flag so no module attribute is assigned inside the compiled forward
Boot with: VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_GLM5_COMPILE=1 (optimization level default O2 -> VLLM_COMPILE).
Usage: glm5_compile_patch.py check|apply|revert"""
import os, shutil, sys, py_compile
V = "/usr/local/lib/python3.12/dist-packages/vllm"
FILES = {
    "kda": f"{V}/models/glm5next/nvidia/kda.py",
    "model": f"{V}/models/glm5next/nvidia/model.py",
    "comp": f"{V}/config/compilation.py",
}
TAG = "patch 0014"
EDITS = {  # file -> list of (old, new, expected_count)
    "kda": [
        ("class Glm5NextLinearAttention(GatedDeltaNetAttention):",
         '''# ---- patch 0014: KDA core as a custom op for torch.compile (opt-in, VLLM_GLM5_COMPILE=1) ----
import os as _os14
from vllm.utils.torch_utils import direct_register_custom_op as _drco14
_GLM5_COMPILE14 = _os14.environ.get("VLLM_GLM5_COMPILE", "0") == "1"


def glm5next_kda_core(
    qkv_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    self = get_forward_context().no_compile_layers[layer_name]
    self._forward(
        qkv_proj_states=qkv_proj_states, g1=g1, beta=beta, core_attn_out=core_attn_out
    )


def glm5next_kda_core_fake(
    qkv_proj_states: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


_drco14(
    op_name="glm5next_kda_core",
    op_func=glm5next_kda_core,
    mutates_args=["core_attn_out"],
    fake_impl=glm5next_kda_core_fake,
)


class Glm5NextLinearAttention(GatedDeltaNetAttention):''', 1),
        ('''        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
        )
''',
         '''        if _GLM5_COMPILE14:  # patch 0014: splitting-op boundary under torch.compile
            torch.ops.vllm.glm5next_kda_core(qkv, g1, beta, core_attn_out, self.prefix)
        else:
            self._forward(
                qkv_proj_states=qkv,
                g1=g1,
                beta=beta,
                core_attn_out=core_attn_out,
            )
''', 1),
    ],
    "comp": [
        ('        "vllm::qwen_gdn_attention_core",\n',
         '        "vllm::qwen_gdn_attention_core",\n        "vllm::glm5next_kda_core",  # patch 0014\n', 1),
    ],
    "model": [
        ("class Glm5NextMLP(nn.Module):",
         '''# ---- patch 0014: env reads hoisted to import time (dynamo-safe constants) ----
import os as _os14
_AUX_DUMP_DIR14 = _os14.environ.get("VLLM_DUMP_AUX_DIR")
_GLM5_COMPILE = _os14.environ.get("VLLM_GLM5_COMPILE", "0") == "1"


class Glm5NextMLP(nn.Module):''', 1),
        ('''                import os as _os
                _dump = _os.environ.get("VLLM_DUMP_AUX_DIR")
''',
         '''                _os = _os14  # patch 0014
                _dump = _AUX_DUMP_DIR14
''', 1),
        ("class Glm5NextForCausalLM(",
         '''if _GLM5_COMPILE:  # patch 0014: opt-in torch.compile for the target model
    from vllm.compilation.decorators import support_torch_compile as _stc14

    Glm5NextModel = _stc14(Glm5NextModel)


class Glm5NextForCausalLM(''', 1),
        ('''            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
''',
         '''            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        _loaded14 = loader.load_weights(weights)
        if _GLM5_COMPILE:  # patch 0014: pre-compute the indexer's lazy fp32 head-gate (no setattr in compiled forward)
            from .attention import Indexer as _Indexer14

            for _m in self.modules():
                if isinstance(_m, _Indexer14) and getattr(_m, "_wp_fp32", None) is None:
                    _m._wp_fp32 = (
                        _m.wk_weights_proj.weight.data[_m.head_dim :, :].t().contiguous().float()
                    )
        return _loaded14
''', 1),
    ],
}
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
if MODE == "check":
    print(" ".join(f"{k}:{'patched' if TAG in open(p).read() else 'unpatched'}" for k, p in FILES.items())); sys.exit(0)
if MODE == "revert":
    for k, p in FILES.items():
        b = p + ".bak0014"
        if os.path.exists(b): shutil.copy2(b, p); print("reverted", k)
    sys.exit(0)
if MODE == "apply":
    for k, p in FILES.items():
        s = open(p).read()
        if TAG in s: print("already patched:", k); continue
        for old, new, want in EDITS[k]:
            assert s.count(old) == want, f"{k}: expected {want} of {old[:60]!r}, found {s.count(old)}"
        for old, new, _ in EDITS[k]: s = s.replace(old, new)
        if not os.path.exists(p + ".bak0014"): shutil.copy2(p, p + ".bak0014")
        open(p, "w").write(s); py_compile.compile(p, doraise=True); print("applied:", k)
    sys.exit(0)
print("usage: check|apply|revert"); sys.exit(2)
