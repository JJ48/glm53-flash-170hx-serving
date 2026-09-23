#!/usr/bin/env python3
"""Fork patch 0013: launch-config knobs for the fla recurrent gated-delta-rule kernel (the linear-attention decode kernel).

Both launchers in third_party/flash_linear_attention/ops/fused_recurrent.py hard-code BV = min(next_pow2(V), 32),
num_warps = 1, num_stages = 3, giving grid (NK, NV, N*HV) one-warp programs that each hold a [BV, BK] fp32 state block
(32 x 128 x 4 B = 16 KB per warp = 128 registers per thread for the state alone) and loop over the T verified tokens.
At batch 1 on the 70-SM CMP 170HX the k=7 profile shows 37 us per launch against ~11 us of HBM traffic: latency-bound,
1.19 ms/step model-wide. Env-gated, default = upstream:
  VLLM_FLA_REC_WARPS=N   num_warps (default 1)
  VLLM_FLA_REC_BV=N      max BV (default 32; power of 2)
  VLLM_FLA_REC_STAGES=N  num_stages (default 3)
Usage: fla_rec_patch.py check|apply|revert"""
import os, shutil, sys, py_compile
P = "/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py"
BAK = P + ".bak0013"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
TAG = "patch 0013"
ENV = 'int(_os.environ.get("VLLM_FLA_REC_{}", "{}") or {})'
REPL = [  # (exact line, replacement lines, expected count)
    ("    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)",
     "    import os as _os  # patch 0013: recurrent-kernel launch knobs\n"
     "    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), " + ENV.format("BV", 32, 32) + ")", 1),
    ("    BV = min(triton.next_power_of_2(V), 32)",
     "    import os as _os  # patch 0013: recurrent-kernel launch knobs\n"
     "    BV = min(triton.next_power_of_2(V), " + ENV.format("BV", 32, 32) + ")", 1),
    ("    num_stages = 3", "    num_stages = " + ENV.format("STAGES", 3, 3), 2),
    ("    num_warps = 1", "    num_warps = " + ENV.format("WARPS", 1, 1), 2),
]
s = open(P).read()
patched = TAG in s
if MODE == "check":
    print("patched" if patched else "unpatched"); sys.exit(0)
if MODE == "revert":
    if not os.path.exists(BAK): print("no backup; nothing to revert"); sys.exit(1)
    shutil.copy2(BAK, P); print("reverted from", BAK); sys.exit(0)
if MODE == "apply":
    if patched: print("already patched"); sys.exit(0)
    lines = s.split("\n"); out = []; counts = {old: 0 for old, _, _ in REPL}
    for ln in lines:
        for old, new, _ in REPL:
            if ln == old:
                counts[old] += 1; ln = new; break
        out.append(ln)
    for old, _, want in REPL:
        assert counts[old] == want, f"expected {want} of {old!r}, found {counts[old]}"
    if not os.path.exists(BAK): shutil.copy2(P, BAK)
    open(P, "w").write("\n".join(out)); py_compile.compile(P, doraise=True)
    print("applied; backup", BAK); sys.exit(0)
print("usage: check|apply|revert"); sys.exit(2)
