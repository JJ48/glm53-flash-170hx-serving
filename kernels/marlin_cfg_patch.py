#!/usr/bin/env python3
"""Fork patch 0016 (env-gated, default = upstream): Marlin MoE tile knobs from the environment.
VLLM_MARLIN_MOE_TK / VLLM_MARLIN_MOE_TN / VLLM_MARLIN_MOE_BPS (ints, -1 = auto) are injected into both
ops.moe_wna16_marlin_gemm calls of experts/marlin_moe.py (thread_k, thread_n, blocks_per_sm). Usage: check|apply|revert"""
import os, shutil, sys, py_compile
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
BAK = P + ".bak0016"; TAG = "patch 0016"
OLD = "        use_fp32_reduce=True,\n        is_zp_float=False,\n    )\n"
NEW = "        use_fp32_reduce=True,\n        is_zp_float=False,\n        **_MARLIN_TILE16,  # patch 0016\n    )\n"
HEAD_OLD = "def _fused_marlin_moe(\n"
HEAD_NEW = ('import os as _os16  # patch 0016\n_MARLIN_TILE16 = {k: int(_os16.environ.get(e, "-1") or -1) for k, e in (("thread_k", "VLLM_MARLIN_MOE_TK"), '
            '("thread_n", "VLLM_MARLIN_MOE_TN"), ("blocks_per_sm", "VLLM_MARLIN_MOE_BPS"))}\n\n\ndef _fused_marlin_moe(\n')
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"; s = open(P).read()
if MODE == "check": print("patched" if TAG in s else "unpatched"); sys.exit(0)
if MODE == "revert":
    if os.path.exists(BAK): shutil.copy2(BAK, P); print("reverted")
    sys.exit(0)
if MODE == "apply":
    if TAG in s: print("already patched"); sys.exit(0)
    assert s.count(OLD) == 2 and s.count(HEAD_OLD) == 1, (s.count(OLD), s.count(HEAD_OLD))
    if not os.path.exists(BAK): shutil.copy2(P, BAK)
    s = s.replace(HEAD_OLD, HEAD_NEW).replace(OLD, NEW); open(P, "w").write(s); py_compile.compile(P, doraise=True); print("applied"); sys.exit(0)
