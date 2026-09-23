#!/usr/bin/env python3
"""Debug patch (env-gated VLLM_MOE_DEBUG_EXPERTS=1, default off): rate-limited print of the number of DISTINCT experts
touched per MoE call in the Marlin MoE path (experts/marlin_moe.py fused_marlin_moe). Decides whether the Marlin MoE
GEMM has headroom: bytes/step = sum over layers of distinct experts x expert slab, against the measured kernel time.
Needs an eager boot (--enforce-eager): under CUDA graphs the Python call does not run per step. Usage: check|apply|revert"""
import os, shutil, sys, py_compile
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
BAK = P + ".bakmoeaudit"; TAG = "moe audit dbg"
OLD = "\n    topk = topk_ids.size(1)\n"
NEW = '''
    topk = topk_ids.size(1)
    if _MOE_AUDIT[0]:  # moe audit dbg (env-gated, rate-limited, never during capture)
        _MOE_AUDIT[1] += 1
        if _MOE_AUDIT[1] % 50 == 0 and not torch.cuda.is_current_stream_capturing():
            print(f"[MOE_EXPERTS n={_MOE_AUDIT[1]} tokens={topk_ids.shape[0]} topk={topk} distinct={int(topk_ids.unique().numel())} E={global_num_experts}]", flush=True)
'''
HEAD_OLD = "def fused_marlin_moe(\n"
HEAD_NEW = 'import os as _os_ma  # moe audit dbg\n_MOE_AUDIT = [_os_ma.environ.get("VLLM_MOE_DEBUG_EXPERTS", "0") == "1", 0]\n\n\ndef fused_marlin_moe(\n'
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
s = open(P).read()
if MODE == "check": print("patched" if TAG in s else "unpatched"); sys.exit(0)
if MODE == "revert":
    if os.path.exists(BAK): shutil.copy2(BAK, P); print("reverted")
    sys.exit(0)
if MODE == "apply":
    if TAG in s: print("already patched"); sys.exit(0)
    assert s.count(OLD) == 1 and s.count(HEAD_OLD) == 1, (s.count(OLD), s.count(HEAD_OLD))
    if not os.path.exists(BAK): shutil.copy2(P, BAK)
    s = s.replace(HEAD_OLD, HEAD_NEW).replace(OLD, NEW); open(P, "w").write(s); py_compile.compile(P, doraise=True); print("applied"); sys.exit(0)
