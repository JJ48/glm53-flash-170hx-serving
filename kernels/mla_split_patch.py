#!/usr/bin/env python3
"""Fork patch 0011: make the sparse-MLA KV-split heuristic fit a 70-SM part.

`_choose_num_kv_splits` (triton_mla_sparse_kernel.py) returns 1 (single-pass kernel) once
num_tokens * num_head_groups * _SPLIT_MAX_OCCUPANCY(4) >= sm_count. GLM-5.3-Flash has 64 query heads -> 4 head groups,
and the CMP 170HX has 70 SMs, so the single-pass path is taken from 5 verified tokens up: 4 tokens run 64 split programs
(91% of SMs), 5 tokens run 20 single-pass programs (29%). That is the measured "5-token cliff": +0.33 ms per MLA layer,
11 MLA layers, +3.6 ms/step. At 8 tokens (k=7, production) attention runs 32 programs on 70 SMs (46%).

Env-gated, default = upstream behaviour:
  VLLM_MLA_SPARSE_SPLIT_MODE=upstream  (default) unchanged
  VLLM_MLA_SPARSE_SPLIT_MODE=fill      smallest power-of-2 split count with baseline*splits >= sm_count, capped by the
                                       per-split-work floor and top-k divisibility exactly as upstream
  VLLM_MLA_SPARSE_KV_SPLITS=N          force N (debug)
Usage: mla_split_patch.py check|apply|revert"""
import os, shutil, sys
P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_mla_sparse_kernel.py"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
TAG = "patch 0011"
OLD = '''    baseline = num_tokens * num_head_groups
    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
        return 1
'''
NEW = '''    baseline = num_tokens * num_head_groups
    import os as _os  # patch 0011: split heuristic for small-SM parts (70-SM CMP 170HX)
    _force = int(_os.environ.get("VLLM_MLA_SPARSE_KV_SPLITS", "0") or 0)
    if _force > 0:
        _f = _force
        while _f > 1 and index_topk % _f != 0:
            _f //= 2
        return max(1, _f)
    if _os.environ.get("VLLM_MLA_SPARSE_SPLIT_MODE", "upstream") == "fill":
        if baseline == 0 or baseline >= sm_count:
            return 1
        _ideal = triton.next_power_of_2(max(1, index_topk // _MIN_TOPK_PER_SPLIT))
        _s = 1
        while baseline * _s < sm_count and _s < _ideal:
            _s *= 2
        while _s > 1 and index_topk % _s != 0:
            _s //= 2
        return max(1, _s)
    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
        return 1
'''
s = open(P).read()
if MODE == "check":
    print("patched" if TAG in s else "unpatched", "| anchor:", s.count(OLD)); sys.exit(0)
if MODE == "apply":
    if TAG in s: print("already patched"); sys.exit(0)
    assert s.count(OLD) == 1, f"anchor count {s.count(OLD)}"
    if not os.path.exists(P + ".mlasplit.bak"): shutil.copy2(P, P + ".mlasplit.bak")
    open(P, "w").write(s.replace(OLD, NEW)); print("applied")
elif MODE == "revert":
    b = P + ".mlasplit.bak"
    if os.path.exists(b): shutil.copy2(b, P); os.remove(b); print("reverted")
    else: print("no backup")
