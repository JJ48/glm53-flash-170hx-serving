#!/usr/bin/env python3
"""Fork patch 0018 (JKeys, 2026-09-15): route the MoE Marlin GEMM to a standalone kernel build.

Env-gated and inert by default. With VLLM_MARLIN_DEV_SO=<path to _marlin_dev.so> the server loads the standalone
extension (upstream Marlin MoE 8e92248f79 built with kernels/marlin/build.sh, same op schema) and replaces
ops.moe_wna16_marlin_gemm with a wrapper that calls torch.ops._marlin_dev.moe_wna16_marlin_gemm. The standalone ops.cu
honours MARLIN_DEV_STAGES (pipeline depth; 6 and 8 are instantiated for the 170HX decode config) so a kernel variant
can be A/B-tested end to end without touching the fork's own extension. Outputs are bit-identical for every stage
count (measured 2026-09-15). Anchors on the patch-0016 block in experts/marlin_moe.py.
Usage: marlin_dev_patch.py check|apply|revert [--root DIST_PACKAGES]"""
import os, sys
ROOT = next((a.split("=")[1] for a in sys.argv if a.startswith("--root=")), "/usr/local/lib/python3.12/dist-packages")
F = os.path.join(ROOT, "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py")
ANCHOR = '_MARLIN_TILE16 = {k: int(_os16.environ.get(e, "-1") or -1) for k, e in (("thread_k", "VLLM_MARLIN_MOE_TK"), ("thread_n", "VLLM_MARLIN_MOE_TN"), ("blocks_per_sm", "VLLM_MARLIN_MOE_BPS"))}\n'
MARK = "# patch 0018 marlin_dev"
BLOCK = f'''{MARK}: VLLM_MARLIN_DEV_SO routes the MoE Marlin GEMM to a standalone build (MARLIN_DEV_STAGES = pipeline depth)
if _os16.environ.get("VLLM_MARLIN_DEV_SO"):
    torch.ops.load_library(_os16.environ["VLLM_MARLIN_DEV_SO"])
    def _marlin_dev_moe_wna16_marlin_gemm(input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros, g_idx, perm,
                                          workspace, sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights, moe_block_size,
                                          top_k, mul_topk_weights, b_q_type, size_m, size_n, size_k, is_k_full, use_atomic_add,
                                          use_fp32_reduce, is_zp_float, thread_k=-1, thread_n=-1, blocks_per_sm=-1):
        return torch.ops._marlin_dev.moe_wna16_marlin_gemm(input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros,
                                                           g_idx, perm, workspace, sorted_token_ids, expert_ids, num_tokens_past_padded,
                                                           topk_weights, moe_block_size, top_k, mul_topk_weights, b_q_type.id, size_m,
                                                           size_n, size_k, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float,
                                                           thread_k, thread_n, blocks_per_sm)
    ops.moe_wna16_marlin_gemm = _marlin_dev_moe_wna16_marlin_gemm
'''
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    s = open(F).read(); applied = MARK in s
    if mode == "check":
        print(f"0018 marlin_dev: {'applied' if applied else 'not applied'} ({F}); anchor {'present' if ANCHOR in s else 'MISSING'}"); return 0
    if mode == "apply":
        if applied: print("0018: already applied"); return 0
        if s.count(ANCHOR) != 1: print(f"0018: anchor count {s.count(ANCHOR)} != 1, refusing"); return 2
        open(F, "w").write(s.replace(ANCHOR, ANCHOR + BLOCK)); print("0018: applied"); return 0
    if mode == "revert":
        if not applied: print("0018: not applied"); return 0
        i = s.index(MARK); j = s.index("    ops.moe_wna16_marlin_gemm = _marlin_dev_moe_wna16_marlin_gemm\n", i) + len("    ops.moe_wna16_marlin_gemm = _marlin_dev_moe_wna16_marlin_gemm\n")
        open(F, "w").write(s[:i] + s[j:]); print("0018: reverted"); return 0
    print("usage: check|apply|revert"); return 1
if __name__ == "__main__": sys.exit(main())
