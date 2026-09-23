#!/bin/bash
# Apply the complete production patch stack to the club-170hx vLLM image, in production's order.
#   base image: ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905
#   run inside the container as root:  bash kernels/apply_stack.sh [--upto PATCH_NAME]
# Apart from the loader fixes 0001/0002 (always on), each patch is inert until its flag is set (serving/PATCH_STACK.md
# lists the flags production sets).
# The order matters: later patches were written against the tree with every earlier patch applied, so apply the
# whole sequence even where you leave a patch switched off. Idempotent: re-running reports "already patched".
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd)
V=${VLLM_DIR:-/usr/local/lib/python3.12/dist-packages/vllm}
UPTO=""; [ "${1:-}" = "--upto" ] && UPTO=${2:?--upto needs a patch name}
say(){ echo "[stack] $*"; }
say "0001/0002 quant config on MLA/KDA projections: $(python3 "$K/quant_config_patch.py" apply | tr '\n' ' ')"
say "0003 per-request k + rolling-acceptance wiring:"; python3 "$K/apply_spec_k.py" --files "$K/files" | sed 's/^/        /'
cp "$K/spec_roll.py" "$V/v1/core/sched/spec_roll.py"; say "spec_roll v4 (rolling-acceptance rule) installed"
ORDER="dflash2_conf_patch gate_async_patch dynsd_mtp_patch gate_sync_patch mla_split_patch pp_cadence_patch fla_rec_patch
glm5_compile_patch moe_audit_patch pp_packed_patch marlin_cfg_patch pp_hop_v2_patch marlin_dev_patch kda_strided_patch
pp_draft_event_patch topp_triton_rows_patch spec_roll_solo_patch aot_mm_key_patch mm_budget_cap_patch wp_precompute_fix_patch"
for s in $ORDER; do
  say "$s: $(python3 "$K/$s.py" apply 2>&1 | tail -3 | tr '\n' ' ')"
  [ "$s" = "$UPTO" ] && { say "stopped after $s (--upto)"; break; }
done
say "done. Set the flags from serving/PATCH_STACK.md (serving/launch.example.sh has them); VLLM_GLM5_COMPILE=1 requires VLLM_GLM5_WP_FIX=1."
