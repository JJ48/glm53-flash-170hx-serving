#!/usr/bin/env bash
# Launch GLM-5.3-Flash the way the results in this repo were measured: 4x CMP 170HX, pipeline-parallel PP4, a 7-token
# DFlash2 drafter with the variable/rolling acceptance rule, and the full patch stack from ../kernels/ applied inside
# the club-170hx image (bash kernels/apply_stack.sh; see serving/PATCH_STACK.md). Adjust paths, the layer split and the
# step-time table to your hardware.
set -euo pipefail

MODEL=${MODEL:-/path/to/glm5.3-flash-w8w4}          # the quantized model (see serving/QUANT_RECIPE.md)
DRAFTER=${DRAFTER:-/path/to/your-dflash2-drafter}   # you supply this; the benchmark drafter is not redistributable
PORT=${PORT:-8000}

# --- patch stack flags (serving/PATCH_STACK.md); every patch is inert without its flag ---
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_GLM5_COMPILE=1   # 0014: torch.compile for GLM-5.3-Flash
export VLLM_GLM5_WP_FIX=1                                   # 0038: REQUIRED whenever 0014 is on (long-prompt fix)
export VLLM_PP_PACKED_HOP=1                                 # 0015: one NCCL op per pipeline hop
export VLLM_MLA_SPARSE_SPLIT_MODE=fill                      # 0011: sparse-MLA split heuristic for 70-SM parts
export VLLM_KDA_STRIDED=1                                   # 0021: KDA recurrence without copies
export VLLM_PP_SPLIT_DRAFT_EVENT=1                          # 0022: overlap next-step prep with the drafter
export VLLM_TOPP_TRITON_MIN_ROWS=2                          # 0025: one-launch Triton top-p
export VLLM_DISABLED_KERNELS=AllSparkLinearKernel           # INT8 projections through Marlin W8A16
# 0018 (optional, +2.9 %): the standalone 6-stage MoE Marlin build from kernels/marlin/build.sh
# export VLLM_MARLIN_DEV_SO=/path/to/_marlin_dev.so MARLIN_DEV_STAGES=6

# --- variable / rolling speculative acceptance (serving/VARIABLE_ACCEPTANCE.md) ---
export VLLM_SPEC_ROLL=1
export VLLM_SPEC_ROLL_SOLO=2                                # 0026: rolling rule only while <= 2 requests run
export VLLM_SPEC_ROLL_STEP_MS=24.9,27.4,29.9,32.5,35.8,38.0,41.0,42.9   # MEASURE THIS ON YOUR HARDWARE

# --- pipeline parallelism ---
export VLLM_PP_LAYER_PARTITION=13,12,12,8                   # layers per rank; the last rank also hosts the drafter
export VLLM_PP_MAX_DECODE_REQS_PER_BATCH=2                  # decode requests per micro-batch (about concurrency / PP depth)
export VLLM_GLM5N_SIDECAR_BLOCK_SIZE=256 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 7 drafts per step; the per-batch-size schedule lists k=1..6 on unreachable batch sizes (9..14) so a FULL CUDA graph is
# captured for every draft length the rolling rule can choose (batch sizes 1-8 keep k=7).
SPEC='{"method":"dflash","model":"'"$DRAFTER"'","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,8,7],[9,9,1],[10,10,2],[11,11,3],[12,12,4],[13,13,5],[14,14,6]]}'

vllm serve "$MODEL" --served-model-name GLM-5.3-Flash --port "$PORT" \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 \
  --max-model-len 131072 --gpu-memory-utilization 0.92 --no-enable-prefix-caching \
  --max-num-seqs 8 --max-num-batched-tokens 1536 \
  --speculative-config "$SPEC" \
  --cudagraph-capture-sizes 1 2 3 4 5 6 7 8 12 16 24 32 48 64 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45

# Options measured after the published results:
#  - image input: --limit-mm-per-prompt '{"image":4,"video":0}' with VLLM_AOT_MM_KEY=1 and VLLM_GLM5_MM_BUDGET=1
#    (patches 0035/0036); the KV pool shrinks by about 12 %.
#  - --long-prefill-token-threshold=1024: a short request arriving during a long prefill waits ~1.8 s instead of ~6.7 s.
#  - --max-logprobs 5: only needed for the perplexity gate (benchmark/ppl_eval.py).
# Production sampling is the model's generation_config default (temperature 1.0 / top-p 0.95); the benchmarks send no
# sampling parameters so they inherit it.
