#!/usr/bin/env bash
# Example launch for GLM-5.3-Flash on 4 GPUs with pipeline parallelism + speculative decoding +
# the variable/rolling acceptance rule. Adjust paths, GPU count, and the step-time table to your box.
# This is illustrative — it is NOT a turnkey production launcher. It assumes a vLLM build with the
# patches from ../kernels/ applied (spec_roll.py on the import path, spec_roll_solo_patch.py applied).
set -euo pipefail

MODEL=${MODEL:-/path/to/glm5.3-flash-w8w4}          # the quantized model (see serving/QUANT_RECIPE.md)
DRAFTER=${DRAFTER:-/path/to/your-mtp-drafter}       # you supply this — the benchmark drafter is not redistributable
PORT=${PORT:-8000}

# --- variable / rolling speculative acceptance (see serving/VARIABLE_ACCEPTANCE.md) ---
export VLLM_SPEC_ROLL=1
export VLLM_SPEC_ROLL_SOLO=4                          # rolling rule only while each micro-batch holds one request
export VLLM_SPEC_ROLL_STEP_MS=24.9,27.4,29.9,32.5,35.8,38.0,41.0,42.9   # MEASURE THIS ON YOUR HARDWARE

# Speculative decoding config: an MTP drafter proposing up to 7 tokens/step.
SPEC='{"method":"dflash","model":"'"$DRAFTER"'","num_speculative_tokens":7}'

vllm serve "$MODEL" \
  --port "$PORT" \
  --pipeline-parallel-size 4 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 1536 \
  --speculative-config "$SPEC" \
  --max-logprobs 5                # needed only if you run the perplexity gate (benchmark/ppl_eval.py)

# Notes:
#  - Pipeline layer split and KV pool size depend on your GPUs; tune --pipeline-parallel-size and
#    the vLLM memory-utilization flag to your hardware.
#  - Production sampling here is the model's generation_config default (temperature 1.0 / top-p 0.95);
#    the benchmarks send no sampling params so they inherit it.
