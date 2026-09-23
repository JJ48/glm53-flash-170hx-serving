# GLM-5.3-Flash on 4× CMP 170HX — optimized serving

Serving **GLM-5.3-Flash** (321B-A18B MoE) efficiently on **4× NVIDIA CMP 170HX** — cheap mining
GPUs with no display output (64 GB HBM2e, PCIe Gen2 ×4) — via pipeline parallelism, weight-only
quantization, and speculative decoding with a **variable (rolling) acceptance** rule.

This repo publishes the serving stack and the optimizations that make it fast: the quantization
recipe, the variable-acceptance speculative-decoding tooling, the serving configuration, and the
measurement harness — plus the [benchmark results](#results-at-a-glance) that show what they
achieve. It does **not** ship model or drafter weights — see
[What is / isn't included](#what-is--isnt-included).

## Results at a glance

Current serving stack, production sampling (temperature 1.0 / top-p 0.95), `max_batched_tokens` 1536.

### Single-stream throughput & acceptance, by workload

| workload  | tok/s | accept_len |
|-----------|------:|-----------:|
| prose     |    73 |       1.91 |
| reasoning |    99 |       2.91 |
| code      |   127 |       4.18 |
| math      |   127 |       4.51 |
| json      |   150 |       4.95 |
| count     |   203 |       7.51 |
| repeat    |   170 |       7.31 |
| **overall** | **120** | **3.96** |

`accept_len` = mean accepted tokens per decode step (1.0 = no speculation benefit; the 7-token
drafter can reach 8.0). Acceptance is strongly content-dependent: open-ended prose is hard to
predict (~1.9), highly structured output is easy (~7.5).

### Throughput vs concurrency

| concurrency | per-request tok/s | system total tok/s | mean latency |
|------------:|------------------:|-------------------:|-------------:|
|           1 |               121 |                107 |        3.7 s |
|           2 |                99 |                170 |        4.7 s |
|           4 |                61 |                199 |        7.9 s |
|           8 |                41 |                245 |       11.8 s |
|          16 |                20 |                248 |       22.0 s |

- **per-request tok/s** = one user's decode rate (unweighted mean over requests).
- **system total tok/s** = total tokens ÷ wall (token-weighted; the box's aggregate output).

At C=1 the two differ only by weighting (the token-heavy prose/reasoning requests are also the
slowest, so they pull the token-weighted total down). As concurrency rises, each request slows
(121 → 20 tok/s) while **system total saturates at ~245 tok/s by 8 concurrent streams** — beyond
that you buy queue depth, not throughput. Acceptance holds ~3.1–3.4 under load; zero failed requests.

### Prefill / time-to-first-token (median of 3)

| prompt tokens |  TTFT   | prefill tok/s |
|--------------:|--------:|--------------:|
|         2,048 | 1.49 s  |         1,378 |
|         4,096 | 2.06 s  |         1,990 |
|         8,192 | 3.22 s  |         2,546 |
|        16,384 | 5.58 s  |         2,937 |
|        32,768 | 10.29 s |         3,186 |

Burst (8 × 23.5k-token prompts submitted at once): 59.3 s wall, TTFT 8 / 34 / 59 s (min/median/max).

Raw numbers are in [`results/`](results/).

## The stack

| | |
|---|---|
| **Hardware** | 4× NVIDIA CMP 170HX, 64 GB HBM2e each, PCIe Gen2 ×4 interconnect |
| **Server** | vLLM, pipeline-parallel **PP4** across all 4 GPUs (layer split 13/12/12/8), no tensor parallelism |
| **Model** | GLM-5.3-Flash — 321B total / 18B active MoE, 45 layers (34 KDA linear-attention + 11 full MLA) |
| **Weight quant** | compressed-tensors, weight-only: attention **INT8** per-channel · MoE experts **INT4 g32** · dense **INT4 g128** · lm_head INT4 g32 — see [`serving/QUANT_RECIPE.md`](serving/QUANT_RECIPE.md) |
| **Speculative decode** | DFlash2 drafter, **7 draft tokens/step**, with a **variable (rolling) acceptance** rule keyed to a per-step time budget — see [`serving/VARIABLE_ACCEPTANCE.md`](serving/VARIABLE_ACCEPTANCE.md) |
| **Prefill** | chunked, `max_num_batched_tokens` 1536 · KV cache pool ~692k tokens |
| **Sampling** | production: temperature 1.0 / top-p 0.95 |
| **Hardware tuning** | cards unlocked (cmpunlocker `--p2p`), per-card HBM overclock, **175 W/card** power cap — see [`serving/HARDWARE_TUNING.md`](serving/HARDWARE_TUNING.md) |
| **Patch stack** | 23 vLLM patches over the club-170hx image, applied in production's order; all but the two loader fixes are inert until their flag is set — see [`serving/PATCH_STACK.md`](serving/PATCH_STACK.md) |

## Hardware prerequisites

The CMP 170HX ships with SM compute, PCIe, GPU-to-GPU P2P, and HBM geometry clamped in firmware.
These results assume the cards are unlocked with **cmpunlocker**
(<https://github.com/asm64-hooligan/cmpunlocker>, GPLv2), which restores full compute, PCIe Gen2,
and **GPU-to-GPU P2P** (`--p2p`). The big win is restoring full SM compute; `--p2p` is used by the
pipeline hops but was worth only a small gain here (~2 %), enabled because it costs nothing. Per-card
HBM clock tuning uses a `--mclk-percard` extension contributed to cmpunlocker — until it merges upstream
it's on the fork at <https://github.com/JJ48/cmpunlocker> (branch `percard-hbm`). Neither the unlock
tool nor the fork is bundled here (GPLv2; this repo is Apache-2.0). See
[`serving/HARDWARE_TUNING.md`](serving/HARDWARE_TUNING.md) for the full hardware setup and the power
cap. cmpunlocker is third-party firmware-level tooling under its own license and NVIDIA's terms;
obtain and use it at your own discretion.

## Reproducing

You need: GLM-5.3-Flash (from its own source), a compatible 7-token speculative drafter (see the licensing
note below — you supply/train your own), the cards unlocked with cmpunlocker (see above), and the
club-170hx vLLM image (`ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905`) with the full patch stack
from [`kernels/`](kernels/) applied — see [`serving/PATCH_STACK.md`](serving/PATCH_STACK.md).

```bash
# 0. Inside the club-170hx container, apply the patch stack (in order; every patch is inert until its flag is set)
bash kernels/apply_stack.sh
#    optional (+2.9 %): build the standalone 6-stage MoE Marlin kernel for patch 0018
bash kernels/marlin/build.sh

# 1. Get the quantized model — download the published checkpoint from Hugging Face:
huggingface-cli download JJ48-24/GLM-5.3-Flash-AWQ-W4A16-aggr-w8-int4g32-mixA --local-dir ./glm5.3-flash-w8w4
#    (or reproduce it from the base model with the recipe: see serving/QUANT_RECIPE.md)

# 2. Launch the server with the production flags (adjust paths, drafter, GPU topology to your box)
bash serving/launch.example.sh

# 3. Run the benchmarks against the OpenAI-compatible endpoint
python benchmark/en_conc.py        --base-url http://127.0.0.1:8000 --levels 1,2,4,8,16
python benchmark/prefill_ladder.py --base-url http://127.0.0.1:8000 --lengths 2048 4096 8192 16384 32768 --burst 8:23500
python benchmark/ppl_eval.py       --base-url http://127.0.0.1:8000 --tag mine
```

All three benchmarks talk to a standard OpenAI-compatible `/v1` endpoint and read acceptance from
the server's `/metrics` (vLLM spec-decode counters). See [`METHODOLOGY.md`](METHODOLOGY.md) for the
measurement rules (production sampling, prompt-average vs token-weighted, balanced prompt sets,
steady-state windows).

## What is / isn't included

**Included** (Apache-2.0 unless noted — see [`NOTICE.md`](NOTICE.md)):
- The benchmark harness (`benchmark/`)
- Full methodology (`METHODOLOGY.md`) and raw results (`results/`)
- The weight-quantization recipe and tooling (`quant/`, `serving/QUANT_RECIPE.md`)
- The complete vLLM patch stack the results were measured with (`kernels/`, `serving/PATCH_STACK.md`), including the
  variable/rolling speculative-acceptance tooling (`serving/VARIABLE_ACCEPTANCE.md`) and the standalone Marlin kernel
  build (`kernels/marlin/`)
- Serving configuration (`serving/`)

**Model weights:** the quantized GLM-5.3-Flash checkpoint is **published on Hugging Face** —
`https://huggingface.co/JJ48-24/GLM-5.3-Flash-AWQ-W4A16-aggr-w8-int4g32-mixA` — download it directly. The quantization recipe is also included here (`quant/`,
`serving/QUANT_RECIPE.md`) for transparency and to reproduce it from the base model. The base model
is under its own license.

**Not included** (bring your own):
- **Drafter weights.** The drafter is **GLM-5.3-DFlash2** by incoai
  (<https://huggingface.co/incoai/GLM-5.3-DFlash2>), **CC BY-NC-ND**. We ship none — our served copy is
  a quantized derivative, which the **No-Derivatives** term doesn't let us redistribute. Get the
  original from incoai (under its license), or reproduce with a compatible drafter you supply; the repo
  documents exactly how a 7-token drafter is wired in.

## Changelog

**2026-09-23**
- **Full patch stack published** ([`serving/PATCH_STACK.md`](serving/PATCH_STACK.md), `kernels/apply_stack.sh`). The
  first release shipped only the rolling-acceptance files: without patches 0001/0002 the published INT8-attention
  checkpoint does not load in the club image, without 0003 the rolling rule is not wired in, and the speed patches
  behind the published numbers were missing.
- **Fixed a long-prompt bug in the compile patch (0014) with patch 0038.** 0014 froze the indexer head-gate of 9 of
  the 11 sparse-attention layers at zero, so for prompts over 2,048 tokens those layers read only the first 2,048
  tokens. Set `VLLM_GLM5_WP_FIX=1` whenever `VLLM_GLM5_COMPILE=1`. Measured on synthetic long documents with planted
  facts: a retrieval-shaped prompt went from 2/12 to 12/12 correct, and decode at 16k-token prompts from 65 to 130 tok/s
  (draft acceptance 1.6 -> 4.0). Time to first token is unchanged (8k 3.14 -> 3.13 s, 16k 5.58 -> 5.53 s). The results
  above were measured with the bug present; they use short prompts, where the model attends to every token, and the
  prefill ladder is unaffected.
- **Launch example corrected** to the measured configuration: every patch flag, the layer split, the per-draft-length
  CUDA-graph schedule, and `VLLM_SPEC_ROLL_SOLO=2` (the example previously said 4).

## Citation

If you use this benchmark, its harness, or its tooling, please cite it (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{jkeys2026glm53flash,
  title        = {{GLM-5.3-Flash on 4× CMP 170HX: Optimized Serving}},
  author       = {{JKeys LLC}},
  year         = {2026},
  howpublished = {\url{https://github.com/JJ48/glm53-flash-170hx-serving}},
  note         = {Apache-2.0}
}
```

## References

This work builds on:

- **vLLM** — the serving engine; the patches in `kernels/` are modifications of it.
  Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*,
  SOSP 2023. <https://github.com/vllm-project/vllm>
- **GLM-5.3-Flash** — the base model benchmarked here (obtain from its official source; the
  quantized checkpoint is at <https://huggingface.co/JJ48-24/GLM-5.3-Flash-AWQ-W4A16-aggr-w8-int4g32-mixA>).
- **compressed-tensors** — the weight-quantization format. <https://github.com/neuralmagic/compressed-tensors>
- **cmpunlocker** — firmware-level unlock for the CMP 170HX (full compute, PCIe Gen2, GPU-to-GPU P2P,
  HBM geometry); required to reach these numbers. GPLv2, by asm64-hooligan.
  <https://github.com/asm64-hooligan/cmpunlocker>. Our per-card HBM clock autotune (`--mclk-percard`
  + `tools/hbmtune`) is contributed upstream; until merged it's on the fork at
  <https://github.com/JJ48/cmpunlocker> (branch `percard-hbm`). See [`serving/HARDWARE_TUNING.md`](serving/HARDWARE_TUNING.md).
- **GLM-5.3-DFlash2** — the speculative drafter, by incoai. CC BY-NC-ND.
  <https://huggingface.co/incoai/GLM-5.3-DFlash2>. Weights not redistributed here (see [`NOTICE.md`](NOTICE.md)).

## License

Code and docs in this repo are Apache-2.0 (see [`LICENSE`](LICENSE)), except where a file's header
states otherwise. Attributions and third-party terms are in [`NOTICE.md`](NOTICE.md).
