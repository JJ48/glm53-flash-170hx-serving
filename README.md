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
note below — you supply/train your own), the cards unlocked with cmpunlocker (see above), and a
vLLM build with the patches in [`kernels/`](kernels/).

```bash
# 1. Get the quantized model — download the published checkpoint from Hugging Face:
huggingface-cli download JJ48-24/GLM-5.3-Flash-AWQ-W4A16-aggr-w8-int4g32-mixA --local-dir ./glm5.3-flash-w8w4
#    (or reproduce it from the base model with the recipe: see serving/QUANT_RECIPE.md)

# 2. Launch the server (adjust paths, drafter, GPU topology to your box)
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
- The variable/rolling speculative-acceptance tooling and vLLM patches (`kernels/`, `serving/VARIABLE_ACCEPTANCE.md`)
- Serving configuration (`serving/`)

**Model weights:** the quantized GLM-5.3-Flash checkpoint is **published on Hugging Face** —
`https://huggingface.co/JJ48-24/GLM-5.3-Flash-AWQ-W4A16-aggr-w8-int4g32-mixA` — download it directly. The quantization recipe is also included here (`quant/`,
`serving/QUANT_RECIPE.md`) for transparency and to reproduce it from the base model. The base model
is under its own license.

**Not included** (bring your own):
- **Drafter weights.** The DFlash2 drafter used here is licensed **CC BY-NC-ND** for internal use only.
  Its **No-Derivatives** term and internal-only grant mean neither the weights nor any quantized
  derivative can be redistributed. The repo documents exactly how a 7-token DFlash2 drafter is wired in
  and configured, so you can reproduce with a compatible drafter you supply or train — we just can't
  ship ours.

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
- **DFlash2 speculative drafter** — used under an internal CC BY-NC-ND license; not redistributed here
  (see [`NOTICE.md`](NOTICE.md)).

## License

Code and docs in this repo are Apache-2.0 (see [`LICENSE`](LICENSE)), except where a file's header
states otherwise. Attributions and third-party terms are in [`NOTICE.md`](NOTICE.md).
