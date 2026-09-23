# Notices and third-party terms

This repository contains original work (Apache-2.0, see `LICENSE`) plus modifications to third-party
software and references to third-party models. Terms below.

## vLLM (Apache-2.0)

The patches in `kernels/` are modifications to [vLLM](https://github.com/vllm-project/vllm), which is
licensed under Apache-2.0. They are redistributed here under Apache-2.0, with changes noted in each
patch file's header. `kernels/files/` contains vLLM source files (the `pristine/` copies patch 0003 uses as its
baseline, and the modules it adds), and `kernels/marlin/` contains a diff against vLLM commit 8e92248f79 plus build
glue for its Marlin MoE kernel; both are Apache-2.0. The patches target the club-170hx community image
(`ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905`), which is not redistributed here. This project is not
affiliated with or endorsed by the vLLM project.

## GLM-5.3-Flash model

GLM-5.3-Flash is a third-party model under its own license. **No model weights are distributed in
this repository.** The quantized checkpoint is published separately on Hugging Face
(`https://huggingface.co/JJ48-24/GLM-5.3-Flash-AWQ-W4A16-aggr-w8-int4g32-mixA`); `quant/` and `serving/QUANT_RECIPE.md` publish the recipe used to produce it.
Obtain the base model from its official source and comply with its license.

## Speculative-decoding drafter (weights NOT included)

The drafter used in these benchmarks is **GLM-5.3-DFlash2** by incoai
(<https://huggingface.co/incoai/GLM-5.3-DFlash2>), licensed **CC BY-NC-ND** (Creative Commons
Attribution-NonCommercial-NoDerivatives). This repository ships no drafter weights: our served copy
is a quantized derivative, which the **No-Derivatives** term does not permit us to redistribute.
Obtain the original from incoai under its license, or supply a compatible drafter you have the right
to use.

What *is* published is our own integration and scheduling tooling — how a 7-token DFlash2 drafter is
wired into the server and the **variable/rolling acceptance** rule (`kernels/`,
`serving/VARIABLE_ACCEPTANCE.md`) — which is original work over Apache-2.0 vLLM and contains no
drafter weights. To reproduce the speculative-decoding results, supply a compatible speculative drafter you
have the right to use (your own, or one you train).

## Benchmark corpora

The perplexity gate uses public-domain texts (Project Gutenberg). The English decode suite is a small
set of synthetic prompts authored for this project and included under Apache-2.0.
