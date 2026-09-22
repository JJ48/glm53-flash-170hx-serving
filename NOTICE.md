# Notices and third-party terms

This repository contains original work (Apache-2.0, see `LICENSE`) plus modifications to third-party
software and references to third-party models. Terms below.

## vLLM (Apache-2.0)

The patches in `kernels/` are modifications to [vLLM](https://github.com/vllm-project/vllm), which is
licensed under Apache-2.0. They are redistributed here under Apache-2.0, with changes noted in each
patch file's header. This project is not affiliated with or endorsed by the vLLM project.

## GLM-5.3-Flash model

GLM-5.3-Flash is a third-party model under its own license. **No model weights are distributed in
this repository.** The quantized checkpoint is published separately on Hugging Face
(`https://huggingface.co/JJ48-24/GLM-5.3-Flash-AWQ-W4A16-aggr-w8-int4g32-mixA`); `quant/` and `serving/QUANT_RECIPE.md` publish the recipe used to produce it.
Obtain the base model from its official source and comply with its license.

## Speculative-decoding drafter (weights NOT included)

The DFlash2 drafter used in these benchmarks is licensed **CC BY-NC-ND** (Creative Commons
Attribution-NonCommercial-NoDerivatives), granted for internal use only. Under its **No-Derivatives**
term and internal-only grant, neither the drafter weights nor any quantized derivative of them may be
redistributed, and this repository does not contain them.

What *is* published is our own integration and scheduling tooling — how a 7-token DFlash2 drafter is
wired into the server and the **variable/rolling acceptance** rule (`kernels/`,
`serving/VARIABLE_ACCEPTANCE.md`) — which is original work over Apache-2.0 vLLM and contains no
drafter weights. To reproduce the speculative-decoding results, supply a compatible speculative drafter you
have the right to use (your own, or one you train).

## Benchmark corpora

The perplexity gate uses public-domain texts (Project Gutenberg). The English decode suite is a small
set of synthetic prompts authored for this project and included under Apache-2.0.
