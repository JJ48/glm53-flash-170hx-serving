# Methodology

How every number in this repo was measured. The rules matter — several are easy to get wrong in a
way that produces plausible-but-misleading figures.

## Sampling

All throughput/latency/acceptance numbers use **production sampling: temperature 1.0, top-p 0.95**
— the model's `generation_config` defaults. A request that sets no sampling parameters inherits
these, so "no params" and "1.0 / 0.95" are the same thing on this server. Perplexity is
sampling-free (greedy log-likelihood) and independent of this.

Speculative acceptance depends on sampling temperature, so quote the temperature with any
acceptance figure. Do not benchmark acceptance at temperature 0 and present it as production.

## Two throughput numbers, and why they differ

For any concurrency level we report both:

- **per-request tok/s** — the *unweighted* mean of each request's `completion_tokens / wall`. This
  is "what one user experiences," and it degrades as concurrency rises.
- **system total tok/s** — `sum(completion_tokens) / total_wall`, i.e. *token-weighted*. This is the
  box's aggregate output, and it rises with concurrency until it saturates.

They are **not** equal even at concurrency 1, because the workload mix has very different
per-request rates (structured output is fast, prose/reasoning is slow) and token counts. The
unweighted mean lets a fast, short request count as much as a slow, long one; the token-weighted
total is dominated by the long slow ones. Report both, label which is which, and never silently mix
them across a table.

## Prompt set and balancing

Decode/acceptance runs use a fixed 20-prompt English suite spanning seven categories (prose, code,
math, json, reasoning, count, repeat). For a concurrency sweep, each level runs the **same balanced
set** (every prompt an equal number of times, e.g. N=40 = each prompt twice) so the per-request
average is a true prompt average and levels are comparable. An unbalanced set (e.g. cycling a
partial extra pass) skews the C=1 anchor — that's a real trap.

## Concurrency sweep

A closed-loop pool of C worker threads drains N requests against the OpenAI-compatible endpoint.
Aggregate output is read from the server's `vllm:generation_tokens_total` counter over the run;
acceptance from the `vllm:spec_decode_*` counters (accept_len = 1 + accepted/drafts). Per-request
rates come from each request's own timing. Report zero-failure as a pass criterion.

Synthetic/random prompts are **not** representative for a speculative-decoding stack: the drafter
can't predict unnatural token sequences, so acceptance collapses to ~1.3 and throughput reads as a
no-speculation worst case. Use real (natural-language) prompts.

## Prefill / TTFT

Fixed synthetic prompts of exact token lengths (2k/4k/8k/16k/32k), a fresh nonce prefix per request
so nothing is reused, prefix caching **off**. One warm-up request per length, then `--reps` counted
requests; report the **median**. The first request after a cold boot is slow — compare like
positions and never compare a long-running server against a fresh boot. Burst = N long prompts
submitted simultaneously; report wall time and the TTFT min/median/max.

## Perplexity (quality gate)

Sampling-free: mean negative log-likelihood of fixed public-domain text via `/v1/completions` with
`prompt_logprobs` (`max_tokens=1`, `echo=false`). The same text chunks are used for every checkpoint
so results are directly comparable; `ppl_eval.py compare a.json b.json` prints the per-text and mean
delta. This is how quantization variants are gated (a variant must stay within a set PPL delta of the
baseline and pass an exact-answer canary).

## General

- Never quote a single run for a small effect. Compare like positions; the first boot is slow.
- Stage/kernel launch-time counters do not predict end-to-end effect on this box; measure the
  end-to-end number.
- Reproducibility of temperature-0 text is not guaranteed run to run — use exact-answer canaries
  (tasks with one correct answer), not text equality.
