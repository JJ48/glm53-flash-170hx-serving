# Variable (rolling) speculative acceptance

Speculative decoding drafts `k` tokens per step and verifies them; the drafter here can propose up
to 7. Verifying more tokens costs more step time but only pays off if those positions are actually
accepted — and acceptance is highly content-dependent (structured output accepts ~7, prose ~2). A
fixed `k` is therefore wrong for most requests. This tooling picks `k` **per request, adaptively**,
from measured acceptance and the measured step-time curve.

Code: [`kernels/spec_roll.py`](../kernels/spec_roll.py) (the estimator),
[`kernels/spec_roll_solo_patch.py`](../kernels/spec_roll_solo_patch.py) (the concurrency guard) and
[`kernels/apply_spec_k.py`](../kernels/apply_spec_k.py) with `kernels/files/` (patch 0003: the per-request draft length
and the scheduler/runner hooks the rule plugs into). All are Apache-2.0 modifications over vLLM, applied in order by
`kernels/apply_stack.sh` (see [`PATCH_STACK.md`](PATCH_STACK.md)).

## How it decides `k`

Per request it keeps a short window of `(accepted, drafted)` pairs and estimates the acceptance
probability of each draft position `j`:

```
P_j = (# steps with accepted >= j) / (# steps with drafted >= j)
```

Acceptance is prefix-based, so `P_j` is the probability that verifying position `j` pays. It then
scores every candidate `k` against the **measured step-time table** `T(n)` (time for `n` verified
tokens) and picks the `k` that maximizes expected accepted-tokens per unit time:

```
k* = argmax_k  (1 + sum_{j<=k} P_j) / T(1 + k)
```

Scoring all `k` (rather than walking `k` up/down one notch per window) is what lets it jump the
box's step-time "cost cliff" — e.g. on this hardware 3→4 tokens adds +2.4 ms, 4→5 adds +5.8 ms,
but 5→8 is only ~1.1 ms/token, so `k` should never sit at 5. Each request starts at the ceiling so
every position is observed immediately and `k` lands where it belongs after a few steps; a periodic
probe re-checks `k+1` so a request whose acceptance improves can climb again.

## Concurrency guard (important)

The rolling rule is a **single-stream / low-concurrency win** and a loss at high concurrency. When a
pipeline micro-batch holds two requests with different `k`, the batch is no longer uniform and the
step falls off the full-CUDA-graph path onto the piecewise path (~7 ms of kernel launches per rank
per step). Measured on this box at concurrency 8: rolling-on 158–164 tok/s vs static-k=7 217 tok/s.

`spec_roll_solo_patch.py` gates the rule by the number of running requests
(`VLLM_SPEC_ROLL_SOLO=N`): apply the per-request `k` only while at most `N` requests run; above that,
every request drafts to the ceiling and the batch stays uniform. The estimator keeps observing
acceptance throughout, so a request that becomes solo again resumes from an up-to-date estimate.
Single-stream behavior is unchanged.

## Configuration

Enable with `VLLM_SPEC_ROLL=1`. All knobs are environment variables:

| env var | default | meaning |
|---|---|---|
| `VLLM_SPEC_ROLL` | `0` | `1` enables the rolling estimator |
| `VLLM_SPEC_ROLL_SOLO` | `0` | apply the rule only while ≤ N requests run (`0` = always; `4` = while each micro-batch holds one request) |
| `VLLM_SPEC_ROLL_STEP_MS` | *(built-in)* | comma-separated measured step time (ms) for 1..K verified tokens — **measure this on your hardware** |
| `VLLM_SPEC_ROLL_WINDOW` | `16` | steps of history kept per request |
| `VLLM_SPEC_ROLL_MIN_OBS` | `4` | observations of a position before it is judged |
| `VLLM_SPEC_ROLL_TAU_MIN` | `0.05` | floor on the break-even probability |
| `VLLM_SPEC_ROLL_PROBE` | `6` | every PROBE-th step verifies `k+1` (0 = never) |
| `VLLM_SPEC_ROLL_JUMP_N` | `3` | consecutive fully-accepted steps that jump back to the ceiling |
| `VLLM_SPEC_ROLL_MIN` | `2` | never draft fewer than this (k=1 measured slower than k=2 here) |
| `VLLM_SPEC_ROLL_START` | *(ceiling)* | initial `k`; empty = start at the ceiling so every position is observed |

### Measuring `STEP_MS` for your box

`STEP_MS` is the single most hardware-specific input: the median decode step time for each verified
token count `1..K`, in milliseconds. Measure it with the drafter attached by forcing a fixed `k` and
timing the decode step (vLLM's stage timer, or wall/step). Example values measured on 4× CMP 170HX
(PP4), for k=1..8:

```
VLLM_SPEC_ROLL_STEP_MS=24.9,27.4,29.9,32.5,35.8,38.0,41.0,42.9
```

Use your own numbers — the cliff location is what makes the estimator pick good `k`, and it moves
with GPU, topology, and model.
