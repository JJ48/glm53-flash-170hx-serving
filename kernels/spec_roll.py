# SPDX-License-Identifier: Apache-2.0
"""Rolling speculative k, iteration 4.7: per-position acceptance scored over the measured step-time table (fork patch 0003).

Iteration 3 judged one number per request ("fraction of recent steps that accepted every draft") and walked k up or down one
notch per window, so a request that should sit at k=2 spent its first ~40 steps descending 7 -> 4 -> 3 -> 2 (8 % of a 300-token
code answer at the 8-token step cost), and a request that should sit at k=7 climbed 4 -> 5 -> 6 -> 7 (6 % lost on JSON).

Iteration 4 keeps, per request, a window of (accepted, drafted) pairs and estimates the acceptance probability of every draft
position j: P_j = #steps with accepted >= j / #steps with drafted >= j (acceptance is prefix-based, so this is the probability
that position j pays). Position j is worth verifying when P_j exceeds its break-even, which is the extra step cost of one more
verified token as a fraction of the tokens the step already yields:

    k* = argmax_k  (1 + sum_{j<=k} P_j) / T(1 + k)

where T(n) is the measured step time for n verified tokens (VLLM_SPEC_ROLL_STEP_MS). Scoring all k rather than walking positions is
what lets it jump the box's 5-token cost cliff (3->4 +2.4 ms, 4->5 +5.8 ms, 5->8 ~1.1 ms/token). The request's k is the deepest j (<= ceiling) with P_j >= tau_j, never below KMIN. Because the request starts at the
ceiling, every position is observed in the first window, so k lands where it belongs after MIN_OBS steps instead of walking;
afterwards, every PROBE-th step verifies one extra position so a request whose acceptance improves can climb again
(WINDOW/PROBE must leave >= MIN_OBS probe observations per window: 32/6 gives 5; 24/8 gave 3 and math got stuck at k=3).

Per-server opt-in: VLLM_SPEC_ROLL=1; per-request override vllm_xargs["spec_roll"] = 0/1; vllm_xargs["spec_tokens"] stays a ceiling.
Env knobs: VLLM_SPEC_ROLL_WINDOW (16), VLLM_SPEC_ROLL_MIN_OBS (4), VLLM_SPEC_ROLL_JUMP_N (3), VLLM_SPEC_ROLL_STEP_MS (table), VLLM_SPEC_ROLL_TAU_MIN (0.05),
VLLM_SPEC_ROLL_PROBE (6), VLLM_SPEC_ROLL_MIN (2), VLLM_SPEC_ROLL_START (empty = ceiling).
"""
from __future__ import annotations

import os

ENABLED = os.environ.get("VLLM_SPEC_ROLL", "0") == "1"
WINDOW = max(6, int(os.environ.get("VLLM_SPEC_ROLL_WINDOW", "16")))        # steps kept per request (short = tracks phases)
MIN_OBS = max(2, int(os.environ.get("VLLM_SPEC_ROLL_MIN_OBS", "4")))       # observations of a position before it is judged
# Step time (ms) by number of verified tokens 1..8 (1 + drafts). MEASURE THIS PER DRAFTER FAMILY: a block drafter (DFlash2) runs one
# draft pass whatever k is, so its curve is linear, while an autoregressive drafter (MTP) adds a pass per k and has a real cliff.
# Mixing the two (the 2026-09-14 error) imports MTP k=4's +5.8 ms step as a phantom cliff at 5 tokens and makes the rule refuse to
# draft past k=4 (box math sat at k 4.2 while k=7 was worth +21 %). 170HX DFlash2, measured on the k=2 / k=3 / k=7 servers
# (step ms = accept_len / tok_s): 3 tokens 28.7, 4 tokens 31.1, 8 tokens 40.6 -> linear, 2.38 ms per verified token; the fit
# reproduces the measured 4-token point exactly. A100 is also near-linear (~19.5 + 1.56/token): "21.1,22.6,24.2,25.8,27.3,28.9,30.4,32.0".
# 170HX MTP (INT8 head), measured 2026-09-15 on the k=2/3/4/5 servers over the English chat suite: 3 tokens 27.5, 4 tokens 31.5,
# 5 tokens 37.3, 6 tokens 40.7 -> deltas +4.0, +5.9, +3.4. The +5.9 ms step at 5 verified tokens is MTP's REAL cliff (it also
# appeared on the A100) and it is the very number that, misattributed to DFlash2, caused the 2026-09-14 error above. Extrapolate
# the unmeasured ends off the LOCAL slope, never a global fit, or the cliff smears into both tails:
# "19.6,23.5,27.5,31.5,37.3,40.7,44.2,47.6". Consequence: MTP tok/s peaks at k=3 and FALLS at k=4, and the best static k moves
# from 2 (prose, reasoning) to 3 (code, math) to 5 (json, counting, repetition) - a 39 % swing on prose if chosen wrong.
_STEP_DEFAULT = "24.0,26.4,28.7,31.1,33.5,35.9,38.3,40.6"
# `or _STEP_DEFAULT`: an env var that is SET BUT EMPTY (a runner whose table derivation silently produced nothing) otherwise
# reaches float("") and kills the server at import with a ValueError that names neither the variable nor the runner.
STEP_MS = [float(x) for x in (os.environ.get("VLLM_SPEC_ROLL_STEP_MS", "").strip() or _STEP_DEFAULT).split(",")]
TAU_MIN = float(os.environ.get("VLLM_SPEC_ROLL_TAU_MIN", "0.05"))          # floor on the break-even probability
PROBE = max(0, int(os.environ.get("VLLM_SPEC_ROLL_PROBE", "6")))           # every PROBE-th step verifies k+1 (0 = never)
JUMP_N = max(0, int(os.environ.get("VLLM_SPEC_ROLL_JUMP_N", "3")))         # consecutive fully-accepted steps that jump to the ceiling (0 = off)
PROBE_SETTLE = max(0, int(os.environ.get("VLLM_SPEC_ROLL_PROBE_SETTLE", "48")))  # steps of frequent probing before decaying
PROBE_DECAY = max(1, int(os.environ.get("VLLM_SPEC_ROLL_PROBE_DECAY", "4")))   # probe period multiplier after that
KMIN = max(1, int(os.environ.get("VLLM_SPEC_ROLL_MIN", "2")))              # never below (k=1 measured slower than k=2 everywhere)
START = os.environ.get("VLLM_SPEC_ROLL_START", "")                          # initial k; empty = the ceiling (observe every position)


def start_for(cap: int) -> int:
    """Initial rolling k for a ceiling: VLLM_SPEC_ROLL_START if set, else the ceiling itself."""
    if START:
        return min(cap, max(KMIN, int(START)))
    return max(KMIN, cap)


def enabled_for(request) -> bool:
    """Server default (VLLM_SPEC_ROLL) unless the request carries vllm_xargs["spec_roll"]."""
    flag = getattr(request, "spec_roll_on", None)
    return ENABLED if flag is None else bool(flag)


def record_step(request, num_accepted: int, num_draft_tokens: int) -> None:
    """Called by the scheduler when a step's output arrives (accepted drafts out of the scheduled drafts)."""
    if num_draft_tokens <= 0 or not enabled_for(request):
        return
    h = request.spec_roll_hist
    h.append((int(num_accepted), int(num_draft_tokens)))
    if len(h) > WINDOW:
        del h[: len(h) - WINDOW]
    try:
        request.spec_roll_n = getattr(request, "spec_roll_n", 0) + 1
    except AttributeError:
        pass


def _choose(hist, cap: int) -> int | None:
    """Best k by expected throughput over the measured step-time table; None until enough observations.

    Iteration 4.3: iteration 4.2 walked positions and stopped at the first one failing its own break-even, so it could not jump a
    cost cliff -- on the 170HX box the step cost is 3->4 tokens +2.4 ms, 4->5 +5.8 ms (graph cost class changes), then 5->8 only
    ~1.1 ms/token, so a greedy walk halts at k=4 and never finds that k=7 wins outright (measured: rule math 124 vs static-7 150).
    Scoring every candidate k against the table fixes that: expected tokens per step at budget k is 1 + sum_{j<=k} P_j (acceptance
    is prefix-based, so P_j is already the probability that position j is reached and accepted), and the score is that divided by
    the measured step time for 1+k verified tokens. Ties prefer the smaller k (cheaper, and less to discard on a rejection).
    """
    if len(hist) < MIN_OBS:
        return None                      # not enough steps yet: keep the current k (the ceiling at first)
    p_pos: list[float] = []
    for j in range(1, cap + 1):
        obs = sum(1 for a, d in hist if d >= j)
        if obs < MIN_OBS:
            break                        # deeper positions unobserved (only probes reach them): cannot score beyond here
        p_pos.append(sum(1 for a, d in hist if a >= j) / obs)
    if not p_pos:
        return None
    best_k, best_score, cum = KMIN, -1.0, 0.0
    for k in range(1, len(p_pos) + 1):
        cum += p_pos[k - 1]
        score = (1.0 + cum) / _step_ms(1 + k)
        if score > best_score + 1e-12:
            best_score, best_k = score, k
    return max(best_k, KMIN)


def _step_ms(tokens: int) -> float:
    """Step time for a batch-1 step verifying `tokens` tokens (table lookup, linear extrapolation past the table)."""
    if tokens <= len(STEP_MS):
        return STEP_MS[max(tokens, 1) - 1]
    slope = (STEP_MS[-1] - STEP_MS[-2]) if len(STEP_MS) > 1 else 2.0
    return STEP_MS[-1] + slope * (tokens - len(STEP_MS))


def _tau(j: int, cum: float) -> float:
    """Break-even acceptance probability for draft position j given the expected tokens already earned by positions < j.
    Keeping position j moves the step from j to j+1 verified tokens: it pays iff P_j / (T(j+1) - T(j)) > (1 + cum) / T(j)."""
    t_j = _step_ms(j)
    c_j = max(_step_ms(j + 1) - t_j, 0.0)
    return c_j * (1.0 + cum) / t_j


def cap_for(request, k_max: int) -> int:
    """Draft budget for the request's next step: min(client hint, rolling k)."""
    cap = k_max
    hint = getattr(request, "spec_max_tokens", None)
    if hint is not None:
        cap = min(cap, max(0, int(hint)))
    pref = getattr(request, "spec_conf_prefix", None)   # patch 0006: confident prefix of the latest drafts (lagged gate)
    if pref is not None and cap > 0:
        cap = min(cap, max(int(pref), min(KMIN, cap)))
    if cap <= 0 or not enabled_for(request):
        return cap
    rc = request.spec_roll_cap
    if rc is None:
        rc = start_for(cap)
    chosen = _choose(request.spec_roll_hist, cap)
    if chosen is not None:
        rc = chosen
    rc = min(rc, cap)
    # Ride a predictable stretch: JUMP_N consecutive steps in which every scheduled draft was accepted means the tail is paying,
    # so go straight to the ceiling instead of waiting for the window to catch up. This is the mechanism that let the A100 rule
    # beat every fixed k on code (session B: 86.0 vs the best hint 78.6); the smoothed estimator alone converges to a constant k
    # and can only match a fixed k. The window estimate pulls it back down as soon as the stretch ends.
    h = request.spec_roll_hist
    # Measured 2026-09-14: jumping on 2 steps (instead of 3) plus a one-step undo was WORSE on the box (prose 57.1 vs 60.4,
    # code k 3.23 vs 2.83 for no gain) even though a clean alternating-phase simulation preferred it - short predictable runs
    # that do not sustain are not worth riding. Keep the 3-step trigger and let the window pull k back down.
    if JUMP_N and rc < cap and len(h) >= JUMP_N and all(a >= d > 0 for a, d in h[-JUMP_N:]):
        rc = cap
    request.spec_roll_cap = rc
    n = getattr(request, "spec_roll_n", 0)
    # Probe one position deeper so P_{k+1} keeps being observed - often while the estimate is young, rarely once it has settled.
    # A steady-state probe every PROBE steps costs ~1 extra verified token per PROBE steps (~2.4 ms of a ~29 ms step on the box),
    # which is most of the gap between the rule's code result (~69) and a fixed k=2 (72.4 at the same acceptance).
    period = PROBE if n < PROBE_SETTLE else PROBE * PROBE_DECAY
    if PROBE and rc < cap and n > 0 and n % period == 0:
        return min(cap, rc + 1)
    return max(rc, min(KMIN, cap))


if __name__ == "__main__":  # CPU self-test with a synthetic per-position acceptance model
    import random

    class R:
        def __init__(self, hint=None):
            self.spec_roll_hist = []
            self.spec_roll_cap = None
            self.spec_roll_prejump = None
            self.spec_max_tokens = hint
            self.spec_roll_on = True

    def simulate(pos_probs, k_max, steps=300, seed=0, static=False):
        """pos_probs[j-1] = probability position j is accepted given position j-1 was; returns (mean k, tokens/step, ms/step)."""
        rng = random.Random(seed)
        r = R(); ks = []; toks = 0; ms = 0.0
        if static:
            r.spec_roll_on = False       # rule bypassed: cap_for returns the ceiling
        for _ in range(steps):
            k = cap_for(r, k_max); ks.append(k)
            acc = 0
            for j in range(k):
                if rng.random() < pos_probs[j]:
                    acc += 1
                else:
                    break
            record_step(r, acc, k)
            toks += 1 + acc; ms += _step_ms(1 + k)        # box cost model: measured step-time table
        return sum(ks) / len(ks), toks / steps, ms / steps

    def simulate_phased(phases, k_max, steps=400, seed=0, static=False, jump=True):
        """phases = [(pos_probs, n_steps), ...] repeated; models accuracy varying *within* one answer."""
        import os as _os
        global JUMP_N
        keep = JUMP_N
        JUMP_N = keep if jump else 0
        rng = random.Random(seed); r = R(); ks = []; toks = 0.0; ms = 0.0
        if static: r.spec_roll_on = False
        i = 0
        while i < steps:
            for probs, n in phases:
                for _ in range(n):
                    if i >= steps: break
                    k = cap_for(r, k_max); ks.append(k)
                    acc = 0
                    for j in range(k):
                        if rng.random() < probs[j]: acc += 1
                        else: break
                    record_step(r, acc, k)
                    toks += 1 + acc; ms += _step_ms(1 + k); i += 1
        JUMP_N = keep
        return sum(ks)/len(ks), toks/ms*1000

    HI = [0.97]*7                     # predictable stretch (boilerplate, repeated identifiers)
    LO = [0.45, 0.25, 0.12, 0.08, 0.06, 0.05, 0.05]   # novel logic
    MID = [0.7, 0.55, 0.4, 0.3, 0.25, 0.2, 0.2]
    print("\nphased profiles (accuracy varies *within* the answer) - the case a fixed k cannot win:")
    for name, phases in [("code-like  (20 hi / 20 lo)", [(HI,20),(LO,20)]),
                         ("code-like  (10 hi / 30 lo)", [(HI,10),(LO,30)]),
                         ("reasoning  (30 mid / 10 hi)", [(MID,30),(HI,10)])]:
        kj, tj = simulate_phased(phases, 7, jump=True)
        kn, tn = simulate_phased(phases, 7, jump=False)
        bests = max((simulate_phased(phases, k, static=True)[1], k) for k in (2,3,4,7))
        print(f"  {name:28s} jump {tj:6.1f} (k {kj:.2f}) | no-jump {tn:6.1f} (k {kn:.2f}) | best fixed k={bests[1]} {bests[0]:6.1f}"
              f" | jump vs best fixed {tj/bests[0]:.3f}x")
