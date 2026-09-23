# SPDX-License-Identifier: Apache-2.0
"""Rolling-acceptance speculative k (fork patch 0003, iteration 3).

The scheduler sees each request's accepted draft count every step, so it can set the request's draft budget for the next
step from a short window of recent steps — no client labels, no per-token confidence, nothing crossing pipeline ranks.
The signal is the fraction of recent steps in which *every* scheduled draft was accepted: when the last draft position is
accepted that often it is paying for itself (raise k), when it rarely is, the tail is wasted verify work (lower k).
Thresholds are separated for hysteresis, and the window is cleared after a change so the next decision is judged at the
new k. An explicit client hint (vllm_xargs["spec_tokens"]) stays a ceiling. Opt-in per server: VLLM_SPEC_ROLL=1; a request can
override the server setting with vllm_xargs["spec_roll"] = 0 / 1 (so one server can serve rule-on and static arms in a comparison).
"""
from __future__ import annotations

import os

ENABLED = os.environ.get("VLLM_SPEC_ROLL", "0") == "1"
WINDOW = max(4, int(os.environ.get("VLLM_SPEC_ROLL_WINDOW", "24")))       # steps kept per request
MIN_SAMPLES = max(2, int(os.environ.get("VLLM_SPEC_ROLL_MIN_SAMPLES", "10")))  # steps at the current k before judging it
UP = float(os.environ.get("VLLM_SPEC_ROLL_UP", "0.6"))        # fraction of fully-accepted steps at/above which k rises
DOWN = float(os.environ.get("VLLM_SPEC_ROLL_DOWN", "0.35"))   # ... at/below which k falls (break-even is ~0.26-0.3; prefer the cheaper k there)
JUMP_UP = float(os.environ.get("VLLM_SPEC_ROLL_JUMP_UP", "0.9"))    # nearly every step fully accepted -> straight to the ceiling
JUMP_DOWN = float(os.environ.get("VLLM_SPEC_ROLL_JUMP_DOWN", "0.1"))  # almost never -> straight to the floor
KMIN = max(1, int(os.environ.get("VLLM_SPEC_ROLL_MIN", "2")))  # never below (k=1 measured slower than k=2 on every workload)
START = os.environ.get("VLLM_SPEC_ROLL_START", "")            # initial k; empty = derived from the ceiling (3 for k<=4, 4 for k=7)


def start_for(cap: int) -> int:
    """Initial rolling k for a ceiling: the explicit VLLM_SPEC_ROLL_START, else ~0.6 x ceiling (3 for 4, 4 for 7), never below 3."""
    if START:
        return min(cap, max(KMIN, int(START)))
    return min(cap, max(KMIN, 3, (cap * 3 + 2) // 5))


def enabled_for(request) -> bool:
    """Server default (VLLM_SPEC_ROLL) unless the request carries vllm_xargs["spec_roll"]."""
    flag = getattr(request, "spec_roll_on", None)
    return ENABLED if flag is None else bool(flag)


def record_step(request, num_accepted: int, num_draft_tokens: int) -> None:
    """Called by the scheduler when a step's output arrives (accepted drafts out of the scheduled drafts)."""
    if num_draft_tokens <= 0 or not enabled_for(request):
        return
    h = request.spec_roll_hist
    h.append(1 if num_accepted >= num_draft_tokens else 0)
    if len(h) > WINDOW:
        del h[: len(h) - WINDOW]


def cap_for(request, k_max: int) -> int:
    """Draft budget for the request's next step: min(client hint, rolling k)."""
    cap = k_max
    hint = getattr(request, "spec_max_tokens", None)
    if hint is not None:
        cap = min(cap, max(0, int(hint)))
    if cap <= 0 or not enabled_for(request):
        return cap
    rc = request.spec_roll_cap
    if rc is None:
        rc = start_for(cap)
    h = request.spec_roll_hist
    if len(h) >= MIN_SAMPLES:
        full = sum(h) / len(h)
        if full >= JUMP_UP and rc < cap:
            rc = cap; h.clear()            # e.g. JSON on DFlash2: straight to the ceiling
        elif full <= JUMP_DOWN and rc > KMIN:
            rc = KMIN; h.clear()           # e.g. prose at a high k: straight to the floor
        elif full >= UP and rc < cap:
            rc += 1; h.clear()             # climb one at a time: 2 -> 3 is the common, cheap, usually right move (session C: a
                                           # 2 -> 5 halving climb cost DFlash2 code 86 -> 75 tok/s); the 0.9 jump covers the rest
        elif full <= DOWN and rc > KMIN:
            rc -= max(1, (rc - KMIN + 1) // 2); h.clear()   # descend by half the remaining distance (7 -> 4 -> 3 -> 2)
    rc = min(rc, cap)
    request.spec_roll_cap = rc
    return max(rc, min(KMIN, cap))


if __name__ == "__main__":  # CPU self-test with a synthetic acceptance model
    import random

    class Req:
        def __init__(self, hint=None):
            self.spec_max_tokens = hint; self.spec_roll_cap = None; self.spec_roll_hist = []; self.spec_roll_on = None

    globals()["ENABLED"] = True
    rng = random.Random(0)
    def simulate(p_accept, k_max, steps=400, hint=None):
        """p_accept[i] = probability draft position i (0-based) is accepted given the previous ones were."""
        r = Req(hint); caps = []
        for _ in range(steps):
            k = cap_for(r, k_max); caps.append(k)
            acc = 0
            for i in range(k):
                if rng.random() < p_accept[i]: acc += 1
                else: break
            record_step(r, acc, k)
        return caps
    prose = simulate([0.72, 0.62, 0.58, 0.5], 4)          # prose-like: last positions rarely all accepted
    jsonl = simulate([0.97, 0.97, 0.96, 0.95], 4)         # json-like: nearly every step fully accepted
    hinted = simulate([0.97, 0.97, 0.96, 0.95], 4, hint=3)
    assert prose[-1] == 2, prose[-1]; assert jsonl[-1] == 4, jsonl[-1]; assert max(hinted) <= 3 and hinted[-1] == 3, hinted[-1]
    d7 = simulate([0.97, 0.97, 0.96, 0.95, 0.95, 0.94, 0.94], 7); assert d7[-1] == 7 and d7.index(7) <= 40, (d7[-1], d7.index(7))   # json-like on DFlash2: 3 -> 7 within a few windows
    p7 = simulate([0.55, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 7); assert p7[-1] == 2 and p7.index(2) <= 25, (p7[-1], p7.index(2))       # prose-like on DFlash2: down to 2 fast
    assert min(simulate([0.3, 0.3, 0.3, 0.3], 4)) >= 2   # never below KMIN
    assert start_for(4) == 3 and start_for(7) == 4 and start_for(2) == 2 and start_for(16) == 10, (start_for(4), start_for(7), start_for(16))
    globals()["ENABLED"] = False
    r = Req(2); assert cap_for(r, 4) == 2 and cap_for(Req(), 4) == 4
    r = Req(); r.spec_roll_on = True; assert cap_for(r, 7) == 4, cap_for(r, 7)                 # per-request opt-in on a rule-off server
    globals()["ENABLED"] = True
    r = Req(); r.spec_roll_on = False; assert cap_for(r, 7) == 7 and simulate([0.3] * 7, 7, 50)[-1] == 2  # per-request opt-out on a rule-on server
    r = Req(3); r.spec_roll_on = False; assert cap_for(r, 7) == 3
    globals()["ENABLED"] = False
    print("spec_roll self-test OK: prose ->", prose[-1], "json ->", jsonl[-1], "hinted(3) ->", hinted[-1], "d7 reached 7 at step", d7.index(7), "p7 reached 2 at step", p7.index(2))
