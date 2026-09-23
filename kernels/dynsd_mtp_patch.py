#!/usr/bin/env python3
"""Fork patch 0009: let an autoregressive speculator (MTP/EAGLE) coexist with a dynamic-SD schedule.

Measured on the box 2026-09-15: with MTP at a k=5 ceiling, a step that drafts FEWER than the ceiling costs +3..6.5 ms,
while a step at the ceiling costs nothing extra. Cause: cudagraph_dispatcher sets uniform_decode_query_len = 1 +
num_speculative_tokens, one fixed value, and the uniform-decode gate requires max_num_scheduled_tokens to equal it, so any
sub-ceiling k falls off the FULL-graph path onto PIECEWISE. That penalty is roughly one whole drafted token and it is what
makes the rolling-k rule unprofitable on MTP (k=5 rule 87.7 vs k=5 static 90.2 tok/s) even though the rule picks the right
k: it wins +16 % on prose, holds on counting, and loses on everything it has to step down for.

DFlash2 solved this with num_speculative_tokens_per_batch_size (launch time 4.7 -> 0.1 ms/rank). MTP cannot use it today
because the drafter builds its OWN cudagraph manager with decode_query_len 1 (it emits one token per pass whatever k is).
The recovered num_new_sampled_tokens_per_step is then 1 - num_speculative_tokens, i.e. <= 0 for any real k, so every
schedule entry maps to a query length of 0 or less and round_up() raises ZeroDivisionError at boot.

Only the VERIFY model's manager needs one graph per k; the drafter's manager needs exactly one shape. This guards the
dynamic branch on a positive recovered value and falls back to the single shape otherwise, which is a no-op for DFlash2
(where the value is 1) and unblocks the schedule for MTP.

Usage: dynsd_mtp_patch.py check|apply|revert
"""
import os, shutil, sys

P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/cudagraph_utils.py"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"

OLD = """            # Each entry is (range_start, range_end, num_speculative_tokens).
            decode_query_lens = [
                x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
            ]
"""
NEW = """            # Each entry is (range_start, range_end, num_speculative_tokens).
            if num_new_sampled_tokens_per_step < 1:   # DYNSDMTP patch 0009
                # An autoregressive drafter (MTP/EAGLE) builds its own manager with decode_query_len 1, so the recovered
                # per-step count is <= 0 here and every entry would map to a query length of 0 or less (ZeroDivisionError
                # in round_up below). That manager only ever runs one token per pass: one shape is all it needs. The
                # verify model's manager has decode_query_len = 1 + num_speculative_tokens, recovers 1, and is unaffected.
                decode_query_lens = [self.decode_query_len]
            else:
                decode_query_lens = [
                    x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
                ]
"""

s = open(P).read()
if MODE == "check":
    print("anchor", "found" if s.count(OLD) == 1 else ("patched" if "DYNSDMTP" in s else "MISSING"))
    sys.exit(0)
if MODE == "apply":
    if "DYNSDMTP" in s:
        print("already patched"); sys.exit(0)
    assert s.count(OLD) == 1, f"anchor not unique: {s.count(OLD)}"
    if not os.path.exists(P + ".dynsdmtp.bak"):
        shutil.copy2(P, P + ".dynsdmtp.bak")
    open(P, "w").write(s.replace(OLD, NEW))
    print(f"applied; backup {P}.dynsdmtp.bak")
elif MODE == "revert":
    b = P + ".dynsdmtp.bak"
    if os.path.exists(b):
        shutil.copy2(b, P); print("reverted")
    else:
        print("no backup")
