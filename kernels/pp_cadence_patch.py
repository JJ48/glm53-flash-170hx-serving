#!/usr/bin/env python3
"""Fork patch 0012: let ONE request have two pipeline steps in flight (env-gated, default unchanged).

Measured 2026-09-15: at batch 1 under PP4, ~5 ms of every ~41 ms step is rank 0's host prep plus the engine's result ->
schedule -> dispatch round trip, all serial after rank 3's tail, because the fork forbids a request from being scheduled
again for `pp_size` steps ("to match the sampled-token broadcast slot ring cadence"). Those 5 ms need only the draft
COUNT, not the values: the runner fills placeholders on-device from the last rank's broadcast, and the non-last ranks'
main stream already waits on that broadcast's event before consuming the slot.

Three hard-wired uses of pp_size become one knob, VLLM_PP_STEP_CADENCE (1..pp_size; unset/0 = pp_size = today):
  pp_utils.PPHandler      ring depth = cadence (a slot filled at step T is consumed at step T+cadence)
  async_scheduler         next_decode_eligible_step = current_step + cadence
  config.max_concurrent_batches  = 2 when cadence == 1 (vanilla async: at most one step ahead; placeholders for more
                                    than one pending step would never be filled)
Usage: pp_cadence_patch.py check|apply|revert"""
import os, shutil, sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
TAG = "patch 0012"
HELPER = '''
def _pp_step_cadence(pp_size: int) -> int:  # patch 0012
    import os as _os
    try:
        v = int(_os.environ.get("VLLM_PP_STEP_CADENCE", "0") or 0)
    except ValueError:
        v = 0
    return pp_size if v <= 0 else max(1, min(v, pp_size))
'''
B = "/usr/local/lib/python3.12/dist-packages/vllm/"
FILES = {
 B + "v1/worker/gpu/pp_utils.py": [
   ("            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)\n",
    "            deque() if self.is_last_rank else deque([None] * _pp_step_cadence(get_pp_group().world_size))  # patch 0012\n"),
 ],
 B + "v1/core/sched/async_scheduler.py": [
   ("                request.next_decode_eligible_step = self.current_step + self.pp_size\n",
    "                request.next_decode_eligible_step = self.current_step + _pp_step_cadence(self.pp_size)  # patch 0012\n"),
 ],
 B + "config/vllm.py": [
   ("            if self.use_v2_model_runner:\n                return pp_size + 1\n",
    "            if self.use_v2_model_runner:\n                return 2 if _pp_step_cadence(pp_size) == 1 else pp_size + 1  # patch 0012\n"),
 ],
}
def insert_helper(src):
    # after the last top-level import line
    lines = src.split("\n"); last = 0
    for i, l in enumerate(lines[:400]):
        if l.startswith("import ") or l.startswith("from "): last = i
    return "\n".join(lines[: last + 1]) + "\n" + HELPER + "\n".join(lines[last + 1:])
if MODE == "check":
    for f, edits in FILES.items():
        s = open(f).read(); print(f.split("vllm/")[-1], "patched" if TAG in s else "unpatched", "| anchors:", [s.count(a) for a, _ in edits])
    sys.exit(0)
if MODE == "apply":
    for f, edits in FILES.items():
        s = open(f).read()
        if TAG in s: print(f.split("vllm/")[-1], "already patched"); continue
        for a, _ in edits: assert s.count(a) == 1, f"{f}: anchor count {s.count(a)}"
        if not os.path.exists(f + ".ppcad.bak"): shutil.copy2(f, f + ".ppcad.bak")
        for a, b in edits: s = s.replace(a, b)
        s = insert_helper(s); open(f, "w").write(s); print(f.split("vllm/")[-1], "applied")
elif MODE == "revert":
    for f in FILES:
        b = f + ".ppcad.bak"
        if os.path.exists(b): shutil.copy2(b, f); os.remove(b); print(f.split("vllm/")[-1], "reverted")
        else: print(f.split("vllm/")[-1], "no backup")
