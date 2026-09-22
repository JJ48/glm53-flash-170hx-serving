#!/usr/bin/env python3
"""Fork patch 0026: the rolling draft-length rule only while ONE request is running; with company, everyone drafts to the ceiling.

Found 2026-09-20 (window e): Spec-Bench at concurrency 8, same code, same night, temperature 0:
    rolling rule on  (production config)  158-164 tok/s, TPOT 45-48 ms, stage timer wall 9.7-9.9 ms per rank per step
    rolling rule off (static k = 7)       217 tok/s,     TPOT 30 ms,    stage timer wall 3.6 ms (launch 1.1 ms)
The rule picks k per REQUEST from a batch-1 step-time table. With more than four requests running, a pipeline micro-batch
holds two requests, their k differ, the batch is no longer uniform, and the step leaves the FULL CUDA-graph path (one graph
per uniform (requests, tokens per request) shape) for the piecewise path: ~7 ms of launches per rank per step and an idle
GPU between the pieces. That costs far more than the verify tokens the rule saves, and at higher concurrency the extra
verified tokens are cheap anyway (the step is dominated by reading the experts, which the larger batch reads regardless).

  VLLM_SPEC_ROLL_SOLO=N   (default 0 = rule for every request, as before; 1 = rule only while one request is running;
                           4 = while at most four are, i.e. while every pipeline micro-batch can hold a single request)
    in AsyncScheduler._update_after_schedule (the path this server runs: async scheduling sizes the next step's draft
    placeholders there) and in Scheduler.update_draft_token_ids (the synchronous path): when more than N requests are
    running, the request's drafts are cut to the client hint / server ceiling only, not to the rolling k.
    (The first version of this patch, 2026-09-20 04:05 UTC, changed only the synchronous path and measured nothing:
    153.6 tok/s at concurrency 8, launch 12.7 ms per step, exactly like the rule.) The rule keeps observing acceptance (record_step is
    untouched), so a request that becomes the only one again continues from an up-to-date estimate.

Single-stream behaviour is unchanged (the rule's prose / chat advantage stays). Usage: spec_roll_solo_patch.py check|apply|revert"""
import os, shutil, sys, py_compile
ROOT = os.environ.get("VLLM_SCHED_DIR", "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched")
F_S = os.path.join(ROOT, "scheduler.py")
F_A = os.path.join(ROOT, "async_scheduler.py")
SUFFIX = ".bak0026"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
TAG = "patch 0026"

S_REPL = [
 ('''from vllm.v1.core.sched import spec_roll as _spec_roll   # Rolling speculative k (fork)
''',
  '''from vllm.v1.core.sched import spec_roll as _spec_roll   # Rolling speculative k (fork)
import os as _p0026_os  # patch 0026: rolling k only while at most N requests are running (VLLM_SPEC_ROLL_SOLO=N, 0 = off)
try:
    _P0026_SOLO = max(0, int(_p0026_os.environ.get("VLLM_SPEC_ROLL_SOLO", "0") or 0))
except ValueError:
    _P0026_SOLO = 0
'''),
 ('''            cap = _spec_roll.cap_for(request, self.num_spec_tokens)
            if len(spec_token_ids) > cap:
''',
  '''            cap = _spec_roll.cap_for(request, self.num_spec_tokens)
            if _P0026_SOLO and len(self.running) > _P0026_SOLO:
                # patch 0026: mixed k inside a micro-batch leaves the FULL CUDA-graph path; keep batches uniform
                cap = self._spec_tokens_for(request)
            if len(spec_token_ids) > cap:
'''),
]
A_REPL = [
 ('''from vllm.v1.core.sched import spec_roll as _spec_roll   # Rolling speculative k (fork)
''',
  '''from vllm.v1.core.sched import spec_roll as _spec_roll   # Rolling speculative k (fork)
from vllm.v1.core.sched.scheduler import _P0026_SOLO  # patch 0026
'''),
 ('''            request.spec_token_ids = self._spec_token_placeholders[
                : _spec_roll.cap_for(request, len(self._spec_token_placeholders))
            ]
''',
  '''            _p0026_cap = _spec_roll.cap_for(request, len(self._spec_token_placeholders))  # keeps the rule's state current
            if _P0026_SOLO and len(self.running) > _P0026_SOLO:
                # patch 0026: with company every request drafts to the ceiling, so micro-batches stay uniform (FULL graphs)
                _p0026_cap = min(len(self._spec_token_placeholders), self._spec_tokens_for(request))
            request.spec_token_ids = self._spec_token_placeholders[:_p0026_cap]
'''),
]


def patch_text(s_src, a_src):
    for old, _ in S_REPL:
        assert s_src.count(old) == 1, f"scheduler.py: expected exactly one of:\n{old}"
    for old, _ in A_REPL:
        assert a_src.count(old) == 1, f"async_scheduler.py: expected exactly one of:\n{old}"
    assert "def _spec_tokens_for(self, request)" in s_src, "scheduler.py: _spec_tokens_for not found"
    for old, new in S_REPL: s_src = s_src.replace(old, new)
    for old, new in A_REPL: a_src = a_src.replace(old, new)
    return s_src, a_src


if __name__ == "__main__":
    ss, aa = open(F_S).read(), open(F_A).read()
    ps, pa = TAG in ss, TAG in aa
    if MODE == "check":
        print("patched" if (ps and pa) else "unpatched" if not (ps or pa) else f"HALF PATCHED (scheduler {ps}, async_scheduler {pa})")
        sys.exit(0 if ps == pa else 1)
    if MODE == "revert":
        if not (ps and pa): print("not patched; nothing reverted"); sys.exit(1)
        missing = [f for f in (F_S, F_A) if not os.path.exists(f + SUFFIX)]
        if missing: print("no backup for", missing, "; nothing reverted"); sys.exit(1)
        if (ss, aa) != patch_text(open(F_S + SUFFIX).read(), open(F_A + SUFFIX).read()):
            print("REFUSED: the installed files are not 'backup + patch 0026'. Revert whatever sits on top first."); sys.exit(1)
        for f in (F_S, F_A): shutil.copy2(f + SUFFIX, f)
        print("reverted both files from *" + SUFFIX); sys.exit(0)
    if MODE == "apply":
        if ps and pa: print("already patched"); sys.exit(0)
        assert not (ps or pa), "half patched: revert first"
        ns, na = patch_text(ss, aa)
        for f in (F_S, F_A): shutil.copy2(f, f + SUFFIX)
        open(F_S, "w").write(ns); open(F_A, "w").write(na)
        for f in (F_S, F_A): py_compile.compile(f, doraise=True)
        print("applied; backups *" + SUFFIX); sys.exit(0)
    print("usage: check|apply|revert"); sys.exit(2)
