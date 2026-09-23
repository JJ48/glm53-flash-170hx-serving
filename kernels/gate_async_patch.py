#!/usr/bin/env python3
"""Fork patch 0006: confidence gate for the ASYNC scheduler (one-step lag).

The handler-side gate (patch 0002) is bypassed under async scheduling: the scheduler never asks the worker for the draft list, it
works from per-request counts. This patch moves the decision to the count path: after propose(), the last PP rank computes each
request's confident prefix length (first position whose DFlash2 confidence (patch 0005) is below the request's `spec_conf`
threshold; floored at VLLM_SPEC_CONF_FLOOR, default 2; threshold <= 0 = full block) and ships it with the step's output
(ModelRunnerOutput.spec_conf_prefix). The scheduler stores it on the request, and spec_roll.cap_for() bounds the next draft count
by it. Net: the drafts proposed at step N gate the count used at step N+2 (one-step lag; replay on logged steps kept ~75 % of the
per-step gate's gain: 105.4 vs 109.9 tok/s at tau 0.6, static-7 94.3).
Usage: gate_async_patch.py check|apply|revert   (own backups: *.d2gate.bak)
"""
import os, shutil, sys
V = "/usr/local/lib/python3.12/dist-packages/vllm"
OUT = f"{V}/v1/outputs.py"; ASY = f"{V}/v1/worker/gpu/async_utils.py"; RUN = f"{V}/v1/worker/gpu/model_runner.py"; SCH = f"{V}/v1/core/sched/scheduler.py"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
TAG = "patch 0006"

OLD_OUT = "    num_nans_in_logits: dict[str, int] | None = None\n"
NEW_OUT = OLD_OUT + "    # Fork (patch 0006): req_id -> confident prefix length of the drafts proposed this step (lagged confidence gate)\n    spec_conf_prefix: dict[str, int] | None = None\n"

OLD_ASY = "        self.model_runner_output.sampled_token_ids = sampled_token_ids\n"
NEW_ASY = OLD_ASY + """        _ev = getattr(self, "spec_prefix_event", None)  # patch 0006: lagged confidence gate
        if _ev is not None:
            _ev.synchronize()
            self.model_runner_output.spec_conf_prefix = dict(
                zip(self.model_runner_output.req_ids, self.spec_prefix_np.tolist())
            )
"""

OLD_RUN = "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"
NEW_RUN = OLD_RUN + """            _pc = getattr(self.speculator, "draft_token_confidence_probs", None)  # patch 0006: lagged confidence gate
            if _pc is not None and async_output is not None:
                import os as _os
                from vllm.v1.worker.gpu.async_utils import async_copy_to_np as _async_copy_to_np
                _n = input_batch.num_reqs
                _steps = self.num_speculative_steps
                _taus = torch.as_tensor(self.spec_conf_np[input_batch.idx_mapping_np], device=self.device)
                _ok = _pc[:_n, :_steps] >= _taus.unsqueeze(1)
                _nv = _ok.to(torch.int32).cumprod(dim=1).sum(dim=1)
                _floor = min(int(_os.environ.get("VLLM_SPEC_CONF_FLOOR", "2") or 0), _steps)
                _nv = torch.where(_taus > 0, torch.clamp(_nv, min=_floor), torch.full_like(_nv, _steps))
                with torch.cuda.stream(self.output_copy_stream):
                    self.output_copy_stream.wait_stream(self.main_stream)
                    async_output.spec_prefix_gpu = _nv
                    async_output.spec_prefix_np = _async_copy_to_np(_nv)
                    async_output.spec_prefix_event = torch.cuda.Event(blocking=True)
                    async_output.spec_prefix_event.record(self.output_copy_stream)
"""

OLD_SCH = "            scheduled_spec_token_ids = (\n                scheduler_output.scheduled_spec_decode_tokens.get(req_id)\n            )\n"
NEW_SCH = """            _pref = getattr(model_runner_output, "spec_conf_prefix", None)  # patch 0006: lagged confidence gate
            if _pref is not None and req_id in _pref and not output_is_stale:
                request.spec_conf_prefix = int(_pref[req_id])
""" + OLD_SCH

edits = [(OUT, OLD_OUT, NEW_OUT), (ASY, OLD_ASY, NEW_ASY), (RUN, OLD_RUN, NEW_RUN), (SCH, OLD_SCH, NEW_SCH)]

if MODE == "check":
    ok = True
    for p, o, _ in edits:
        s = open(p).read(); found = s.count(o) == 1; done = TAG in s
        ok &= found or done
        print(os.path.basename(p), "found" if found else ("patched" if done else "MISSING"))
    sys.exit(0 if ok else 1)
if MODE == "apply":
    for p, o, n in edits:
        s = open(p).read()
        if TAG in s:
            print(os.path.basename(p), "already patched"); continue
        assert s.count(o) == 1, f"anchor not unique/missing in {p}"
        bak = p + ".d2gate.bak"
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        open(p, "w").write(s.replace(o, n)); print(os.path.basename(p), "patched")
elif MODE == "revert":
    for p, _, _ in edits:
        bak = p + ".d2gate.bak"
        if os.path.exists(bak):
            shutil.copy2(bak, p); print(os.path.basename(p), "reverted")
        else:
            print(os.path.basename(p), "no backup")
