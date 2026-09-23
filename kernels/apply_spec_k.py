#!/usr/bin/env python3
"""Fork patch 0003: per-request speculative k (vllm_xargs["spec_tokens"]), rolling-acceptance k (VLLM_SPEC_ROLL, per-request
vllm_xargs["spec_roll"] override) and a draft-confidence gate
(vllm_xargs["spec_conf"], server default VLLM_SPEC_CONF_MIN).

Idempotent exact-text edits on the installed vLLM (the club-170hx image, upstream 487ecf187 + overlay):
  request.py        parse the per-request cap
  scheduler.py      honor the cap for padded first-decode steps and at draft intake
  speculator.py     record the argmax probability of every draft token; per-batch active step count
  autoregressive/   draft only the steps the batch needs; zero the confidence of skipped slots
  utils.py          DraftTokensHandler hands the scheduler draft lists cut by confidence and cap
  model_runner.py   per-request cap/threshold arrays, active steps before propose, handler inputs
plus the new module spec_decode/confidence_gate.py (copied from files/).
Usage: apply_spec_k.py [--root DIST_PACKAGES] [--files FILES_DIR]
"""
import argparse, os, shutil, sys

MARK = "Per-request speculative budget (fork)"


def edit(path, replacements, marker=MARK):
    s = open(path).read()
    if marker in s:
        print(f"  {os.path.relpath(path)}: already patched"); return
    for old, new in replacements:
        if old not in s:
            sys.exit(f"PATCH FAILED: anchor not found in {path}:\n{old[:200]}")
        if s.count(old) != 1:
            sys.exit(f"PATCH FAILED: anchor not unique ({s.count(old)}x) in {path}:\n{old[:200]}")
        s = s.replace(old, new, 1)
    if marker not in s:
        sys.exit(f"PATCH BUG: marker missing after edit of {path}")
    open(path, "w").write(s); print(f"  {os.path.relpath(path)}: patched")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/usr/local/lib/python3.12/dist-packages")
    ap.add_argument("--files", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"))
    ap.add_argument("--revert", action="store_true", help="restore the six edited files from files/pristine (image = upstream 487ecf187 + overlay) and remove confidence_gate.py")
    a = ap.parse_args(); V = os.path.join(a.root, "vllm")
    if a.revert:
        pr = os.path.join(a.files, "pristine/vllm")
        for f in ("v1/request.py", "v1/core/sched/scheduler.py", "v1/worker/gpu/spec_decode/speculator.py",
                  "v1/worker/gpu/spec_decode/autoregressive/speculator.py", "v1/worker/gpu/spec_decode/utils.py", "v1/worker/gpu/model_runner.py",
                  "v1/core/sched/async_scheduler.py"):
            shutil.copyfile(os.path.join(pr, f), os.path.join(V, f)); print(f"  {f}: restored")
        for extra in ("v1/worker/gpu/spec_decode/confidence_gate.py", "v1/worker/gpu/spec_decode/static_budget.py", "v1/core/sched/spec_roll.py"):
            fp = os.path.join(V, extra)
            if os.path.exists(fp): os.remove(fp); print(f"  {extra}: removed")
        print("apply_spec_k: reverted"); return
    # 0. new module
    src = os.path.join(a.files, "vllm/v1/worker/gpu/spec_decode/confidence_gate.py")
    dst = os.path.join(V, "v1/worker/gpu/spec_decode/confidence_gate.py")
    shutil.copyfile(src, dst); print(f"  {os.path.relpath(dst)}: installed")
    src2 = os.path.join(a.files, "vllm/v1/worker/gpu/spec_decode/static_budget.py")
    dst2 = os.path.join(V, "v1/worker/gpu/spec_decode/static_budget.py")
    shutil.copyfile(src2, dst2); print(f"  {os.path.relpath(dst2)}: installed")
    src3 = os.path.join(a.files, "vllm/v1/core/sched/spec_roll.py")
    dst3 = os.path.join(V, "v1/core/sched/spec_roll.py")
    shutil.copyfile(src3, dst3); print(f"  {os.path.relpath(dst3)}: installed")

    # 1. request.py: per-request cap parsed once
    edit(os.path.join(V, "v1/request.py"), [(
        "        self.spec_token_ids: list[int] = []\n",
        "        self.spec_token_ids: list[int] = []\n"
        "        # " + MARK + ": vllm_xargs[\"spec_tokens\"] caps this request's drafts;\n"
        "        # the scheduler clamps it to the server's num_speculative_tokens.\n"
        "        self.spec_max_tokens: int | None = _parse_spec_max_tokens(\n"
        "            getattr(sampling_params, \"extra_args\", None)\n"
        "        )\n"),
        ("\nclass Request:\n",
         "\ndef _parse_spec_max_tokens(extra_args) -> int | None:\n"
         "    if not extra_args or extra_args.get(\"spec_tokens\") is None:\n"
         "        return None\n"
         "    try:\n"
         "        return max(0, int(extra_args[\"spec_tokens\"]))\n"
         "    except (TypeError, ValueError):\n"
         "        return None\n\n\nclass Request:\n")])

    # 2. scheduler.py: padded first decode step + intake cap
    edit(os.path.join(V, "v1/core/sched/scheduler.py"), [
        ("                        num_new_tokens = 1 + self.num_spec_tokens\n",
         "                        num_new_tokens = 1 + self._spec_tokens_for(request)\n"),
        ("                    ] * self.num_spec_tokens\n",
         "                    ] * self._spec_tokens_for(request)\n"),
        ("            request.spec_token_ids = spec_token_ids\n",
         "            cap = getattr(request, \"spec_max_tokens\", None)\n"
         "            if cap is not None and len(spec_token_ids) > cap:\n"
         "                spec_token_ids = spec_token_ids[:cap]\n"
         "            request.spec_token_ids = spec_token_ids\n"),
        ("    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:\n",
         "    def _spec_tokens_for(self, request) -> int:\n"
         "        # " + MARK + ": per-request cap on scheduled drafts.\n"
         "        cap = getattr(request, \"spec_max_tokens\", None)\n"
         "        if cap is None:\n"
         "            return self.num_spec_tokens\n"
         "        return max(0, min(int(cap), self.num_spec_tokens))\n\n"
         "    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:\n")])

    # 3. base speculator: confidence buffer + recording, active step count
    edit(os.path.join(V, "v1/worker/gpu/spec_decode/speculator.py"), [
        ("from typing import Any\n", "import os\nfrom typing import Any\n"),
        ("from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample\n",
         "from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample\n"
         "from vllm.v1.worker.gpu.spec_decode.confidence_gate import draft_probs_from_logits\n"),
        ("        self.supports_mm_inputs = False\n",
         "        self.supports_mm_inputs = False\n\n"
         "        # " + MARK + ": argmax probability of every draft token (batch\n"
         "        # row x step), read by DraftTokensHandler to cut low-confidence draft\n"
         "        # tails; and the number of draft steps the current batch wants\n"
         "        # (max per-request spec_tokens), set by the runner before propose().\n"
         "        self.draft_token_confidence_probs = torch.ones(\n"
         "            self.max_num_reqs,\n"
         "            self.num_speculative_steps,\n"
         "            dtype=torch.float32,\n"
         "            device=device,\n"
         "        )\n"
         "        self.compute_draft_confidence = (\n"
         "            os.environ.get(\"VLLM_SPEC_DRAFT_CONFIDENCE\", \"1\") == \"1\"\n"
         "        )\n"
         "        self.num_active_steps = self.num_speculative_steps\n"),
        ("        if draft_logits is not None:\n"
         "            logits = self.model.compute_logits(hidden_states)\n"
         "            # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise\n",
         "        if draft_logits is not None:\n"
         "            logits = self.model.compute_logits(hidden_states)\n"
         "            self._record_draft_confidence(logits, draft_step)\n"
         "            # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise\n"),
        ("                use_fp64=self.use_fp64_gumbel,\n"
         "            )\n"
         "        return self._greedy_sample_draft(hidden_states)\n",
         "                use_fp64=self.use_fp64_gumbel,\n"
         "            )\n"
         "        if self.use_local_argmax_reduction or not self.compute_draft_confidence:\n"
         "            return self._greedy_sample_draft(hidden_states)\n"
         "        logits = self.model.compute_logits(hidden_states)\n"
         "        self._record_draft_confidence(logits, draft_step)\n"
         "        return logits.argmax(dim=-1)\n\n"
         "    def _record_draft_confidence(\n"
         "        self, logits: torch.Tensor, draft_step: torch.Tensor\n"
         "    ) -> None:\n"
         "        # Graph-safe: pure tensor ops, column chosen by the on-device step index.\n"
         "        if not self.compute_draft_confidence:\n"
         "            return\n"
         "        n = logits.shape[0]\n"
         "        probs = draft_probs_from_logits(logits).to(torch.float32)\n"
         "        col = draft_step.to(torch.int64).reshape(1, 1).expand(n, 1)\n"
         "        self.draft_token_confidence_probs[:n].scatter_(1, col, probs.unsqueeze(1))\n")])

    # 4. autoregressive speculator: draft only the active steps
    edit(os.path.join(V, "v1/worker/gpu/spec_decode/autoregressive/speculator.py"), [
        ("        if self.num_speculative_steps == 1:\n"
         "            # Early exit.\n"
         "            return self.draft_tokens[:num_reqs, :1]\n",
         "        # " + MARK + ": draft only the steps the batch's per-request caps\n"
         "        # need; slots past that are marked zero-confidence so the handler\n"
         "        # never hands them to the scheduler.\n"
         "        num_active_steps = max(\n"
         "            1, min(self.num_speculative_steps, int(self.num_active_steps))\n"
         "        )\n"
         "        if num_active_steps < self.num_speculative_steps:\n"
         "            self.draft_token_confidence_probs[:num_reqs, num_active_steps:].zero_()\n"
         "        if self.num_speculative_steps == 1:\n"
         "            # Early exit.\n"
         "            return self.draft_tokens[:num_reqs, :1]\n"
         "        if num_active_steps == 1:\n"
         "            return self.draft_tokens[:num_reqs]\n"),
        ("        for step in range(1, self.num_speculative_steps):\n"
         "            # Rebuild every step when positions advance, or just once\n",
         "        num_steps = max(1, min(self.num_speculative_steps, int(self.num_active_steps)))\n"
         "        for step in range(1, num_steps):\n"
         "            # Rebuild every step when positions advance, or just once\n")])

    # 5. DraftTokensHandler: confidence/cap-aware draft lists
    old_cls = open(os.path.join(V, "v1/worker/gpu/spec_decode/utils.py")).read()
    start = old_cls.index("class DraftTokensHandler:"); end = old_cls.index("        return DraftTokenIds(self.req_ids, draft_token_ids)\n", start) + len("        return DraftTokenIds(self.req_ids, draft_token_ids)\n")
    new_cls = '''class DraftTokensHandler:
    """Hands the scheduler each request's draft list (real ids when structured
    outputs need them, placeholders otherwise). __MARK__: the lists are
    cut to the request's spec_tokens cap and, when a confidence threshold is set,
    at the first draft token whose probability falls below it (acceptance is
    sequential, so everything after that token would be verified for nothing).
    """

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        self.conf_np: np.ndarray | None = None
        self.caps_np: np.ndarray | None = None
        self.taus_np: np.ndarray | None = None

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        confidences: torch.Tensor | None = None,
        caps_np: np.ndarray | None = None,
        taus_np: np.ndarray | None = None,
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        self.caps_np = caps_np
        self.taus_np = taus_np
        need_ids = input_batch.has_structured_output_reqs
        gate = (
            confidences is not None
            and taus_np is not None
            and len(taus_np) > 0
            and float(np.max(taus_np)) > 0.0
        )
        if not need_ids and not gate:
            # Nothing to transfer: the scheduler only needs counts.
            self.draft_tokens_np = None
            self.conf_np = None
            return

        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            if need_ids:
                # For spec decoding + structured outputs, we must transfer the
                # draft tokens back to the scheduler for grammar validation.
                self.draft_tokens_np = async_copy_to_np(draft_tokens)
                # draft_tokens is a temporary allocation on the main stream and read
                # here on copy_stream; without record_stream, the caching allocator
                # may reuse its memory before the async copy executes.
                draft_tokens.record_stream(self.copy_stream)
            else:
                self.draft_tokens_np = None
            if gate:
                conf = confidences[: input_batch.num_reqs]
                self.conf_np = async_copy_to_np(conf)
                conf.record_stream(self.copy_stream)
            else:
                self.conf_np = None
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None or self.conf_np is not None:
            self.copy_event.synchronize()
        if self.draft_tokens_np is not None:
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        draft_token_ids = gate_draft_lists(
            draft_token_ids, self.conf_np, self.caps_np, self.taus_np
        )
        return DraftTokenIds(self.req_ids, draft_token_ids)
'''.replace("__MARK__", MARK)
    path = os.path.join(V, "v1/worker/gpu/spec_decode/utils.py")
    if MARK in old_cls:
        print(f"  {os.path.relpath(path)}: already patched")
    else:
        s = old_cls[:start] + new_cls + old_cls[end:]
        s = s.replace("from vllm.v1.worker.gpu.input_batch import InputBatch\n",
                      "from vllm.v1.worker.gpu.input_batch import InputBatch\n"
                      "from vllm.v1.worker.gpu.spec_decode.confidence_gate import gate_draft_lists\n", 1)
        open(path, "w").write(s); print(f"  {os.path.relpath(path)}: patched")

    # 6. model runner: per-request arrays, active steps, handler inputs
    edit(os.path.join(V, "v1/worker/gpu/model_runner.py"), [
        ("from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler\n",
         "from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler\n"
         "from vllm.v1.worker.gpu.spec_decode.confidence_gate import (\n"
         "    parse_spec_conf,\n    parse_spec_tokens,\n    server_default_conf_min,\n)\n"),
        ("        self.draft_tokens_handler = DraftTokensHandler(self.device)\n",
         "        self.draft_tokens_handler = DraftTokensHandler(self.device)\n"
         "        # " + MARK + ": static cap and confidence threshold per request\n"
         "        # slot (vllm_xargs spec_tokens / spec_conf), gathered per batch for the\n"
         "        # draft handler and the drafter's active step count.\n"
         "        self.spec_cap_np = np.full(\n"
         "            self.max_num_reqs, max(self.num_speculative_steps, 0), dtype=np.int32\n"
         "        )\n"
         "        self.spec_conf_np = np.full(\n"
         "            self.max_num_reqs, server_default_conf_min(), dtype=np.float32\n"
         "        )\n"),
        ("            req_index = self.req_states.req_id_to_index[req_id]\n"
         "            if self.adaptive_verification is not None:\n",
         "            req_index = self.req_states.req_id_to_index[req_id]\n"
         "            extra_args = (\n"
         "                getattr(sampling_params, \"extra_args\", None) if sampling_params else None\n"
         "            )\n"
         "            cap = parse_spec_tokens(extra_args, self.num_speculative_steps)\n"
         "            self.spec_cap_np[req_index] = (\n"
         "                self.num_speculative_steps if cap is None else cap\n"
         "            )\n"
         "            self.spec_conf_np[req_index] = parse_spec_conf(\n"
         "                extra_args, server_default_conf_min()\n"
         "            )\n"
         "            if self.adaptive_verification is not None:\n"),
        ("            draft_tokens = self.speculator.propose(\n"
         "                input_batch,\n"
         "                attn_metadata,\n"
         "                slot_mappings_by_layer,\n",
         "            # Draft only as many steps as the batch needs: a request's scheduled draft\n"
         "            # count (the scheduler's rolling/hinted cap) when it has one, else its hint cap.\n"
         "            _caps = self.spec_cap_np[input_batch.idx_mapping_np]\n"
         "            _sched = input_batch.num_draft_tokens_per_req\n"
         "            if _sched is not None and len(_sched) == len(_caps):\n"
         "                _need = [int(d) if d > 0 else int(c) for d, c in zip(_sched, _caps)]\n"
         "            else:\n"
         "                _need = [int(c) for c in _caps]\n"
         "            self.speculator.num_active_steps = max(1, max(_need) if _need else self.num_speculative_steps)\n"
         "            draft_tokens = self.speculator.propose(\n"
         "                input_batch,\n"
         "                attn_metadata,\n"
         "                slot_mappings_by_layer,\n"),
        ("            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n",
         "            self.speculator.num_active_steps = self.num_speculative_steps\n"
         "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"),
        ("            self.draft_tokens_handler.set_draft_tokens(\n"
         "                input_batch,\n"
         "                self.req_states.draft_tokens[input_batch.idx_mapping],\n",
         "            self.draft_tokens_handler.set_draft_tokens(\n"
         "                input_batch,\n"
         "                self.req_states.draft_tokens[input_batch.idx_mapping],\n"
         "                confidences=getattr(\n"
         "                    self.speculator, \"draft_token_confidence_probs\", None\n"
         "                ),\n"
         "                caps_np=self.spec_cap_np[input_batch.idx_mapping_np],\n"
         "                taus_np=self.spec_conf_np[input_batch.idx_mapping_np],\n")])
    # 7. model runner, second iteration: runner-side compaction of the verified draft slots (async/PP scheduler path)
    edit(os.path.join(V, "v1/worker/gpu/model_runner.py"), [
        ("from vllm.v1.worker.gpu.spec_decode.confidence_gate import (\n",
         "import os as _spec_os\n"
         "from vllm.v1.worker.gpu.spec_decode.static_budget import StaticDraftBudget\n"
         "from vllm.v1.worker.gpu.spec_decode.confidence_gate import (\n"),
        ("            max_total_logits=get_max_chunk_logits(self.vocab_size),\n"
         "        )\n"
         "\n"
         "        self.block_tables = BlockTables(\n",
         "            max_total_logits=get_max_chunk_logits(self.vocab_size),\n"
         "        )\n"
         "        # Static draft budget (fork): runner-side cap on the verified draft slots\n"
         "        # (vllm_xargs spec_tokens). The async/PP scheduler fixes k placeholder slots\n"
         "        # per request; unverified slots count as rejected, so its accounting holds.\n"
         "        self.static_budget = (\n"
         "            StaticDraftBudget(\n"
         "                self.req_states,\n"
         "                self.model_state.num_new_sampled_tokens_per_step,\n"
         "                self.spec_cap_np,\n"
         "            )\n"
         "            if self.num_speculative_steps > 0\n"
         "            and self.adaptive_verification is None\n"
         "            and _spec_os.environ.get(\"VLLM_SPEC_STATIC_BUDGET\", \"1\") == \"1\"\n"
         "            else None\n"
         "        )\n"
         "\n"
         "        self.block_tables = BlockTables(\n"),
        ("        if self.adaptive_verification is not None and draft_tokens:\n"
         "            num_toks = self.adaptive_verification.get_num_tokens(\n"
         "                num_tokens_per_req, draft_tokens\n"
         "            )\n",
         "        if self.adaptive_verification is not None and draft_tokens:\n"
         "            num_toks = self.adaptive_verification.get_num_tokens(\n"
         "                num_tokens_per_req, draft_tokens\n"
         "            )\n"
         "        if self.static_budget is not None and draft_tokens:\n"
         "            num_toks = self.static_budget.get_num_tokens(\n"
         "                num_tokens_per_req, draft_tokens, num_toks\n"
         "            )\n"),
        ("        # Get query_start_loc.\n"
         "        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)\n",
         "        if (\n"
         "            self.static_budget is not None\n"
         "            and num_draft_tokens_per_req is not None\n"
         "            and self.static_budget.pending\n"
         "        ):\n"
         "            # Static draft budget (fork): exact per-request compaction on both CPU and GPU.\n"
         "            num_scheduled_tokens_np, cu_num_logits_np, total_num_draft_tokens = (\n"
         "                self.static_budget.compact(\n"
         "                    num_draft_tokens_per_req, num_scheduled_tokens_np, req_ids\n"
         "                )\n"
         "            )\n"
         "            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)\n"
         "            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens\n"
         "        # Get query_start_loc.\n"
         "        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)\n")],
        marker="Static draft budget (fork)")
    # 8. rolling-acceptance k: request state, per-step recording, and the cap where the async scheduler hands out placeholders
    edit(os.path.join(V, "v1/request.py"), [(
        "        self.spec_max_tokens: int | None = _parse_spec_max_tokens(\n"
        "            getattr(sampling_params, \"extra_args\", None)\n"
        "        )\n",
        "        self.spec_max_tokens: int | None = _parse_spec_max_tokens(\n"
        "            getattr(sampling_params, \"extra_args\", None)\n"
        "        )\n"
        "        # Rolling speculative k (fork): current rolling cap and the recent\n"
        "        # fully-accepted-step history (see vllm/v1/core/sched/spec_roll.py).\n"
        "        self.spec_roll_cap: int | None = None\n"
        "        self.spec_roll_hist: list[int] = []\n"
        "        self.spec_roll_on: bool | None = _parse_spec_roll(getattr(sampling_params, \"extra_args\", None))\n"),
        ("\ndef _parse_spec_max_tokens(extra_args) -> int | None:\n",
         "\ndef _parse_spec_roll(extra_args) -> bool | None:\n"
         "    # Rolling speculative k (fork): vllm_xargs[\"spec_roll\"] = 0/1 overrides the server's VLLM_SPEC_ROLL for this request.\n"
         "    if not extra_args or extra_args.get(\"spec_roll\") is None:\n"
         "        return None\n"
         "    try:\n"
         "        return bool(int(extra_args[\"spec_roll\"]))\n"
         "    except (TypeError, ValueError):\n"
         "        return None\n\n"
         "\ndef _parse_spec_max_tokens(extra_args) -> int | None:\n")], marker="Rolling speculative k (fork)")
    edit(os.path.join(V, "v1/core/sched/scheduler.py"), [
        ("from vllm.v1.spec_decode.metrics import SpecDecodingStats\n",
         "from vllm.v1.spec_decode.metrics import SpecDecodingStats\n"
         "from vllm.v1.core.sched import spec_roll as _spec_roll   # Rolling speculative k (fork)\n"),
        ("                num_rejected = num_draft_tokens - num_accepted\n",
         "                num_rejected = num_draft_tokens - num_accepted\n"
         "                if not output_is_stale:\n"
         "                    _spec_roll.record_step(request, num_accepted, num_draft_tokens)\n"),
        ("            cap = getattr(request, \"spec_max_tokens\", None)\n"
         "            if cap is not None and len(spec_token_ids) > cap:\n"
         "                spec_token_ids = spec_token_ids[:cap]\n",
         "            cap = _spec_roll.cap_for(request, self.num_spec_tokens)\n"
         "            if len(spec_token_ids) > cap:\n"
         "                spec_token_ids = spec_token_ids[:cap]\n")], marker="Rolling speculative k (fork)")
    edit(os.path.join(V, "v1/core/sched/async_scheduler.py"), [
        ("from vllm.v1.core.sched.scheduler import Scheduler\n",
         "from vllm.v1.core.sched import spec_roll as _spec_roll   # Rolling speculative k (fork)\n"
         "from vllm.v1.core.sched.scheduler import Scheduler\n"),
        ("            request.spec_token_ids = self._spec_token_placeholders\n",
         "            # Per-request draft budget: client hint (vllm_xargs spec_tokens) and the\n"
         "            # rolling-acceptance k, both sliced from the shared placeholder list.\n"
         "            request.spec_token_ids = self._spec_token_placeholders[\n"
         "                : _spec_roll.cap_for(request, len(self._spec_token_placeholders))\n"
         "            ]\n")], marker="Rolling speculative k (fork)")
    print("apply_spec_k: done")


if __name__ == "__main__":
    main()
