#!/usr/bin/env python3
"""Fork patch 0022: the first pipeline rank prepares the next step WHILE the drafter runs on the last rank.

What the traces of the production stack show (rank 0 and rank 3, 2026-09-16, single stream, 35.4 ms per step):
  - the last rank sends the step's result in three broadcasts: sampled tokens, (num_sampled, num_rejected), and -- about
    2.1 ms later, after the drafter has run -- the proposed draft tokens;
  - the receiver records ONE event behind all three, and the first rank's main stream waits for it before it runs
    post_update and the whole input preparation (about 39 launches and a handful of small host-to-device copies: 1.57 ms
    of GPU timeline in the trace, about 0.7 ms in reality because the profiler inflates eager sections) and only then
    the forward.
  Almost nothing in that preparation needs the draft token VALUES: positions, sequence lengths, slot mappings, block
  tables and every attention metadata builder depend on the counts only. The values are read once, by
  combine_sampled_and_draft_tokens, to fill input_ids.

  VLLM_PP_SPLIT_DRAFT_EVENT=1   (default = upstream; first pipeline rank only; off with VLLM_PP_HOP_V2 or PCP)
    - PPHandler.receive records an event after the second broadcast; the draft tokens get their own event.
    - get_prev_sampled_outputs waits for the first event only, so post_update and the input preparation run while the
      drafter is still busy on the last rank.
    - the draft tokens are scattered into the request state late: after the attention metadata is built, right before
      the forward inputs are assembled, the main stream waits for the draft event, scatters, and runs
      combine_sampled_and_draft_tokens a second time. That call is a pure function of the request state, so input_ids
      end up exactly as upstream builds them (the first call wrote the previous step's draft tokens into the same
      slots; they are overwritten before anything reads them).
    - rows whose request was freed between the consume and the late scatter are skipped (same generation counter the
      handler already uses), so a re-used row keeps the zeros add_request wrote.

Stream order guarantees the forward never sees stale draft tokens; nothing is added on the CPU side (no syncs).
Measured 2026-09-20: canary 19/19 exact (single stream and 8 at once), step rate +1.97 % ± 0.20 at single stream
(37.22 -> 36.51 ms/step), nothing at concurrency 8. Usage: pp_draft_event_patch.py check|apply|revert"""
import os, shutil, sys, py_compile
ROOT = os.environ.get("VLLM_GPU_WORKER_DIR", "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu")
F_PP = os.path.join(ROOT, "pp_utils.py")
F_MR = os.path.join(ROOT, "model_runner.py")
SUFFIX = ".bak0022"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
TAG = "patch 0022"

PP_REPL = [
 ('''    draft_tokens: torch.Tensor | None = None  # [num_reqs, max_sample_len - 1]
''',
  '''    draft_tokens: torch.Tensor | None = None  # [num_reqs, max_sample_len - 1]
    # patch 0022: when set, `event` covers the sampled tokens and the counts only and this one covers the draft tokens
    draft_event: torch.cuda.Event | None = None
'''),
 ('''        self.broadcast_stream = torch.cuda.Stream(device)
''',
  '''        self.broadcast_stream = torch.cuda.Stream(device)
        # patch 0022: set by the model runner (VLLM_PP_SPLIT_DRAFT_EVENT=1, first pipeline rank, no PCP)
        self.p0022_split = False
'''),
 ('''        self.main_stream.wait_event(slot.event)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=slot.draft_tokens,
        )
''',
  '''        self.main_stream.wait_event(slot.event)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=slot.draft_tokens,
            # patch 0022: with a draft_event the wait above covered the sampled tokens and the counts only; the consumer
            # waits for draft_event right before it needs the draft token values and re-checks freed rows then
            draft_event=slot.draft_event,
            p0022_recheck=(slot.idx_mapping_np, slot.gen_at_receive_np, exclude_mask),
        )
'''),
 ('''            self.broadcast_stream.wait_stream(self.main_stream)
            if _HOP_V2 and self.max_sample_len > 1:  # patch 0017: one broadcast for sampled+combined+draft
''',
  '''            self.broadcast_stream.wait_stream(self.main_stream)
            _p0022_event = None  # patch 0022
            if _HOP_V2 and self.max_sample_len > 1:  # patch 0017: one broadcast for sampled+combined+draft
'''),
 ('''                torch.distributed.broadcast(
                    combined, src=self.last_rank, group=self.broadcast_group
                )
                # Spec decode: 3rd broadcast on this group — the proposed draft tokens
''',
  '''                torch.distributed.broadcast(
                    combined, src=self.last_rank, group=self.broadcast_group
                )
                if self.p0022_split and self.max_sample_len > 1:
                    # patch 0022: sampled tokens + counts are complete here; the draft tokens follow ~2 ms later
                    _p0022_event = self.broadcast_stream.record_event()
                # Spec decode: 3rd broadcast on this group — the proposed draft tokens
'''),
 ('''            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
''',
  '''            event = self.broadcast_stream.record_event()
            _p0022_draft_event = None
            if _p0022_event is not None:  # patch 0022: `event` = sampled + counts, the draft tokens get their own
                event, _p0022_draft_event = _p0022_event, event
            num_sampled, num_rejected = combined.unbind(dim=0)
'''),
 ('''            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())
''',
  '''            gen_at_receive_np,
            draft_tokens,
            _p0022_draft_event,
        )
        return bool(need_sampled_mask.all())
'''),
]

MR_HELPERS = '''
# ---- patch 0022: the first pipeline rank prepares the next step while the drafter runs (VLLM_PP_SPLIT_DRAFT_EVENT=1) ----
import os as _p0022_os
_P0022 = (_p0022_os.environ.get("VLLM_PP_SPLIT_DRAFT_EVENT", "0") == "1"
          and _p0022_os.environ.get("VLLM_PP_HOP_V2", "0") != "1")
# ---- end patch 0022 helpers ----

'''
MR_HELPERS_ANCHOR = "import vllm.envs as envs\n"

MR_REPL = [
 ('''            self.pp_handler = PPHandler(
                max_num_reqs=self.max_num_reqs,
                num_speculative_steps=self.num_speculative_steps,
                device=self.device,
            )
''',
  '''            self.pp_handler = PPHandler(
                max_num_reqs=self.max_num_reqs,
                num_speculative_steps=self.num_speculative_steps,
                device=self.device,
            )
            # patch 0022: only the first rank's preparation is on the critical path (the others wait for hidden states)
            self.pp_handler.p0022_split = _P0022 and self.is_first_pp_rank
        self._p0022_pending = None  # (draft_tokens, idx_mapping, draft_event, recheck) of a consumed slot, not yet scattered
'''),
 ('''    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
''',
  '''    def _p0022_scatter(self, draft_tokens, idx_mapping) -> None:
        num_rows, k = draft_tokens.shape
        if num_rows > 0:
            dst = self.req_states.draft_tokens
            src = draft_tokens.to(dst.dtype)
            _scatter_draft_tokens_kernel[(num_rows,)](
                dst, dst.stride(0), src, src.stride(0), idx_mapping, k,
                BLOCK_K=max(1, triton.next_power_of_2(k)),
            )

    def _p0022_flush(self, input_batch) -> None:
        """patch 0022: late half of update_pp_decode_requests. Waits (on the GPU, not on the CPU) for the draft tokens,
        scatters them, and rebuilds input_ids for this step's batch when it verifies draft tokens."""
        if self._p0022_pending is None:
            return
        draft_tokens, idx_mapping, draft_event, recheck = self._p0022_pending
        self._p0022_pending = None
        assert self.pp_handler is not None
        idx_np, gen_np, excluded = recheck
        freed = self.pp_handler.req_idx_gen_np[idx_np] != gen_np
        if excluded is not None:
            freed = freed | excluded
        if freed.all():
            return
        if freed.any() and not (excluded is not None and (freed == excluded).all()):
            # a request was freed between the consume and now: its row may already belong to a new request
            idx_mapping = async_copy_to_gpu(np.where(freed, -1, idx_np), device=self.device)
        self.pp_handler.main_stream.wait_event(draft_event)
        self._p0022_scatter(draft_tokens, idx_mapping)
        if input_batch is not None and self.is_first_pp_rank and input_batch.num_draft_tokens > 0:
            # same call as in prepare_inputs, now with this step's draft tokens in the request state
            combine_sampled_and_draft_tokens(
                self.input_buffers.input_ids,
                input_batch.idx_mapping,
                self.req_states.last_sampled_tokens,
                input_batch.query_start_loc,
                input_batch.seq_lens,
                self.req_states.prefill_len.gpu,
                self.req_states.draft_tokens,
                input_batch.cu_num_logits,
                int(input_batch.logits_indices.shape[0]),
                self.model_state.num_new_sampled_tokens_per_step,
            )

    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
            self._p0022_flush(None)  # patch 0022: a step that returned before its forward leaves its scatter here
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
'''),
 ('''                draft_tokens = outputs.pop("draft_tokens", None)
                idx_mapping = outputs["idx_mapping"]
                self.postprocess_sampled(**outputs)
                if draft_tokens is not None:
''',
  '''                draft_tokens = outputs.pop("draft_tokens", None)
                _p0022_event = outputs.pop("draft_event", None)  # patch 0022
                _p0022_recheck = outputs.pop("p0022_recheck", None)
                idx_mapping = outputs["idx_mapping"]
                self.postprocess_sampled(**outputs)
                if draft_tokens is not None and _p0022_event is not None and self.pcp_manager is None:
                    # patch 0022: the values are not needed until input_ids are final; see _p0022_flush
                    self._p0022_pending = (draft_tokens, idx_mapping, _p0022_event, _p0022_recheck)
                    draft_tokens = None
                elif _p0022_event is not None:
                    self.pp_handler.main_stream.wait_event(_p0022_event)  # PCP: upstream order
                if draft_tokens is not None:
'''),
 ('''                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
            )

        input_ids = input_batch.input_ids
        inputs_embeds = None
''',
  '''                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
            )

        if self._p0022_pending is not None:  # patch 0022: before anything reads input_ids
            self._p0022_flush(None if dummy_run else input_batch)
        input_ids = input_batch.input_ids
        inputs_embeds = None
'''),
]


def patch_text(pp_src, mr_src):
    for old, _ in PP_REPL:
        assert pp_src.count(old) == 1, f"pp_utils.py: expected exactly one of:\n{old}"
    for old, _ in MR_REPL:
        assert mr_src.count(old) == 1, f"model_runner.py: expected exactly one of:\n{old}"
    assert mr_src.count(MR_HELPERS_ANCHOR) == 1, "model_runner.py: helper anchor not found"
    for name in ("async_copy_to_gpu", "combine_sampled_and_draft_tokens", "_scatter_draft_tokens_kernel"):
        assert name in mr_src, f"model_runner.py does not know {name}"
    for old, new in PP_REPL:
        pp_src = pp_src.replace(old, new)
    for old, new in MR_REPL:
        mr_src = mr_src.replace(old, new)
    mr_src = mr_src.replace(MR_HELPERS_ANCHOR, MR_HELPERS_ANCHOR + MR_HELPERS, 1)
    return pp_src, mr_src


if __name__ == "__main__":
    ps, ms = open(F_PP).read(), open(F_MR).read()
    a, b = TAG in ps, TAG in ms
    if MODE == "check":
        print("patched" if (a and b) else "unpatched" if not (a or b) else f"HALF PATCHED (pp_utils {a}, model_runner {b})")
        sys.exit(0 if a == b else 1)
    if MODE == "revert":
        missing = [f for f in (F_PP, F_MR) if not os.path.exists(f + SUFFIX)]
        if missing: print("no backup for", missing, "; nothing reverted"); sys.exit(1)
        if not (a and b): print("not patched; nothing reverted"); sys.exit(1)
        # the backups are whole files: refuse when something else changed these files after this patch went in
        exp = patch_text(open(F_PP + SUFFIX).read(), open(F_MR + SUFFIX).read())
        if (ps, ms) != exp:
            print("REFUSED: the installed files are not 'backup + patch 0022' (a later patch on top?). Revert that one first."); sys.exit(1)
        for f in (F_PP, F_MR): shutil.copy2(f + SUFFIX, f)
        print("reverted both files from *" + SUFFIX); sys.exit(0)
    if MODE == "apply":
        if a and b: print("already patched"); sys.exit(0)
        assert not (a or b), "half patched: revert first"
        np_, nm = patch_text(ps, ms)  # every assert runs before anything is written
        for f in (F_PP, F_MR): shutil.copy2(f, f + SUFFIX)  # always a fresh backup of what is installed NOW
        open(F_PP, "w").write(np_); open(F_MR, "w").write(nm)
        for f in (F_PP, F_MR): py_compile.compile(f, doraise=True)
        print("applied; backups *" + SUFFIX); sys.exit(0)
    print("usage: check|apply|revert"); sys.exit(2)
