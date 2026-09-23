# SPDX-License-Identifier: Apache-2.0
"""Runner-side per-request draft budget (fork patch 0003, second iteration).

Under pipeline parallelism the fork runs the asynchronous scheduler: every decode request gets k placeholder draft slots
before the drafts exist, and the runner's draft lists are re-padded to k when they arrive, so trimming them on the
scheduler side changes nothing. The place where the verify batch can shrink is the runner's prepare_inputs, where
upstream's adaptive verification already compacts the batch below the scheduled slots (unverified slots count as rejected,
so the scheduler's accounting stays valid). This class does the static version of that: each request verifies at most
``spec_tokens`` (vllm_xargs) of its k scheduled drafts. Caps come from sampling params, which every PP rank holds, so all
ranks build the same batch without communication. CPU and GPU layouts are both computed from the same exact counts.
"""
from __future__ import annotations

import numpy as np


class StaticDraftBudget:
    def __init__(self, req_states, num_bonus_tokens: int, spec_cap_np: np.ndarray):
        self.req_states = req_states
        self.num_bonus_tokens = int(num_bonus_tokens)
        self.spec_cap_np = spec_cap_np  # [max_num_reqs] int32, owned by the runner (filled in add_requests)
        self.pending = False
        self._kept: dict[str, int] = {}

    def get_num_tokens(self, num_tokens_per_req, draft_tokens, num_toks: int) -> int:
        """Scheduler-order pass: remember which requests are capped below their scheduled drafts and return the
        compacted batch token count (unchanged when nothing is capped, which keeps the uniform fast path)."""
        self.pending = False
        self._kept = {}
        total_cut = 0
        for req_id in num_tokens_per_req:
            drafts = len(draft_tokens.get(req_id, ()))
            if drafts == 0:
                continue
            idx = self.req_states.req_id_to_index.get(req_id)
            if idx is None:
                continue
            kept = min(drafts, max(int(self.spec_cap_np[idx]), 0))
            if kept < drafts:
                self._kept[req_id] = kept
                total_cut += drafts - kept
        if total_cut == 0:
            return int(num_toks)
        self.pending = True
        return int(num_toks) - total_cut

    def compact(self, num_draft_tokens_per_req: np.ndarray, num_scheduled_tokens: np.ndarray, req_ids):
        """Batch-order pass: per-request compacted token counts, cumulative logits offsets, total verified drafts."""
        self.pending = False
        kept = num_draft_tokens_per_req.astype(np.int64).copy()
        for i, req_id in enumerate(req_ids):
            k = self._kept.get(req_id)
            if k is not None:
                kept[i] = min(kept[i], k)
        compacted = (num_scheduled_tokens.astype(np.int64) - (num_draft_tokens_per_req.astype(np.int64) - kept)).astype(
            num_scheduled_tokens.dtype
        )
        cu = np.empty(len(req_ids) + 1, dtype=np.int32)
        cu[0] = 0
        np.cumsum(kept + self.num_bonus_tokens, out=cu[1:])
        return compacted, cu, int(kept.sum())


if __name__ == "__main__":  # CPU self-test
    class RS:
        req_id_to_index = {"a": 5, "b": 2, "c": 9}
    caps = np.full(16, 3, dtype=np.int32); caps[5] = 2; caps[9] = 0
    b = StaticDraftBudget(RS(), 1, caps)
    sched = {"b": 4, "a": 4, "c": 4}; drafts = {"b": [-1, -1, -1], "a": [-1, -1, -1], "c": [-1, -1, -1]}
    n = b.get_num_tokens(sched, drafts, 12); assert n == 12 - 1 - 3, n; assert b.pending
    comp, cu, tot = b.compact(np.array([3, 3, 3], dtype=np.int32), np.array([4, 4, 4], dtype=np.int32), ["a", "b", "c"])
    assert comp.tolist() == [3, 4, 1] and cu.tolist() == [0, 3, 7, 8] and tot == 5, (comp, cu, tot)
    assert not b.pending
    # nothing capped -> untouched count, not pending
    assert b.get_num_tokens({"b": 4}, {"b": [-1, -1, -1]}, 4) == 4 and not b.pending
    # a request with no drafts (prefill) is ignored
    assert b.get_num_tokens({"a": 100}, {}, 100) == 100 and not b.pending
    print("static_budget self-test OK")
