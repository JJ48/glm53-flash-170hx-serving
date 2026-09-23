#!/usr/bin/env python3
"""Fork patch 0017 (opt-in VLLM_PP_HOP_V2=1; requires patch 0015 and VLLM_PP_PACKED_HOP=1): the last two byte/op cuts
on the pipeline critical path measured with a pipeline-bubble trace.
  (b) unproduced DFlash aux-tap slots cross the hop as EMPTY tensors instead of full zero tensors (model.py); the
      packed hop (0015) skips zero-numel entries in the flat buffer and the receiver materialises empties. The
      receiving model replaces a slot when it produces the tap, so downstream sees the same tensors as before.
  (c) the tail's three broadcasts (sampled tokens, combined counts, draft tokens) become ONE broadcast issued after
      the drafter (pp_utils.py): non-last ranks cannot start the next step without the draft, so nothing waits longer.
Usage: pp_hop_v2_patch.py check|apply|revert   (revert 0017 before reverting 0014/0015: layered backups .bak0017)"""
import os, shutil, sys, py_compile
V = "/usr/local/lib/python3.12/dist-packages/vllm"
F = {"model": f"{V}/models/glm5next/nvidia/model.py", "ps": f"{V}/distributed/parallel_state.py", "ppu": f"{V}/v1/worker/gpu/pp_utils.py"}
TAG = "patch 0017"
E = {
 "model": [
  ('_GLM5_COMPILE = _os14.environ.get("VLLM_GLM5_COMPILE", "0") == "1"\n',
   '_GLM5_COMPILE = _os14.environ.get("VLLM_GLM5_COMPILE", "0") == "1"\n_HOP_V2 = _os14.environ.get("VLLM_PP_HOP_V2", "0") == "1"  # patch 0017\n', 1),
  ('''                if aux is None:
                    aux = hidden_states.new_zeros(
                        (hidden_states.shape[0], hidden_states.shape[-1])
                    )
''',
   '''                if aux is None:
                    aux = hidden_states.new_zeros(  # patch 0017: unproduced slot -> empty, not zeros
                        (0 if _HOP_V2 else hidden_states.shape[0], hidden_states.shape[-1])
                    )
''', 1)],
 "ps": [
  ('''            and tensor_list
            and all(
                t.is_cuda and t.numel() > 0 and t.dtype == tensor_list[0].dtype
                for t in tensor_list
            )
        ):
            self.send_object(list(metadata_list) + [("__packed_hop__", True)], dst=dst)
            flat = torch.cat([t.reshape(-1) for t in tensor_list])
''',
   '''            and tensor_list
            and all(t.is_cuda and t.dtype == tensor_list[0].dtype for t in tensor_list)
            and any(t.numel() > 0 for t in tensor_list)
        ):  # patch 0017: zero-numel entries (empty aux slots) ride in the metadata only
            self.send_object(list(metadata_list) + [("__packed_hop__", True)], dst=dst)
            flat = torch.cat([t.reshape(-1) for t in tensor_list if t.numel() > 0])
''', 1),
  ('''                if isinstance(v, TensorMetadata):
                    n = int(torch.Size(v.size).numel())
                    out[k] = flat[off : off + n].view(v.size)
                    off += n
''',
   '''                if isinstance(v, TensorMetadata):
                    n = int(torch.Size(v.size).numel())
                    if n == 0:  # patch 0017
                        out[k] = torch.empty(v.size, dtype=v.dtype, device=v.device)
                        continue
                    out[k] = flat[off : off + n].view(v.size)
                    off += n
''', 1)],
 "ppu": [
  ('class PPHandler:\n', 'import os as _os17  # patch 0017\n_HOP_V2 = _os17.environ.get("VLLM_PP_HOP_V2", "0") == "1"\n\n\nclass PPHandler:\n', 1),
  ("        with torch.cuda.stream(self.broadcast_stream):\n            self.broadcast_stream.wait_stream(self.main_stream)\n            sampled_tokens = torch.empty(\n                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device\n            )\n            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)\n            torch.distributed.broadcast(\n                sampled_tokens, src=self.last_rank, group=self.broadcast_group\n            )\n            torch.distributed.broadcast(\n                combined, src=self.last_rank, group=self.broadcast_group\n            )\n            # Spec decode: 3rd broadcast on this group — the proposed draft tokens\n            # relayed by the last rank (matches broadcast_draft's order). NCCL\n            # matches by op order on the communicator; the deferred event covers it.\n            draft_tokens = None\n            if self.max_sample_len > 1:\n                draft_tokens = torch.empty(\n                    num_reqs,\n                    self.max_sample_len - 1,\n                    dtype=torch.int64,\n                    device=self.device,\n                )\n                torch.distributed.broadcast(\n                    draft_tokens, src=self.last_rank, group=self.broadcast_group\n                )\n", "        with torch.cuda.stream(self.broadcast_stream):\n            self.broadcast_stream.wait_stream(self.main_stream)\n            if _HOP_V2 and self.max_sample_len > 1:  # patch 0017: one broadcast for sampled+combined+draft\n                L = self.max_sample_len\n                flat = torch.empty(num_reqs * (2 * L + 1), dtype=torch.int64, device=self.device)\n                torch.distributed.broadcast(flat, src=self.last_rank, group=self.broadcast_group)\n                a = num_reqs * L\n                sampled_tokens = flat[:a].view(num_reqs, L)\n                combined = flat[a : a + 2 * num_reqs].view(2, num_reqs).to(torch.int32)\n                draft_tokens = flat[a + 2 * num_reqs :].view(num_reqs, L - 1)\n                flat.record_stream(self.main_stream)\n            else:\n                sampled_tokens = torch.empty(\n                    num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device\n                )\n                combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)\n                torch.distributed.broadcast(\n                    sampled_tokens, src=self.last_rank, group=self.broadcast_group\n                )\n                torch.distributed.broadcast(\n                    combined, src=self.last_rank, group=self.broadcast_group\n                )\n                # Spec decode: 3rd broadcast on this group — the proposed draft tokens\n                # relayed by the last rank (matches broadcast_draft's order). NCCL\n                # matches by op order on the communicator; the deferred event covers it.\n                draft_tokens = None\n                if self.max_sample_len > 1:\n                    draft_tokens = torch.empty(\n                        num_reqs,\n                        self.max_sample_len - 1,\n                        dtype=torch.int64,\n                        device=self.device,\n                    )\n                    torch.distributed.broadcast(\n                        draft_tokens, src=self.last_rank, group=self.broadcast_group\n                    )\n", 1),
  ('        with torch.cuda.stream(self.broadcast_stream):\n            self.broadcast_stream.wait_stream(self.main_stream)\n            torch.distributed.broadcast(\n                sampled_token_ids.contiguous(),\n', '        if _HOP_V2 and self.max_sample_len > 1:  # patch 0017: defer to broadcast_draft (one op)\n            self._v2_pending = (sampled_token_ids.contiguous(), num_sampled, num_rejected)\n            return\n        with torch.cuda.stream(self.broadcast_stream):\n            self.broadcast_stream.wait_stream(self.main_stream)\n            torch.distributed.broadcast(\n                sampled_token_ids.contiguous(),\n', 1),
  ("        with torch.cuda.stream(self.broadcast_stream):\n            # wait_stream so the side-stream broadcast sees propose()'s output.\n            self.broadcast_stream.wait_stream(self.main_stream)\n            torch.distributed.broadcast(\n                draft_tokens, src=self.last_rank, group=self.broadcast_group\n            )\n            draft_tokens.record_stream(self.broadcast_stream)\n", "        with torch.cuda.stream(self.broadcast_stream):\n            # wait_stream so the side-stream broadcast sees propose()'s output.\n            self.broadcast_stream.wait_stream(self.main_stream)\n            if _HOP_V2:  # patch 0017: sampled + combined + draft in one broadcast\n                s, ns, nr = self._v2_pending\n                self._v2_pending = None\n                flat = torch.cat(\n                    [s.reshape(-1), torch.stack((ns, nr), dim=0).reshape(-1).to(torch.int64), draft_tokens.reshape(-1)]\n                )\n                torch.distributed.broadcast(flat, src=self.last_rank, group=self.broadcast_group)\n                for t in (s, ns, nr, draft_tokens, flat):\n                    t.record_stream(self.broadcast_stream)\n                return\n            torch.distributed.broadcast(\n                draft_tokens, src=self.last_rank, group=self.broadcast_group\n            )\n            draft_tokens.record_stream(self.broadcast_stream)\n", 1)],
}
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
if MODE == "check": print(" ".join(f"{k}:{'patched' if TAG in open(p).read() else 'unpatched'}" for k, p in F.items())); sys.exit(0)
if MODE == "revert":
    for k, p in F.items():
        if os.path.exists(p + ".bak0017"): shutil.copy2(p + ".bak0017", p); print("reverted", k)
    sys.exit(0)
if MODE == "apply":
    for k, p in F.items():
        s = open(p).read()
        if TAG in s: print("already patched:", k); continue
        for old, new, want in E[k]: assert s.count(old) == want, f"{k}: expected {want} of {old[:70]!r}, found {s.count(old)}"
        if not os.path.exists(p + ".bak0017"): shutil.copy2(p, p + ".bak0017")
        for old, new, _ in E[k]: s = s.replace(old, new)
        open(p, "w").write(s); py_compile.compile(p, doraise=True); print("applied:", k)
    sys.exit(0)
