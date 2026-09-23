#!/usr/bin/env python3
"""Fork patch 0015: pack each pipeline-parallel hop into ONE NCCL op (opt-in, VLLM_PP_PACKED_HOP=1; default = upstream).

Measured on the box (kineto traces, shared clock, k=7): the three PP hops cost 0.87/0.86/0.73 ms of GPU idle per
step = 2.46 ms (5.8 % of the 42.25 ms period), 4-5x the 64 KiB micro-benchmark. Cause: GroupCoordinator.isend_tensor_dict
sends the metadata by a blocking gloo send_object and then issues one NCCL isend PER TENSOR — the hop carries 6
tensors (hidden_states + 5 DFlash aux-tap slots, all allocated every hop), each a separate host-bounced NCCL op.
This patch, when every tensor is CUDA, non-empty and of one dtype and no all-gather slicing applies (TP=1):
  sender: appends ("__packed_hop__", True) to the metadata, cats the tensors into one flat buffer, one isend;
  receiver: allocates one flat buffer of the summed numel, one irecv, returns views into it (data lands after the
  handle wait exactly as before). Anything else falls through to the upstream per-tensor path.
Usage: pp_packed_patch.py check|apply|revert"""
import os, shutil, sys, py_compile
P = "/usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py"
BAK = P + ".bak0015"; TAG = "patch 0015"
EDITS = [
    ("def _split_tensor_dict(\n",
     'import os as _os15  # patch 0015\n_PP_PACKED = _os15.environ.get("VLLM_PP_PACKED_HOP", "0") == "1"\n\n\ndef _split_tensor_dict(\n', 1),
    ("        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)\n        self.send_object(metadata_list, dst=dst)\n\n        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]\n",
     "        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)\n"
     "        if (  # patch 0015: one flat buffer, one NCCL op per hop\n"
     "            _PP_PACKED\n"
     "            and all_gather_size == 1\n"
     "            and tensor_list\n"
     "            and all(\n"
     "                t.is_cuda and t.numel() > 0 and t.dtype == tensor_list[0].dtype\n"
     "                for t in tensor_list\n"
     "            )\n"
     "        ):\n"
     "            self.send_object(list(metadata_list) + [(\"__packed_hop__\", True)], dst=dst)\n"
     "            flat = torch.cat([t.reshape(-1) for t in tensor_list])\n"
     "            handle = torch.distributed.isend(flat, dst=self.ranks[dst], group=group)\n"
     "            flat.record_stream(torch.cuda.current_stream(flat.device))\n"
     "            return [handle]\n"
     "        self.send_object(metadata_list, dst=dst)\n\n"
     "        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]\n", 1),
    ("        recv_metadata_list = self.recv_object(src=src)\n        tensor_dict: dict[str, Any] = {}\n        handles: list[Handle] = []\n",
     "        recv_metadata_list = self.recv_object(src=src)\n"
     "        if (  # patch 0015: packed hop -> one irecv into a flat buffer, views out\n"
     "            _PP_PACKED\n"
     "            and recv_metadata_list\n"
     "            and recv_metadata_list[-1] == (\"__packed_hop__\", True)\n"
     "        ):\n"
     "            metas = [\n"
     "                (k, v) for k, v in recv_metadata_list[:-1] if isinstance(v, TensorMetadata)\n"
     "            ]\n"
     "            total = sum(int(torch.Size(v.size).numel()) for _, v in metas)\n"
     "            flat = torch.empty(total, dtype=metas[0][1].dtype, device=metas[0][1].device)\n"
     "            handle = torch.distributed.irecv(flat, src=self.ranks[src], group=group)\n"
     "            out: dict[str, Any] = {}\n"
     "            off = 0\n"
     "            for k, v in recv_metadata_list[:-1]:\n"
     "                if isinstance(v, TensorMetadata):\n"
     "                    n = int(torch.Size(v.size).numel())\n"
     "                    out[k] = flat[off : off + n].view(v.size)\n"
     "                    off += n\n"
     "                else:\n"
     "                    out[k] = v\n"
     "            return out, [handle], []\n"
     "        tensor_dict: dict[str, Any] = {}\n"
     "        handles: list[Handle] = []\n", 1),
]
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
s = open(P).read()
if MODE == "check": print("patched" if TAG in s else "unpatched"); sys.exit(0)
if MODE == "revert":
    if os.path.exists(BAK): shutil.copy2(BAK, P); print("reverted")
    sys.exit(0)
if MODE == "apply":
    if TAG in s: print("already patched"); sys.exit(0)
    for old, new, want in EDITS: assert s.count(old) == want, f"expected {want} of {old[:50]!r}, found {s.count(old)}"
    if not os.path.exists(BAK): shutil.copy2(P, BAK)
    for old, new, _ in EDITS: s = s.replace(old, new)
    open(P, "w").write(s); py_compile.compile(P, doraise=True); print("applied"); sys.exit(0)
