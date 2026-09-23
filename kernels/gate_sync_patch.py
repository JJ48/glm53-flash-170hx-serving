#!/usr/bin/env python3
"""Fork patch 0010: remove the per-step host stall introduced by patch 0006 (the lagged confidence gate).

Measured 2026-09-15 with per-phase laps inside the last PP rank's sample_tokens: of ~35 ms of host time per step, 33 ms
sat in the gate block, whose `torch.as_tensor(<numpy>, device=cuda)` is a PAGEABLE host-to-device copy. That copy is
synchronous and stream-ordered, so the host blocks until everything queued on the main stream has drained. The block runs
whenever the speculator merely HAS a draft_token_confidence_probs tensor, i.e. on every DFlash2 step regardless of whether
the gate is enabled, so every DFlash2 configuration paid it. Rank 3's host loop (sample_tokens -> execute_model ->
sample_tokens) is the step's critical path, so the stall set the step period (41 ms) rather than the GPU chain (33 ms).

Fix: keep the per-request thresholds as a device-resident tensor, written once when a request is added (async fill_, no
memcpy) and indexed on-device per step. Semantics are identical. Usage: gate_sync_patch.py check|apply|revert"""
import os, shutil, sys
P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
TAG = "patch 0010"
EDITS = [
  # 1. allocation: device twin of spec_conf_np
  ("""        self.spec_conf_np = np.full(
            self.max_num_reqs, server_default_conf_min(), dtype=np.float32
        )
""",
   """        self.spec_conf_np = np.full(
            self.max_num_reqs, server_default_conf_min(), dtype=np.float32
        )
        # patch 0010: device-resident twin, indexed on-device per step (a per-step torch.as_tensor from numpy stalled the host)
        self.spec_conf_gpu = torch.full(
            (self.max_num_reqs,), float(server_default_conf_min()), dtype=torch.float32, device=self.device
        )
"""),
  # 2. write site: keep the twin in sync with an async fill (no host<->device memcpy)
  ("""            self.spec_conf_np[req_index] = parse_spec_conf(
                extra_args, server_default_conf_min()
            )
""",
   """            self.spec_conf_np[req_index] = parse_spec_conf(
                extra_args, server_default_conf_min()
            )
            self.spec_conf_gpu[req_index : req_index + 1].fill_(float(self.spec_conf_np[req_index]))  # patch 0010
"""),
  # 3. the per-step stall
  ("""                _taus = torch.as_tensor(self.spec_conf_np[input_batch.idx_mapping_np], device=self.device)
""",
   """                _taus = self.spec_conf_gpu[input_batch.idx_mapping[:_n]]  # patch 0010: on-device gather, no pageable H2D
"""),
]
s = open(P).read()
if MODE == "check":
    print("patched" if TAG in s else "unpatched"); [print(f"  anchor {i}: {s.count(a)}") for i, (a, _) in enumerate(EDITS)]; sys.exit(0)
if MODE == "apply":
    if TAG in s: print("already patched"); sys.exit(0)
    for i, (a, _) in enumerate(EDITS): assert s.count(a) == 1, f"anchor {i} count {s.count(a)}"
    if not os.path.exists(P + ".gatesync.bak"): shutil.copy2(P, P + ".gatesync.bak")
    for a, b in EDITS: s = s.replace(a, b)
    open(P, "w").write(s); print("applied")
elif MODE == "revert":
    b = P + ".gatesync.bak"
    if os.path.exists(b): shutil.copy2(b, P); os.remove(b); print("reverted")
    else: print("no backup")
