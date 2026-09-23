#!/usr/bin/env python3
"""Fork patch 0005: per-position draft confidence for the DFlash2 drafter + a floor for the confidence gate.

The fork's confidence gate (patch 0002, vllm_xargs["spec_conf"] / VLLM_SPEC_CONF_MIN) cuts a request's draft list at the first
position whose draft probability is below the threshold. Autoregressive drafters (MTP/EAGLE) record that probability per draft
step; the DFlash2 block drafter never did, so its confidence buffer stayed at 1.0 and the gate was inert for it.

(a) dflash2/speculator.py: after the selector walk, the probability of the chosen candidate under the realized selector scores
    (softmax over the top-k candidates at each block position) is written into draft_token_confidence_probs[:, :steps].
(b) confidence_gate.py: VLLM_SPEC_CONF_FLOOR (default 2) keeps at least that many drafts when the gate is active for a row
    (a 1-token step has no FULL graph on a k>=2 server; k=1 measured slower than k=2 everywhere).
Usage: dflash2_conf_patch.py check|apply|revert
"""
import os, shutil, sys

D = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode"
SPEC = f"{D}/dflash2/speculator.py"
GATE = f"{D}/confidence_gate.py"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"

OLD_A = "        self._sample_path(candidate_ids, scores, num_reqs)\n"
NEW_A = """        self._sample_path(candidate_ids, scores, num_reqs)
        import os as _os
        if getattr(self, "compute_draft_confidence", False):  # DFLASH2CONF: per-position draft confidence for the gate
            # Probability of the chosen candidate under the realized selector scores at each block position
            # (softmax over the top-k candidates; the walk kernel stored the score row it actually used).
            _rs = self._selector_scores[:num_reqs]
            _probs = torch.softmax(_rs, dim=-1)
            _chosen = candidate_ids == self.draft_tokens[:num_reqs].unsqueeze(-1)
            self.draft_token_confidence_probs[:num_reqs, : self.num_speculative_steps] = (
                (_probs * _chosen).sum(dim=-1)
            )
            if (
                _os.environ.get("VLLM_DFLASH2_CONF_DEBUG") == "1"
                and not torch.cuda.is_current_stream_capturing()
                and getattr(self, "_d2conf_dbg_n", 0) < 12
            ):
                self._d2conf_dbg_n = getattr(self, "_d2conf_dbg_n", 0) + 1
                _top2 = torch.topk(_rs[0], 2, dim=-1).values
                print(
                    f"[D2CONF n={self._d2conf_dbg_n} reqs={num_reqs} conf0={[round(x, 3) for x in self.draft_token_confidence_probs[0, : self.num_speculative_steps].tolist()]} "
                    f"chosen_hits={_chosen[0].sum(dim=-1).tolist()} score_margin0={[round(x, 2) for x in (_top2[:, 0] - _top2[:, 1]).tolist()]} "
                    f"compute={getattr(self, 'compute_draft_confidence', None)}]",
                    flush=True,
                )
"""
OLD_B = """    ok = (conf >= taus) | (taus <= 0.0)
    return np.cumprod(ok.astype(np.int32), axis=1).sum(axis=1).astype(np.int32)
"""
NEW_B = """    ok = (conf >= taus) | (taus <= 0.0)
    nv = np.cumprod(ok.astype(np.int32), axis=1).sum(axis=1).astype(np.int32)
    floor = conf_floor()  # DFLASH2CONF: an active gate never cuts below the floor (1-token steps have no FULL graph)
    if floor > 0 and conf.shape[1] > 0:
        nv = np.where(taus.reshape(-1) > 0.0, np.maximum(nv, min(floor, conf.shape[1])), nv).astype(np.int32)
    return nv
"""
OLD_C = "def parse_spec_tokens(extra_args: dict | None, num_spec_tokens: int) -> int | None:\n"
NEW_C = """def conf_floor() -> int:
    \"\"\"Minimum drafts kept by an active confidence gate (VLLM_SPEC_CONF_FLOOR, default 2; 0 disables).\"\"\"
    try:
        return max(0, int(os.environ.get("VLLM_SPEC_CONF_FLOOR", "2") or 0))
    except ValueError:
        return 2


def parse_spec_tokens(extra_args: dict | None, num_spec_tokens: int) -> int | None:
"""
OLD_D = """    num_valid = None
    if conf_np is not None and taus_np is not None:
        num_valid = num_valid_from_confidence_np(conf_np, taus_np)
"""
NEW_D = """    num_valid = None
    if conf_np is not None and taus_np is not None:
        num_valid = num_valid_from_confidence_np(conf_np, taus_np)
    if os.environ.get("VLLM_DFLASH2_CONF_DEBUG") == "1":  # DFLASH2CONF debug: what the gate sees (first 10 calls)
        global _D2GATE_N
        try:
            _D2GATE_N += 1
        except NameError:
            _D2GATE_N = 1
        if _D2GATE_N <= 10:
            _c = None if conf_np is None else [round(float(x), 3) for x in np.asarray(conf_np)[0][:8]]
            print(f"[D2GATE n={_D2GATE_N} conf0={_c} taus0={None if taus_np is None else float(np.asarray(taus_np)[0])} "
                  f"caps0={None if caps_np is None else int(np.asarray(caps_np)[0])} num_valid0={None if num_valid is None else int(num_valid[0])} "
                  f"len0={len(draft_token_ids[0]) if draft_token_ids else None}]", flush=True)
"""
edits = [(SPEC, [(OLD_A, NEW_A)]), (GATE, [(OLD_B, NEW_B), (OLD_C, NEW_C), (OLD_D, NEW_D)])]

if MODE == "check":
    ok = True
    for path, pairs in edits:
        s = open(path).read()
        for o, _ in pairs:
            found = s.count(o) == 1
            ok &= found or "DFLASH2CONF" in s
            print(os.path.basename(path), "anchor", "found" if found else ("patched" if "DFLASH2CONF" in s else "MISSING"))
    sys.exit(0 if ok else 1)
if MODE == "apply":
    for path, pairs in edits:
        s = open(path).read()
        if "DFLASH2CONF" in s:
            print(os.path.basename(path), "already patched"); continue
        for o, _ in pairs:
            assert s.count(o) == 1, f"anchor not unique/missing in {path}"
        bak = path + ".d2conf.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        for o, n in pairs:
            s = s.replace(o, n)
        open(path, "w").write(s); print(os.path.basename(path), "patched; backup", bak)
elif MODE == "revert":
    for path, _ in edits:
        bak = path + ".d2conf.bak"
        if os.path.exists(bak):
            shutil.copy2(bak, path); print(os.path.basename(path), "reverted")
        else:
            print(os.path.basename(path), "no backup")
