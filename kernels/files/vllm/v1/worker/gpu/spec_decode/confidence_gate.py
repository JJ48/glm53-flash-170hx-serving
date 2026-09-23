# SPDX-License-Identifier: Apache-2.0
"""Per-request speculative budget: a static cap (workload k) and a confidence gate.

Speculative verification cost grows with every drafted token the target has to score
(~2-2.8 ms per extra token on the 4x A100 PP4 GLM-5.3-Flash setup), while acceptance is
sequential: once position j is rejected, positions > j are discarded unverified. Both
levers therefore act on the *number of draft tokens handed to the scheduler*:

* ``spec_tokens`` (request ``vllm_xargs``): a static per-request cap k_req <= k. The
  client knows its workload (prose/code want k=2, structured output k=3 on GLM-5.3-Flash).
* ``spec_conf`` (request ``vllm_xargs``) or ``VLLM_SPEC_CONF_MIN`` (server default): drop
  the draft from the first position whose draft-token probability falls below the
  threshold. A low-confidence draft token is unlikely to be accepted, and everything
  after it would be wasted verification work.

The helpers here are pure tensor/list code so they can be unit-tested on CPU.
"""
from __future__ import annotations

import os

import torch

# Server-wide default for the confidence gate; 0 disables it. Requests override via
# vllm_xargs["spec_conf"].
SPEC_CONF_MIN_ENV = "VLLM_SPEC_CONF_MIN"
# Hard cap on the static per-request k (sanity; the real cap is the server's k).
SPEC_TOKENS_KEY = "spec_tokens"
SPEC_CONF_KEY = "spec_conf"


def server_default_conf_min() -> float:
    try:
        return float(os.environ.get(SPEC_CONF_MIN_ENV, "0") or 0.0)
    except ValueError:
        return 0.0


def parse_spec_tokens(extra_args: dict | None, num_spec_tokens: int) -> int | None:
    """Static per-request cap from SamplingParams.extra_args; None when absent/invalid."""
    if not extra_args:
        return None
    v = extra_args.get(SPEC_TOKENS_KEY)
    if v is None:
        return None
    try:
        k = int(v)
    except (TypeError, ValueError):
        return None
    return max(0, min(k, num_spec_tokens))


def parse_spec_conf(extra_args: dict | None, default: float) -> float:
    """Per-request confidence threshold in [0, 1); falls back to the server default."""
    if extra_args and extra_args.get(SPEC_CONF_KEY) is not None:
        try:
            return min(max(float(extra_args[SPEC_CONF_KEY]), 0.0), 0.999)
        except (TypeError, ValueError):
            pass
    return default


def num_valid_from_confidence(
    conf: torch.Tensor,  # [num_reqs, k] draft-token probabilities (float)
    tau: torch.Tensor | float,  # [num_reqs] or scalar threshold; <= 0 disables
) -> torch.Tensor:  # [num_reqs] int32: leading positions with conf >= tau
    if not torch.is_tensor(tau):
        tau = torch.full((conf.shape[0],), float(tau), dtype=conf.dtype, device=conf.device)
    ok = conf >= tau.to(conf.dtype).unsqueeze(1)
    # cumprod over the position axis: 1 until the first failing position, 0 after.
    return ok.to(torch.int32).cumprod(dim=1).sum(dim=1).to(torch.int32)


def apply_caps(
    num_valid: torch.Tensor,  # [num_reqs] int32 from the confidence gate
    caps: torch.Tensor | None,  # [num_reqs] int32 static per-request caps (k when absent)
) -> torch.Tensor:
    if caps is None:
        return num_valid
    return torch.minimum(num_valid, caps.to(num_valid.dtype))


def num_valid_from_confidence_np(conf, taus):
    """numpy twin of num_valid_from_confidence: conf [n, k], taus [n] (<= 0 disables the row)."""
    import numpy as np

    conf = np.asarray(conf, dtype=np.float32); taus = np.asarray(taus, dtype=np.float32).reshape(-1, 1)
    ok = (conf >= taus) | (taus <= 0.0)
    return np.cumprod(ok.astype(np.int32), axis=1).sum(axis=1).astype(np.int32)


def truncate_draft_lists(
    draft_token_ids: list[list[int]], num_valid: list[int]
) -> list[list[int]]:
    """Shorten each request's draft list to its valid length (host side, after the D2H copy)."""
    out = []
    for ids, n in zip(draft_token_ids, num_valid):
        n = int(n)
        out.append(ids if n >= len(ids) else ids[:n])
    return out


def gate_draft_lists(draft_token_ids, conf_np=None, caps_np=None, taus_np=None):
    """Host-side decision used by DraftTokensHandler.get_draft_tokens: cut each request's
    draft list at the first low-confidence position (conf_np [n, k] vs taus_np [n]; a
    threshold <= 0 leaves the row alone) and at its static cap (caps_np [n])."""
    import numpy as np

    num_valid = None
    if conf_np is not None and taus_np is not None:
        num_valid = num_valid_from_confidence_np(conf_np, taus_np)
    if caps_np is not None:
        caps = np.asarray(caps_np, dtype=np.int32)
        num_valid = caps if num_valid is None else np.minimum(num_valid, caps)
    if num_valid is None:
        return draft_token_ids
    return truncate_draft_lists(draft_token_ids, num_valid.tolist())


def draft_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Probability of the argmax draft token per row, computed in fp32 without
    materialising a softmax over the vocabulary twice."""
    lf = logits.float()
    return torch.exp(lf.max(dim=-1).values - torch.logsumexp(lf, dim=-1))


if __name__ == "__main__":  # CPU self-test
    conf = torch.tensor([[0.9, 0.8, 0.7], [0.9, 0.2, 0.9], [0.1, 0.9, 0.9], [0.5, 0.5, 0.5]])
    nv = num_valid_from_confidence(conf, 0.4)
    assert nv.tolist() == [3, 1, 0, 3], nv.tolist()
    nv2 = num_valid_from_confidence(conf, torch.tensor([0.85, 0.0, 0.0, 0.6]))
    assert nv2.tolist() == [1, 3, 3, 0], nv2.tolist()
    assert num_valid_from_confidence(conf, 0.0).tolist() == [3, 3, 3, 3]  # gate disabled
    capped = apply_caps(nv, torch.tensor([2, 3, 3, 1], dtype=torch.int32))
    assert capped.tolist() == [2, 1, 0, 1], capped.tolist()
    assert truncate_draft_lists([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [2, 0, 5]) == [[1, 2], [], [7, 8, 9]]
    assert parse_spec_tokens({"spec_tokens": 2}, 3) == 2 and parse_spec_tokens({"spec_tokens": 9}, 3) == 3
    assert parse_spec_tokens({"spec_tokens": "x"}, 3) is None and parse_spec_tokens(None, 3) is None and parse_spec_tokens({}, 3) is None
    assert parse_spec_conf({"spec_conf": 0.4}, 0.0) == 0.4 and parse_spec_conf({}, 0.3) == 0.3 and parse_spec_conf({"spec_conf": "bad"}, 0.3) == 0.3
    logits = torch.tensor([[2.0, 1.0, 0.0], [5.0, 0.0, 0.0]])
    p = draft_probs_from_logits(logits)
    assert torch.allclose(p, torch.softmax(logits, -1).max(-1).values), p
    import numpy as np
    nv3 = num_valid_from_confidence_np(conf.numpy(), np.array([0.4, 0.4, 0.4, 0.0]))
    assert nv3.tolist() == [3, 1, 0, 3], nv3.tolist()
    lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]
    g = gate_draft_lists(lists, conf.numpy(), np.array([3, 2, 3, 3]), np.array([0.4, 0.4, 0.0, 0.6]))
    assert g == [[1, 2, 3], [4], [7, 8, 9], []], g          # row1: conf gate 1 then cap 2 -> 1; row2: gate off, cap 3; row3: 0.5 < 0.6 -> 0
    assert gate_draft_lists(lists) == lists and gate_draft_lists(lists, None, np.array([1, 1, 1, 1])) == [[1], [4], [7], [-1]]
    print("confidence_gate self-test OK")
