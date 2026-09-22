#!/usr/bin/env python3
"""Concurrency sweep against an OpenAI-compatible vLLM endpoint.

Fires a fixed prompt set at each concurrency level and reports BOTH throughput views (see
../METHODOLOGY.md): per-request tok/s (unweighted mean of each request's rate) and system-total
tok/s (token-weighted, total tokens / wall). Speculative acceptance is read from the server's
Prometheus /metrics (vLLM spec-decode counters) as a delta over each level.

Use a balanced request count (a multiple of the prompt-set size) so every prompt is weighted
equally and levels are comparable. Use REAL natural-language prompts — synthetic/random prompts
collapse drafter acceptance and give a no-speculation worst case.

Prompts JSON format: {"name": {"messages": [...], "max_tokens": N, "chat_template_kwargs": {...}}, ...}
(the last two keys optional). A small example set is in ./prompts_en.json.

Usage:
  en_conc.py --base-url http://127.0.0.1:8000 --model MODEL \
             --prompts prompts_en.json --levels 1,2,4,8,16 --reqs 40,40,40,40,40
"""
import argparse, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor


def post_chat(base_url, body, timeout=3600):
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def scrape_spec(base_url):
    """Return (accepted_total, draft_total) from vLLM's spec-decode counters, labels summed."""
    acc = draft = 0.0
    try:
        txt = urllib.request.urlopen(base_url + "/metrics", timeout=10).read().decode()
    except Exception:
        return None
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^(vllm:[A-Za-z0-9_:]+)(\{[^}]*\})?\s+([-+0-9.eE]+)", line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(3))
        if name == "vllm:spec_decode_num_accepted_tokens_total":
            acc += val
        elif name == "vllm:spec_decode_num_draft_tokens_total":
            draft += val
    return acc, draft


def accept_len(before, after):
    if not before or not after:
        return None
    d_acc, d_draft = after[0] - before[0], after[1] - before[1]
    return round(1 + d_acc / d_draft, 3) if d_draft > 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="GLM-5.3-Flash")
    p.add_argument("--prompts", default="prompts_en.json")
    p.add_argument("--levels", default="1,2,4,8,16")
    p.add_argument("--reqs", default="40,40,40,40,40", help="requests per level; use a multiple of the prompt count")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    work = list(json.load(open(a.prompts)).items())
    base, model = a.base_url.rstrip("/"), a.model

    def one(idx):
        _, pr = work[idx % len(work)]
        body = {"model": model, "messages": pr["messages"], "max_tokens": pr.get("max_tokens", 1024)}
        if pr.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = pr["chat_template_kwargs"]
        t0 = time.perf_counter()
        try:
            r = post_chat(base, body); dt = time.perf_counter() - t0
            return {"ok": True, "tok": r["usage"]["completion_tokens"], "dt": dt}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "dt": time.perf_counter() - t0, "err": f"{type(e).__name__}: {e}"[:200]}

    levels = [int(x) for x in a.levels.split(",")]
    reqs = [int(x) for x in a.reqs.split(",")]
    rows = []
    print("%3s %4s %13s %14s %11s %11s %8s %5s" %
          ("C", "N", "per_req_tok/s", "sys_total_t/s", "accept_len", "mean_e2e_s", "wall_s", "fail"))
    for C, N in zip(levels, reqs):
        mb = scrape_spec(base); t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=C) as ex:
            res = list(ex.map(one, range(N)))
        wall = time.perf_counter() - t0; ma = scrape_spec(base)
        ok = [r for r in res if r["ok"]]; fail = len(res) - len(ok)
        toks = sum(r["tok"] for r in ok)
        per = [r["tok"] / r["dt"] for r in ok if r["dt"] > 0]
        agg = toks / wall if wall else 0
        pr = sum(per) / len(per) if per else 0
        e2e = sum(r["dt"] for r in ok) / len(ok) if ok else 0
        al = accept_len(mb, ma)
        row = {"concurrency": C, "n": N, "per_request_tok_s": round(pr, 1),
               "system_total_tok_s": round(agg, 1), "accept_len": al,
               "mean_latency_s": round(e2e, 2), "wall_s": round(wall, 1), "failed": fail}
        rows.append(row)
        print("%3d %4d %13.1f %14.1f %11s %11.2f %8.1f %5d" %
              (C, N, pr, agg, str(al), e2e, wall, fail), flush=True)
        if fail:
            print("   first error:", next((r["err"] for r in res if not r["ok"]), ""), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
