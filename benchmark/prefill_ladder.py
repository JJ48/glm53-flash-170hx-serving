#!/usr/bin/env python3
"""Prefill / TTFT ladder against a running server (inside the container): exact-length prompts, streaming, medians.

For every requested prompt length: build ONE fixture of exactly that many tokens (binary search on the server's /tokenize,
cached per length in /tmp), then send `--reps` streaming requests, each with a fresh nonce prefix so nothing can be reused,
and record TTFT (first token of the stream), the prompt tokens the server reports, and prefill tok/s = prompt_tokens / TTFT.
The first request of each length is a warm-up and is NOT counted (the first long prompt after a boot pays the indexer's
Triton autotune, seconds at 16k+).

Sampling follows production: no temperature or top_p in the request unless --temperature is given.

  --lengths 2048 4096 8192 16384 32768   prompt lengths (tokens, exact)
  --reps 3                               counted repetitions per length
  --burst N:L                            additionally: N prompts of length L at once (default off; e.g. 8:23500)
  --label NAME                           written into ./prefill_<NAME>.json

Usage: prefill_ladder.py --label NAME [--port 8000] [--lengths ...] [--reps 3] [--burst 8:23500]"""
import argparse, concurrent.futures as cf, json, os, statistics, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True); ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--model", default="GLM-5.3-Flash")
ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--burst", default="")
ap.add_argument("--max-tokens", type=int, default=16); ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--cache-dir", default="/tmp/prefill_fixtures")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
SENT = ("The CNC lathe program begins with a safety block, sets the work offset, selects the roughing tool and turns the "
        "outer diameter in successive passes while the operator monitors spindle load and coolant flow, recording every "
        "measurement in the traveler before the part moves to the inspection bench. ")


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def ntok(text):
    return json.load(post("/tokenize", {"model": a.model, "prompt": text}, 600))["count"]


def fixture(target):
    """Text of exactly `target` tokens once a 40-character nonce prefix is prepended (cached on disk per length)."""
    os.makedirs(a.cache_dir, exist_ok=True)
    path = os.path.join(a.cache_dir, f"{target}.txt")
    if os.path.exists(path):
        t = open(path).read()
        if ntok(NONCE_EXAMPLE + t) == target: return t
    words = (SENT * max(1, target // 40)).split(" ")
    lo, hi = 1, len(words)
    while lo < hi:                                   # smallest word count whose token length reaches the target
        mid = (lo + hi) // 2
        if ntok(NONCE_EXAMPLE + " ".join(words[:mid])) < target: lo = mid + 1
        else: hi = mid
    text = " ".join(words[:lo])
    for _ in range(80):                              # then trim or extend by single words
        c = ntok(NONCE_EXAMPLE + text)
        if c == target: break
        text = text.rsplit(" ", 1)[0] if c > target else text + " " + words[len(text.split(" ")) % len(words)]
    open(path, "w").write(text)
    return text


NONCE_EXAMPLE = "[" + "0" * 32 + "] "


def stream_once(prompt):
    body = {"model": a.model, "prompt": prompt, "max_tokens": a.max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if a.temperature is not None: body["temperature"] = a.temperature
    t0 = time.perf_counter(); ttft = None; usage = None
    with post("/v1/completions", body) as r:
        for line in r:
            if not line.startswith(b"data:"): continue
            s = line[5:].strip()
            if s == b"[DONE]": break
            d = json.loads(s)
            if ttft is None and d.get("choices") and d["choices"][0].get("text"):
                ttft = time.perf_counter() - t0
            if d.get("usage"): usage = d["usage"]
    return ttft, time.perf_counter() - t0, (usage or {}).get("prompt_tokens")


rows = []
print(f"{'len':>7}{'rep':>5}{'prompt_tok':>11}{'TTFT s':>9}{'prefill tok/s':>14}   {a.label}")
for n in a.lengths:
    text = fixture(n)
    for rep in range(-1, a.reps):                    # rep -1 = warm-up, not counted
        prompt = f"[{uuid.uuid4().hex}] " + text
        try:
            ttft, total, pt = stream_once(prompt)
            tps = pt / ttft if (ttft and pt) else None
            print(f"{n:>7}{rep:>5}{pt if pt else 0:>11}{ttft if ttft else 0:>9.3f}{tps if tps else 0:>14.0f}"
                  + ("   (warm-up, not counted)" if rep < 0 else ""), flush=True)
            if rep >= 0: rows.append({"label": a.label, "len": n, "rep": rep, "prompt_tokens": pt, "ttft_s": ttft,
                                      "prefill_tps": tps, "total_s": total})
        except Exception as e:  # noqa: BLE001
            print(f"{n:>7}{rep:>5}  ERROR {type(e).__name__}: {e}"[:200], flush=True)
            if rep >= 0: rows.append({"label": a.label, "len": n, "rep": rep, "error": f"{type(e).__name__}: {e}"[:200]})

if a.burst:
    cnt, blen = (int(x) for x in a.burst.split(":"))
    text = fixture(blen)
    prompts = [f"[{uuid.uuid4().hex}] " + text for _ in range(cnt)]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(cnt) as ex: out = list(ex.map(stream_once, prompts))
    wall = time.perf_counter() - t0
    tt = sorted(x[0] for x in out if x[0])
    print(f"BURST {cnt} x {blen}: wall {wall:.2f} s | TTFT min {tt[0]:.2f} median {statistics.median(tt):.2f} max {tt[-1]:.2f} s", flush=True)
    rows.append({"label": a.label, "burst": a.burst, "wall_s": wall, "ttft_s": tt})

print(f"\n=== {a.label}: median of {a.reps} reps per length")
print(f"{'len':>7}{'prompt_tok':>11}{'TTFT s':>9}{'prefill tok/s':>14}{'spread s':>10}")
for n in a.lengths:
    ok = [r for r in rows if r.get("len") == n and r.get("ttft_s")]
    if not ok: print(f"{n:>7}   no successful rep"); continue
    ts = sorted(r["ttft_s"] for r in ok)
    print(f"{n:>7}{ok[0]['prompt_tokens']:>11}{statistics.median(ts):>9.3f}"
          f"{statistics.median([r['prefill_tps'] for r in ok]):>14.0f}{ts[-1] - ts[0]:>10.3f}")
out = f"./prefill_{a.label}.json"
json.dump(rows, open(out, "w"), indent=1); print("saved", out)
