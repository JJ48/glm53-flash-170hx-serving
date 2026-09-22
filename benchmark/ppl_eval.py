#!/usr/bin/env python3
"""Quality gate that does not depend on sampling: mean negative log-likelihood (perplexity) of fixed public-domain text under a running
vLLM server, via /v1/completions with prompt_logprobs (max_tokens=1, echo). Same chunks for every checkpoint -> directly comparable.

  python3 ppl_eval.py --base-url http://127.0.0.1:8000 --tag orig    --out ./ppl-results
  python3 ppl_eval.py --base-url http://127.0.0.1:8000 --tag kda-w8  --out ./ppl-results
  python3 ppl_eval.py compare ./ppl-results/orig.json ./ppl-results/kda-w8.json
"""
import argparse, json, math, os, re, sys, time, urllib.request
TEXTS = {  # Project Gutenberg plain-text mirrors (public domain), sliced deterministically below
    "pride_prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "origin_species": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
    "federalist": "https://www.gutenberg.org/cache/epub/1404/pg1404.txt",
    "frankenstein": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
}
def fetch(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True); p = os.path.join(cache_dir, re.sub(r"\W+", "_", url)[-80:] + ".txt")
    if not os.path.exists(p):
        open(p, "wb").write(urllib.request.urlopen(url, timeout=120).read())
    t = open(p, encoding="utf-8", errors="replace").read()
    a = t.find("*** START"); b = t.rfind("*** END")
    if a > 0: t = t[t.find("\n", a) + 1:]
    if b > 0: t = t[:b]
    return re.sub(r"\r\n", "\n", t)
def chunks(text, n_chunks, chars):
    step = max(chars, (len(text) - chars) // max(n_chunks, 1))
    return [text[i:i + chars] for i in range(0, min(len(text) - chars, step * n_chunks), step)][:n_chunks]
def nll_of(base, model, prompt):
    body = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "echo": False, "prompt_logprobs": 1}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    pl = r["choices"][0].get("prompt_logprobs")
    if pl is None: raise SystemExit("server did not return prompt_logprobs (vLLM: enable via request; check --max-logprobs)")
    lps = []
    for entry in pl:
        if not entry: continue                       # first prompt token carries no logprob
        vals = list(entry.values())
        if len(vals) == 1: v = vals[0]                 # actual token == top-1
        else:                                          # top-1 plus the actual token (rank != 1)
            nontop = [x for x in vals if x.get("rank") not in (1, None)]
            v = nontop[0] if nontop else vals[-1]
        lps.append(v["logprob"])
    return -sum(lps) / len(lps), len(lps)
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
        print(f"{'text':18s} {'nll_a':>8s} {'nll_b':>8s} {'ppl_a':>8s} {'ppl_b':>8s} {'dPPL%':>7s}")
        for k in a["per_text"]:
            na, nb = a["per_text"][k]["nll"], b["per_text"][k]["nll"]
            print(f"{k:18s} {na:8.4f} {nb:8.4f} {math.exp(na):8.3f} {math.exp(nb):8.3f} {100*(math.exp(nb)/math.exp(na)-1):7.2f}")
        na, nb = a["mean_nll"], b["mean_nll"]
        print(f"{'MEAN':18s} {na:8.4f} {nb:8.4f} {math.exp(na):8.3f} {math.exp(nb):8.3f} {100*(math.exp(nb)/math.exp(na)-1):7.2f}   ({a['tag']} vs {b['tag']}, {a['tokens']} tokens)")
        return
    p = argparse.ArgumentParser(); p.add_argument("--base-url", default="http://127.0.0.1:8000"); p.add_argument("--model", default="GLM-5.3-Flash")
    p.add_argument("--tag", required=True); p.add_argument("--out", default="./ppl-results"); p.add_argument("--chunks", type=int, default=6)
    p.add_argument("--chars", type=int, default=6000); p.add_argument("--cache", default="./ppl-texts")
    p.add_argument("--texts-dir", default=None, help="use local text files from this dir (unseen text) instead of the Gutenberg set")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = {"tag": a.tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "per_text": {}, "chunks": a.chunks, "chars": a.chars}
    tot_nll = 0.0; tot_tok = 0
    if a.texts_dir:
        sources = {os.path.splitext(f)[0]: os.path.join(a.texts_dir, f) for f in sorted(os.listdir(a.texts_dir))}
    else:
        sources = TEXTS
    for name, src in sources.items():
        text = open(src, encoding="utf-8", errors="replace").read() if a.texts_dir else fetch(src, a.cache); nll_sum = 0.0; ntok = 0
        for i, ch in enumerate(chunks(text, a.chunks, a.chars)):
            nll, n = nll_of(a.base_url, a.model, ch); nll_sum += nll * n; ntok += n
            print(f"{name} chunk {i}: nll {nll:.4f} over {n} tokens", flush=True)
        res["per_text"][name] = {"nll": nll_sum / ntok, "tokens": ntok}; tot_nll += nll_sum; tot_tok += ntok
    res["mean_nll"] = tot_nll / tot_tok; res["tokens"] = tot_tok; res["ppl"] = math.exp(res["mean_nll"])
    json.dump(res, open(os.path.join(a.out, a.tag + ".json"), "w"), indent=1)
    print(f"{a.tag}: mean NLL {res['mean_nll']:.4f}, PPL {res['ppl']:.3f} over {tot_tok} tokens")
if __name__ == "__main__":
    main()
