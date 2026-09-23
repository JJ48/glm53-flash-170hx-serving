#!/usr/bin/env python3
"""Fork patch 0038: fix patch 0014's indexer head-gate precompute, which freezes 9 of 11 DSA layers at ZERO
(opt-in, VLLM_GLM5_WP_FIX=1).

Found 2026-09-23 with a live capture of the indexer's inputs on the production config: patch 0014
(VLLM_GLM5_COMPILE=1, production since 2026-09-15) pre-computes each Indexer's fp32 head-gate `_wp_fp32` at the end of
the language model's load_weights, but only `if getattr(_m, "_wp_fp32", None) is None`. The served class is the
multimodal wrapper, whose loader groups the weight stream by prefix: the only vision tensors sit in shard 9, so the
language model's load_weights runs TWICE (shards 1-8 + the start of 9, then 10-44). The first call's precompute sets
`_wp_fp32` for every indexer from whatever is loaded so far: layers 43 (shard 3) and 3 (shard 6) get their real
weights_proj, the other nine get the zero-initialised rows, and the `is None` guard stops the second call from fixing
them. With a zero head-gate every indexer logit is 0, every pool ties, and top-k returns the first 512 pools: those nine
DSA layers see only the first 2,048 tokens of any prompt (measured: a hard cliff at token 2,048 in every prompt shape).

What: with the flag set, the precompute runs on every load_weights call (the last call sees every weight), and after it
logs how many indexers have an all-zero head-gate (the final line must say 0). Load-time code only, never traced.
File: models/glm5next/nvidia/model.py (patch 0014's block). Changing model.py invalidates the AOT artifacts once, so the
first boot recompiles.
Usage: wp_precompute_fix_patch.py check|dry|apply|revert"""
import os, shutil, sys, py_compile
V = "/usr/local/lib/python3.12/dist-packages/vllm"
FILES = {"model": f"{V}/models/glm5next/nvidia/model.py"}
TAG = "patch 0038"
EDITS = {
    "model": [
        ('_GLM5_COMPILE = _os14.environ.get("VLLM_GLM5_COMPILE", "0") == "1"\n',
         '_GLM5_COMPILE = _os14.environ.get("VLLM_GLM5_COMPILE", "0") == "1"\n'
         '_WP_FIX38 = _os14.environ.get("VLLM_GLM5_WP_FIX", "0") == "1"  # patch 0038: recompute head-gates every load call\n', 1),
        ('                if isinstance(_m, _Indexer14) and getattr(_m, "_wp_fp32", None) is None:\n',
         '                if isinstance(_m, _Indexer14) and (_WP_FIX38 or getattr(_m, "_wp_fp32", None) is None):  # patch 0038\n', 1),
        ('        return _loaded14\n',
         '''        if _GLM5_COMPILE and _WP_FIX38:  # patch 0038: prove the head-gates are populated after this load call
            from .attention import Indexer as _Indexer38

            _ix = [_m for _m in self.modules() if isinstance(_m, _Indexer38)]
            _zero = sum(1 for _m in _ix if getattr(_m, "_wp_fp32", None) is None or float(_m._wp_fp32.abs().sum()) == 0.0)
            logger.info("patch 0038: indexer head-gates after this load_weights call: %d indexers, %d all-zero", len(_ix), _zero)
        return _loaded14
''', 1),
    ],
}
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
if MODE == "check":
    print(" ".join(f"{k}:{'patched' if TAG in open(p).read() else 'unpatched'}" for k, p in FILES.items())); sys.exit(0)
if MODE == "dry":
    bad = 0
    for k, p in FILES.items():
        s = open(p).read()
        if TAG in s: print(f"{k}: already patched"); continue
        for old, new, want in EDITS[k]:
            n = s.count(old); print(f"{k}: anchor {old.strip()[:70]!r} found {n} (want {want})"); bad += n != want
        ok = "logger = init_logger(__name__)" in s; print(f"{k}: needs logger: {'present' if ok else 'MISSING'}"); bad += not ok
    print("dry: OK" if bad == 0 else "dry: MISMATCH"); sys.exit(1 if bad else 0)
if MODE == "revert":
    for k, p in FILES.items():
        b = p + ".bak0038"
        if os.path.exists(b): shutil.copy2(b, p); py_compile.compile(p, doraise=True); print("reverted", k)
        else: print("no backup for", k)
    sys.exit(0)
if MODE == "apply":
    for k, p in FILES.items():
        s = open(p).read()
        if TAG in s: print("already patched:", k); continue
        for old, new, want in EDITS[k]:
            assert s.count(old) == want, f"{k}: expected {want} of {old[:60]!r}, found {s.count(old)}"
        for old, new, _ in EDITS[k]: s = s.replace(old, new)
        if not os.path.exists(p + ".bak0038"): shutil.copy2(p, p + ".bak0038")
        open(p, "w").write(s); py_compile.compile(p, doraise=True); print("applied:", k)
    sys.exit(0)
print("usage: check|dry|apply|revert"); sys.exit(2)
