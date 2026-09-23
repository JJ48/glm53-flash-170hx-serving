#!/usr/bin/env python3
"""Fork patch 0035: the AOT torch.compile cache key must tell a vision boot from a text boot (opt-in, VLLM_AOT_MM_KEY=1).

Why: vLLM stores AOT-compiled graphs under a key built from VllmConfig.compute_hash() plus the forward's qualname and
line (compilation/decorators.py, __call__). ModelConfig.compute_hash ignores multimodal_config and limit_mm_per_prompt,
so a boot with images enabled (--limit-mm-per-prompt image:4) and the text-only production boot (image:0) share one key.
But the first pipeline rank of a vision boot calls the compiled Glm5NextModel.forward with input_ids=None and an
inputs_embeds tensor (the runner merges the image embeddings itself: v1/worker/gpu/model_runner.py, the
requires_raw_input_tokens branch), while the text boot calls it with an input_ids tensor and inputs_embeds=None. The
stored artifact is specialised on that: its entry code reads the dynamic size off the input_ids tensor it was traced
with, guards are not evaluated on load, so the vision boot loads the text artifact and dies in
torch._dynamo.utils.call_size with "'NoneType' object has no attribute 'size'" (boot of 2026-09-22 19:06, Worker_PP0).
Ranks 1-3 always receive input_ids=None and inputs_embeds=None, which is why only rank 0 fell over.

What: with the flag set, a first call that carries an inputs_embeds tensor appends "p0035:inputs_embeds=1,input_ids=N"
to the key factors, so that call gets its own artifact (a fresh compile the first time, a cache hit after). Calls
without inputs_embeds (every text boot, every non-first rank) append nothing: their keys stay byte-identical to
today's, so switching the flag on never invalidates the production text artifacts. One log line when it takes effect,
outside any compiled region.
File: compilation/decorators.py (a helper next to _model_hash_key; four lines in __call__).
Usage: aot_mm_key_patch.py check|dry|apply|revert"""
import os, shutil, sys, py_compile
V = "/usr/local/lib/python3.12/dist-packages/vllm"
FILES = {"decorators": f"{V}/compilation/decorators.py"}
TAG = "patch 0035"
EDITS = {  # file -> list of (old, new, expected_count)
    "decorators": [
        ("def _model_hash_key(fn: Callable[..., Any]) -> str:\n",
         '''# ---- patch 0035: AOT cache key carries the None-pattern of the optional forward inputs (opt-in, VLLM_AOT_MM_KEY=1) ----
import os as _os35

_AOT_MM_KEY35 = _os35.environ.get("VLLM_AOT_MM_KEY", "0") == "1"
_ANNOUNCED35 = False


def _p0035_key_factor(mod: Any, args: Any, kwargs: Any) -> str | None:
    """Extra AOT cache-key factor: whether this call passes inputs_embeds (and input_ids).

    An AOT artifact is specialised on the None-pattern of its inputs (its entry code reads dynamic sizes
    off the tensors it was traced with), and VllmConfig.compute_hash ignores the multimodal limits that
    decide whether the runner passes input_ids (text boot) or inputs_embeds (vision boot, first PP rank).
    Returns None (key unchanged) unless inputs_embeds is a tensor, so text boots keep today's keys.
    """
    global _ANNOUNCED35
    try:
        sig = inspect.signature(mod.__class__.forward)
        bound = sig.bind(mod, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if a.get("inputs_embeds") is None:
            return None
        factor = "p0035:inputs_embeds=1,input_ids=%d" % int(a.get("input_ids") is not None)
    except Exception as e:  # a cache-key helper must never take a boot down
        logger.warning("patch 0035: could not bind forward arguments (%s); AOT key unchanged", e)
        return None
    if not _ANNOUNCED35:
        _ANNOUNCED35 = True
        logger.info("patch 0035 live: AOT cache key of %s carries %s", mod.__class__.__name__, factor)
    return factor


def _model_hash_key(fn: Callable[..., Any]) -> str:
''', 1),
        ("            factors.append(_model_hash_key(self.forward))\n",
         '''            factors.append(_model_hash_key(self.forward))
            if _AOT_MM_KEY35:  # patch 0035: a call with inputs_embeds (vision boot, rank 0) gets its own artifact
                _f35 = _p0035_key_factor(self, args, kwargs)
                if _f35 is not None:
                    factors.append(_f35)
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
            n = s.count(old); print(f"{k}: anchor {old.strip()[:60]!r} found {n} (want {want})"); bad += n != want
    print("dry: OK" if bad == 0 else "dry: ANCHOR MISMATCH"); sys.exit(1 if bad else 0)
if MODE == "revert":
    for k, p in FILES.items():
        b = p + ".bak0035"
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
        if not os.path.exists(p + ".bak0035"): shutil.copy2(p, p + ".bak0035")
        open(p, "w").write(s); py_compile.compile(p, doraise=True); print("applied:", k)
    sys.exit(0)
print("usage: check|dry|apply|revert"); sys.exit(2)
