#!/usr/bin/env python3
"""Fork patch 0025: the one-launch Triton top-k/top-p for ANY number of logit rows (upstream: only from 8 rows up).

Production samples at the checkpoint's default (temperature 1.0, top_p 0.95), so every decode step runs top-p over the
154,880-token vocabulary for 1 + k logit rows. `apply_top_k_top_p` takes the Triton pivot kernel only when there are at
least 8 rows; with the rolling draft length most prose / chat / reasoning steps have 2-7 rows and fall to the PyTorch
path: sort, softmax, cumsum, compare, index write, masked_fill, scatter = 22-28 kernel launches. Measured on the RTX 3080
with LM-like logits (`dev/fork-patches/topp_bench.py`): sort path 0.46-0.82 ms for 2-8 rows, Triton kernel 0.19-0.22 ms
(0.23-0.35 ms for a flat distribution), identical kept-token sets in every case. The English set at the production default
showed steps 1.1-1.8 ms longer than at temperature 0 exactly in the low-k categories. (An earlier version of this text added
"15-17 us per launch on the box": that figure was profiler overhead; the 170HX needs 5-6 us per tiny eager kernel.)

  VLLM_TOPP_TRITON_MIN_ROWS=N   (default 8 = upstream; use 2 on the box)
  170HX, no profiler (topp_bench.py, 2026-09-20): sort path 0.85-1.03 ms for 2-8 rows (0.72-1.24 ms flat), Triton kernel
  0.26-0.32 ms (0.40-0.55 ms flat), identical kept sets; at ONE row the sort path wins (0.32 vs 0.42 ms), hence 2.

The Triton implementation is upstream's own and already serves every batch of 8 rows and more; this only moves the
threshold. Usage: topp_triton_rows_patch.py check|apply|revert"""
import os, shutil, sys, py_compile
F = os.environ.get("VLLM_TOPK_TOPP_SAMPLER_PY", "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/ops/topk_topp_sampler.py")
SUFFIX = ".bak0025"
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
TAG = "patch 0025"

REPL = [
 ('''    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)
''',
  '''    if HAS_TRITON and logits.shape[0] >= _P0025_MIN_ROWS:  # patch 0025: upstream threshold is 8
        return apply_top_k_top_p_triton(logits, k, p)
'''),
]
HELPERS = '''
# ---- patch 0025: row threshold of the Triton top-k/top-p kernel (VLLM_TOPP_TRITON_MIN_ROWS, default 8 = upstream) ----
import os as _p0025_os
try:
    _P0025_MIN_ROWS = max(1, int(_p0025_os.environ.get("VLLM_TOPP_TRITON_MIN_ROWS", "8")))
except ValueError:
    _P0025_MIN_ROWS = 8
# ---- end patch 0025 helpers ----


'''
ANCHOR = "def apply_top_k_top_p(\n"


def patch_text(src):
    for old, _ in REPL:
        assert src.count(old) == 1, f"expected exactly one of:\n{old}"
    assert src.count(ANCHOR) == 1, "apply_top_k_top_p not found"
    for old, new in REPL:
        src = src.replace(old, new)
    return src.replace(ANCHOR, HELPERS.lstrip("\n") + ANCHOR, 1)


if __name__ == "__main__":
    s = open(F).read()
    patched = TAG in s
    if MODE == "check":
        print("patched" if patched else "unpatched"); sys.exit(0)
    if MODE == "revert":
        if not patched: print("not patched; nothing reverted"); sys.exit(1)
        if not os.path.exists(F + SUFFIX): print("no backup; nothing reverted"); sys.exit(1)
        if s != patch_text(open(F + SUFFIX).read()):
            print("REFUSED: the installed file is not 'backup + patch 0025'. Revert whatever sits on top first."); sys.exit(1)
        shutil.copy2(F + SUFFIX, F); print("reverted from *" + SUFFIX); sys.exit(0)
    if MODE == "apply":
        if patched: print("already patched"); sys.exit(0)
        new = patch_text(s)
        shutil.copy2(F, F + SUFFIX)
        open(F, "w").write(new); py_compile.compile(F, doraise=True)
        print("applied; backup *" + SUFFIX); sys.exit(0)
    print("usage: check|apply|revert"); sys.exit(2)
