#!/usr/bin/env python3
"""Fork patch 0036: profile the largest image at the processor's own token cap, not the largest square under it
(opt-in, VLLM_GLM5_MM_BUDGET=1).

Why: the encoder cache budget is sized from get_max_image_tokens(), which the inherited Glm4vProcessingInfo derives from
get_image_size_with_most_features(): it hands smart_resize a 9999999x9999999 square and takes what fits the pixel budget,
i.e. the largest SQUARE under max_image_tokens = 8000 -> 89x89 merged patches = 7,921 tokens. A real page is not square:
smart_resize fits it anywhere up to 8,000 tokens (a 3000x4000 phone photo of a drawing lands at 103x77 = 7,931), so any
image between 7,922 and 8,000 tokens exceeds the profiled budget and the request is refused (2026-09-22 vision window:
HTTP 400 on every 3000x4000 image, 1400x700 fine). The processor's cap is the true per-item maximum; the profiler
under-reads it by up to 79 tokens purely through the square search.

What: with the flag set, get_image_size_with_most_features() returns an aligned canvas of exactly max_image_tokens merged
patches (8000 -> 80x100 -> 2240x2800 px; general form: the divisor pair of max_image_tokens closest to square). The dummy
image then carries max_image_tokens, so the encoder budget and the profile run both cover everything the processor can
emit; the KV pool gives up the activation memory of 79 extra tokens (about 1 % of one image). One log line when live.
File: models/glm5next/nvidia/multimodal.py (one method on Glm5NextProcessingInfo).
Usage: mm_budget_cap_patch.py check|dry|apply|revert"""
import os, shutil, sys, py_compile
V = "/usr/local/lib/python3.12/dist-packages/vllm"
FILES = {"mm": f"{V}/models/glm5next/nvidia/multimodal.py"}
TAG = "patch 0036"
EDITS = {
    "mm": [
        ("class Glm5NextProcessingInfo(Glm4vProcessingInfo):\n",
         '''# ---- patch 0036: size the profiling image at the processor's token cap (opt-in, VLLM_GLM5_MM_BUDGET=1) ----
import os as _os36

from vllm.logger import init_logger as _init_logger36

_MM_BUDGET36 = _os36.environ.get("VLLM_GLM5_MM_BUDGET", "0") == "1"
_logger36 = _init_logger36(__name__)
_announced36 = [False]


class Glm5NextProcessingInfo(Glm4vProcessingInfo):
''', 1),
        ("    def _processor_pixel_budget(self, proc) -> tuple[int, int]:\n",
         '''    def get_image_size_with_most_features(self) -> ImageSize:  # patch 0036
        """The inherited search takes the largest SQUARE under the pixel budget (89x89 = 7,921 tokens for
        max_image_tokens = 8000), but smart_resize fits a real page anywhere up to max_image_tokens (a 3000x4000
        photo lands at 103x77 = 7,931), so the encoder budget came out short of what the processor can emit.
        Return an aligned canvas of exactly max_image_tokens merged patches instead."""
        if not _MM_BUDGET36:
            return super().get_image_size_with_most_features()
        proc = self.get_hf_processor().image_processor
        tokens = int(proc.max_image_tokens)
        factor = int(proc.patch_size) * int(proc.merge_size) * int(getattr(proc, "patch_expand_factor", 1) or 1)
        a = max(d for d in range(1, int(tokens**0.5) + 1) if tokens % d == 0)
        b = tokens // a
        size = ImageSize(width=b * factor, height=a * factor)
        if not _announced36[0]:
            _announced36[0] = True
            inherited = self._get_vision_info(
                image_width=9999999, image_height=9999999, num_frames=1,
                max_image_pixels=self._get_image_max_pixels(),
            )[1]
            _logger36.info(
                "patch 0036 live: profiling image sized to %d tokens (%dx%d merged patches, %dx%d px) = "
                "the processor's max_image_tokens; the inherited square search gave %d",
                tokens, a, b, size.height, size.width, inherited,
            )
        return size

    def _processor_pixel_budget(self, proc) -> tuple[int, int]:
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
        print(f"{k}: needs 'ImageSize': {'present' if 'ImageSize' in s else 'MISSING'}"); bad += "ImageSize" not in s
    print("dry: OK" if bad == 0 else "dry: ANCHOR/IMPORT MISMATCH"); sys.exit(1 if bad else 0)
if MODE == "revert":
    for k, p in FILES.items():
        b = p + ".bak0036"
        if os.path.exists(b): shutil.copy2(b, p); py_compile.compile(p, doraise=True); print("reverted", k)
        else: print("no backup for", k)
    sys.exit(0)
if MODE == "apply":
    for k, p in FILES.items():
        s = open(p).read()
        if TAG in s: print("already patched:", k); continue
        for old, new, want in EDITS[k]:
            assert s.count(old) == want, f"{k}: expected {want} of {old[:60]!r}, found {s.count(old)}"
        assert "ImageSize" in s, f"{k}: 'ImageSize' missing from the file; the patch relies on it"
        for old, new, _ in EDITS[k]: s = s.replace(old, new)
        if not os.path.exists(p + ".bak0036"): shutil.copy2(p, p + ".bak0036")
        open(p, "w").write(s); py_compile.compile(p, doraise=True); print("applied:", k)
    sys.exit(0)
print("usage: check|dry|apply|revert"); sys.exit(2)
