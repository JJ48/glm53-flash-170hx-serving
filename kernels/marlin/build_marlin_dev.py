#!/usr/bin/env python3
"""Build the upstream Marlin MoE kernel (8e92248f79) as a standalone stable-ABI extension `_marlin_dev` for one arch
(MARLIN_ARCH=80 default, 86 for RTX 30-series; MARLIN_NVCC_EXTRA adds nvcc flags, e.g. -allow-unsupported-compiler).
Run inside the container from this directory after `python3 csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py 8.0`.
Logs to build.log; the resulting .so path is printed at the end."""
import glob, os, sys, time, torch
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__)); csrc = os.path.join(here, "csrc")
moe = os.path.join(csrc, "libtorch_stable", "moe", "marlin_moe_wna16")
ARCH = os.environ.get("MARLIN_ARCH", "80"); EXTRA = os.environ.get("MARLIN_NVCC_EXTRA", "").split()
sources = [os.path.join(here, "bindings_dev.cpp"), os.path.join(moe, "ops.cu")] + sorted(glob.glob(os.path.join(moe, "sm*_kernel_*.cu")))
print(f"sources: {len(sources)} ({len(sources)-2} instantiation files), arch sm_{ARCH}, extra nvcc flags {EXTRA}", flush=True)
t0 = time.time()
mod = load(name="_marlin_dev", sources=sources, extra_include_paths=[csrc, moe, os.path.join(csrc, "libtorch_stable")],
           extra_cflags=["-O3", "-std=c++17", "-DTORCH_STABLE_ONLY", "-DUSE_CUDA"],
           extra_cuda_cflags=["-O3", "-std=c++17", "-DTORCH_STABLE_ONLY", "-DUSE_CUDA", f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}", "--expt-relaxed-constexpr", "-static-global-template-stub=false"] + EXTRA,
           build_directory=os.path.join(here, os.environ.get("MARLIN_BUILD_DIR", "build")), verbose=True, is_python_module=False, is_standalone=False)
print(f"built in {time.time()-t0:.0f}s; op present: {hasattr(torch.ops, '_marlin_dev') and hasattr(torch.ops._marlin_dev, 'moe_wna16_marlin_gemm')}", flush=True)
