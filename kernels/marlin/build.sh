#!/bin/bash
# Build the standalone MoE Marlin kernel that patch 0018 loads (VLLM_MARLIN_DEV_SO), with the extra 6- and 8-stage
# decode instantiations and the MARLIN_DEV_STAGES knob (marlin_stages.patch), from upstream vLLM 8e92248f79.
# Run inside the club-170hx container (it has nvcc and the matching torch). CPU-only build; takes a while.
#   bash kernels/marlin/build.sh [WORKDIR]      -> prints the path of the built _marlin_dev .so
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); W=${1:-/tmp/marlin-build}
SHA=8e92248f79814bc12a195f5d2529cc245768017c
rm -rf "$W"; git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git "$W"
cd "$W"; git sparse-checkout init --cone; git sparse-checkout set csrc; git checkout -q "$SHA"
git apply "$HERE/marlin_stages.patch"
cp "$HERE/bindings_dev.cpp" "$HERE/build_marlin_dev.py" .
python3 csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py 8.0
mkdir -p build
MAX_JOBS=${MAX_JOBS:-16} python3 build_marlin_dev.py
find "$W/build" -name '_marlin_dev*.so' | head -1
