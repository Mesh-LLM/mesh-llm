#!/usr/bin/env bash
# Detect the CUDA toolkit version selected for a build.
#
# This is the shared source for local CUDA architecture selection. It reports
# the selected compiler's version (CUDACXX, CMAKE_CUDA_COMPILER, NVCC, then
# PATH nvcc) or toolkit-owned version metadata. It intentionally does not use
# driver capability reports: nvidia-smi cannot prove which toolkit built the
# runtime. A missing or unreadable toolkit is an error, not a CUDA 12 default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/cuda-toolkit.sh"

if ! version="$(cuda_toolkit_manifest_version)"; then
    exit 1
fi

printf '%s\n' "$version"
