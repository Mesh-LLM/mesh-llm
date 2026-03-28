#!/usr/bin/env bash
# build-linux-amd.sh — build llama.cpp (ROCm/HIP) + mesh-llm on Linux
#
# Usage: scripts/build-linux-amd.sh [amdgpu_targets]
#   amdgpu_targets  Semicolon-separated AMDGPU targets, e.g.
#                   "gfx90a;gfx942;gfx1100". If omitted, a broad default is used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="$REPO_ROOT/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
MESH_DIR="$REPO_ROOT/mesh-llm"
UI_DIR="$MESH_DIR/ui"

AMDGPU_TARGETS="${1:-gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [[ ! -d "$ROCM_PATH" ]]; then
    echo "Error: ROCm not found at $ROCM_PATH" >&2
    exit 1
fi

export ROCM_PATH
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"

if ! command -v hipconfig >/dev/null 2>&1; then
    echo "Error: hipconfig not found. Ensure ROCm is installed and PATH includes $ROCM_PATH/bin." >&2
    exit 1
fi

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "Cloning michaelneale/llama.cpp (rebase-upstream-master)..."
    git clone -b rebase-upstream-master \
        https://github.com/michaelneale/llama.cpp.git "$LLAMA_DIR"
else
    cd "$LLAMA_DIR"
    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != "rebase-upstream-master" ]]; then
        echo "⚠️  llama.cpp is on branch '$CURRENT_BRANCH', switching to rebase-upstream-master..."
        git checkout rebase-upstream-master
    fi
    echo "Pulling latest rebase-upstream-master from origin..."
    git pull --ff-only origin rebase-upstream-master
    cd "$REPO_ROOT"
fi

echo "Using ROCm from $ROCM_PATH"
echo "Building for AMDGPU targets: $AMDGPU_TARGETS"

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" \
    -DGGML_HIP=ON \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_METAL=OFF \
    -DGGML_RPC=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DAMDGPU_TARGETS="$AMDGPU_TARGETS"

cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"
echo "llama.cpp ROCm build complete: $BUILD_DIR/bin/"

if [[ -d "$MESH_DIR" ]]; then
    if [[ -d "$UI_DIR" ]]; then
        echo "Building mesh-llm UI..."
        (cd "$UI_DIR" && npm ci && npm run build)
    fi
    echo "Building mesh-llm..."
    (cd "$REPO_ROOT" && cargo build --release --locked -p mesh-llm)
    echo "Mesh binary: target/release/mesh-llm"
fi
