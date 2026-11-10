#!/usr/bin/env bash
# Builds the patched llama.cpp (mesh-llm pin + patch queue, incl. the DSpine
# model_support patch) llama-server for the DSpine vs DFlash A/B. Runs in the
# jy workspace root, where the repo is cloned at mesh-llm/.
set -euo pipefail

cd mesh-llm

echo "=== prepare llama.cpp (pin + patch queue)"
scripts/prepare-llama.sh pinned

BUILD=.deps/llama.cpp/build-dspine-ab

echo "=== configure"
cmake -S .deps/llama.cpp -B "$BUILD" \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

echo "=== build llama-server"
cmake --build "$BUILD" -j "$(nproc)" --target llama-server

"$BUILD/bin/llama-server" --version
echo "=== build done"
