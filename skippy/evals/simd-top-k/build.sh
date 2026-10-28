#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 || ( $1 != metal && $1 != cuda ) ]]; then
    echo 'usage: build.sh metal|cuda BASELINE_WORKSPACE CURRENT_WORKSPACE' >&2
    exit 2
fi
backend=$1
baseline=$2
current=$3
test "$(git -C "$baseline" rev-parse HEAD)" = 3947de0cbc73fe5cbcaae548b297f03a401532d7
test "$(git -C "$current" rev-parse HEAD)" = a9b37b71bb9ebcff51e321e7fcedba323f7d9458
scripts/prepare-llama.sh pinned
test "$(git -C .deps/llama.cpp rev-parse 'HEAD^{tree}')" = c81b95e77324501872f9aac8acc2488d9365df63
LLAMA_WORKDIR="$PWD/.deps/simd-B-source" LLAMA_UPSTREAM_URL="$PWD/.deps/llama.cpp" "$baseline/scripts/prepare-llama.sh" pinned
LLAMA_WORKDIR="$PWD/.deps/simd-C-source" LLAMA_UPSTREAM_URL="$PWD/.deps/llama.cpp" "$current/scripts/prepare-llama.sh" pinned
architecture=()
if [[ $backend == cuda ]]; then architecture=(-DCMAKE_CUDA_ARCHITECTURES=121); fi
for arm in B C D R; do
    native="$PWD/.deps/llama.cpp"
    upstream=ON
    highway=OFF
    disabled=OFF
    if [[ $arm == B ]]; then native="$PWD/.deps/simd-B-source"; upstream=OFF; fi
    if [[ $arm == C ]]; then native="$PWD/.deps/simd-C-source"; fi
    if [[ $arm == D || $arm == R ]]; then highway=ON; fi
    if [[ $arm == R ]]; then disabled=ON; fi
    cmake -S skippy/evals/simd-top-k -B ".deps/simd-$arm" -G Ninja \
        "-DLLAMA_ROOT=$native" "-DSIMD_BACKEND=$backend" \
        "-DSKIPPY_EVAL_COMMON_SAMPLER=$upstream" "-DLLAMA_SAMPLER_HIGHWAY=$highway" \
        "-DLLAMA_SAMPLER_HIGHWAY_DISABLED=$disabled" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${architecture[@]}"
    cmake --build ".deps/simd-$arm" --parallel 4 --target simd-thermal-probe
    ".deps/simd-$arm/simd-thermal-probe"
    cmake --build ".deps/simd-$arm" --parallel 4 --target grammar-sampling grammar-verification
    if [[ $arm == D || $arm == R ]]; then
        cmake --build ".deps/simd-$arm" --parallel 4 --target test-sampler-highway
        ctest --test-dir ".deps/simd-$arm/native" --output-on-failure -R '^test-sampler-highway$'
    fi
done
