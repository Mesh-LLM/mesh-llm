#!/usr/bin/env bash
set -euo pipefail
if [[ ( $# != 1 && $# != 3 ) || ( $1 != metal && $1 != cuda ) ]]; then
    echo 'usage: build.sh metal|cuda [BASELINE_WORKSPACE CURRENT_WORKSPACE]' >&2
    exit 2
fi
backend=$1
arms=(D R)
if [[ $# == 3 ]]; then
    baseline=$2
    current=$3
    test "$(git -C "$baseline" rev-parse HEAD)" = 3947de0cbc73fe5cbcaae548b297f03a401532d7
    test "$(git -C "$current" rev-parse HEAD)" = a9b37b71bb9ebcff51e321e7fcedba323f7d9458
    arms=(B C D R)
fi
scripts/prepare-llama.sh pinned
test "$(git -C .deps/llama.cpp rev-parse 'HEAD^{tree}')" = e4cc2b65b42834fe2ab257a2517afd969ba7f9c5
if [[ $# == 3 ]]; then
    LLAMA_WORKDIR="$PWD/.deps/simd-B-source" LLAMA_UPSTREAM_URL="$PWD/.deps/llama.cpp" "$baseline/scripts/prepare-llama.sh" pinned
    LLAMA_WORKDIR="$PWD/.deps/simd-C-source" LLAMA_UPSTREAM_URL="$PWD/.deps/llama.cpp" "$current/scripts/prepare-llama.sh" pinned
fi
just sampler-simd-controller "$PWD/.deps/llama.cpp"
configuration=(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
if [[ $backend == cuda ]]; then configuration+=(-DCMAKE_CUDA_ARCHITECTURES=121); fi
for arm in "${arms[@]}"; do
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
        "-DLLAMA_SAMPLER_HIGHWAY_DISABLED=$disabled" "${configuration[@]}"
    cmake --build ".deps/simd-$arm" --parallel 4 --target simd-thermal-probe
    ".deps/simd-$arm/simd-thermal-probe"
    cmake --build ".deps/simd-$arm" --parallel 4 --target grammar-sampling
    if [[ $arm != B ]]; then
        cmake --build ".deps/simd-$arm" --parallel 4 --target grammar-verification
    fi
    if [[ $arm == D || $arm == R ]]; then
        cmake --build ".deps/simd-$arm" --parallel 4 --target test-sampler-highway
        ctest --test-dir ".deps/simd-$arm/native" --output-on-failure -R '^test-sampler-highway$'
    fi
done
