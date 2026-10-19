#!/usr/bin/env bash
# Builds the parity probes against the prepared llama.cpp checkout and the
# locally packaged native runtime (`just skippy`).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
llama="$root/.deps/llama.cpp"
lib="${SKIPPY_RUNTIME_LIB:-$(ls -d "$root"/target/debug/native-runtimes/*/lib | head -1)}"
mkdir -p "$here/build"
for probe in mtp_parity_probe batch_invariance_probe margin_probe; do
  "${CXX:-c++}" -std=c++17 -O2 -I"$llama/include" -I"$llama/src" -I"$llama/ggml/include" \
    "$here/$probe.cpp" -L"$lib" -lllama -lggml -lggml-base -Wl,-rpath,"$lib" -o "$here/build/$probe"
done
echo "built probes in $here/build"
