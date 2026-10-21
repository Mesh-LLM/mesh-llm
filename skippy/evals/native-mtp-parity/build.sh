#!/usr/bin/env bash
# Builds the native MTP probes against the prepared llama.cpp checkout and the
# locally packaged native runtime (`just skippy`).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
llama="$root/.deps/llama.cpp"
if [[ -n "${SKIPPY_RUNTIME_LIB:-}" ]]; then
  lib="$SKIPPY_RUNTIME_LIB"
else
  shopt -s nullglob
  runtime_libs=("$root"/target/debug/native-runtimes/*/lib)
  if ((${#runtime_libs[@]} == 0)); then
    printf '%s\n' "no packaged Skippy runtime found; run just skippy or set SKIPPY_RUNTIME_LIB" >&2
    exit 1
  fi
  lib="${runtime_libs[0]}"
fi
if [[ ! -d "$lib" ]]; then
  printf 'Skippy runtime library directory does not exist: %s\n' "$lib" >&2
  exit 1
fi
out="$root/target/debug/native-mtp-probes"
mkdir -p "$out"
for probe in mtp_parity_probe batch_invariance_probe margin_probe; do
  "${CXX:-c++}" -std=c++17 -O2 -I"$llama/include" -I"$llama/src" -I"$llama/ggml/include" \
    "$here/$probe.cpp" -L"$lib" -lllama -lggml -lggml-base -Wl,-rpath,"$lib" -o "$out/$probe"
done
echo "built probes in $out"
