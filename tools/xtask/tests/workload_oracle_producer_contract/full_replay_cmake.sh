#!/bin/bash
set -euo pipefail
printf '%s\0' cmake "$@" '' >> "${FULL_REPLAY_TRACE:?}"
args=("$@")
build=""
for ((i=0;i<${#args[@]};i++)); do
  case "${args[i]}" in -B|--build) build="${args[i+1]}" ;; esac
done
[[ -n "$build" && "$build" == "$FULL_REPLAY_BUILD" ]]
mkdir -p "$build"
if [[ "${args[0]}" == --build ]]; then
  for path in src/libllama.a common/libllama-common.a common/libllama-common-base.a ggml/src/libggml.a ggml/src/libggml-base.a ggml/src/libggml-cpu.a tools/mtmd/libmtmd.a vendor/hash/libvendor-hash.a; do
    mkdir -p "$build/$(dirname "$path")"
    touch "$build/$path"
  done
  mkdir -p "$build/bin"
  cp "$FULL_REPLAY_GENERATOR" "$build/bin/skippy-model-fixture-generator"
fi
