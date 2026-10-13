#!/bin/bash
set -euo pipefail
args=("$@")
printf '%s\n' "$*" >> "${CMAKE_STUB_LOG:?}"
target=""
generator="${CMAKE_GENERATOR:-Unix Makefiles}"
for ((index=0; index<${#args[@]}; index++)); do
  case "${args[index]}" in
    -B|--build) target="${args[index+1]}" ;;
    -G) generator="${args[index+1]}" ;;
  esac
done
mkdir -p "$target"
if [[ "${args[0]}" == --build ]]; then
  for name in libllama libllama-common libmtmd; do
    touch "$target/$name.dylib" "$target/$name.so"
  done
else
  printf 'CMAKE_GENERATOR:INTERNAL=%s\n' "$generator" > "$target/CMakeCache.txt"
fi
