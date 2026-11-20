#!/usr/bin/env bash
# scripts/check-host-imports.sh refuses backend libraries and nothing else.
set -euo pipefail
check="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/check-host-imports.sh"
fail=0

expect() { # <accepted|refused> <import name>
    local got=accepted
    printf '%s\n' "$2" | "$check" --names >/dev/null 2>&1 || got=refused
    [[ "$got" == "$1" ]] || { echo "FAIL: '$2' was $got, want $1" >&2; fail=1; }
}

for name in /usr/lib/libcuda.so.1 libcuda 'C:\Windows\NVCUDA.DLL' cudart64_12.dll \
    cublasLt64_12.dll cublas64_13.dll libcudart.so.12 libcublas.so.12 libnccl.so.2 \
    libamdhip64.so.6 libhipblas.so libhsa-runtime64.so.1 amdhip64.dll rocblas.dll \
    libvulkan.so.1 vulkan-1.dll @rpath/libggml-base.dylib ggml.dll libllama.so llama.dll \
    /System/Library/Frameworks/Metal.framework/Versions/A/Metal; do
    expect refused "$name"
done

for name in libcudax.so cudart64_.dll 'cudart64_1/2.dll' xllama.so /usr/lib/libSystem.B.dylib \
    libc.so.6 libssl.so.3 KERNEL32.dll \
    /System/Library/Frameworks/Security.framework/Versions/A/Security; do
    expect accepted "$name"
done

# One backend library among ordinary ones is enough.
printf 'libc.so.6\nlibcudart.so.12\n' | "$check" --names >/dev/null 2>&1 &&
    { echo "FAIL: a list holding libcudart.so.12 was accepted" >&2; fail=1; }
# An empty listing is a failed listing.
printf '' | "$check" --names >/dev/null 2>&1 &&
    { echo "FAIL: an empty import list was accepted" >&2; fail=1; }

# A grep that fails must not read as "no backend library found".
stub="$(mktemp -d)"
trap 'rm -rf "$stub"' EXIT
printf '#!/bin/sh\nexit 2\n' >"$stub/grep"
chmod +x "$stub/grep"
out="$(printf 'libc.so.6\n' | PATH="$stub:$PATH" "$check" --names 2>&1)" &&
    { echo "FAIL: a failing grep was accepted" >&2; fail=1; }
[[ "$out" == *"grep failed with status 2"* ]] ||
    { echo "FAIL: a failing grep was not reported: $out" >&2; fail=1; }

((fail == 0)) && echo "check-host-imports-test: ok"
exit "$fail"
