#!/usr/bin/env bash
# The MeshLLM host must not link an inference backend. The backend lives in
# the Skippy runtime, loaded at run time, so one host works with any of them.
# This lists the libraries a built host imports and fails if one of them is a
# backend library: CUDA, ROCm, Vulkan, Metal, ggml or llama.
#
#   scripts/check-host-imports.sh <host-binary>
#   scripts/check-host-imports.sh --names      check names read from stdin
#
# It reads imports with otool (macOS), objdump (Linux) or the llvm-objdump
# that ships with Rust's llvm-tools (Windows).
set -euo pipefail

# Anchored at the start of a name or after a path separator, except Metal.
backend='(^|[/\\])(libcuda(\.so|\.dylib|\.dll|$)|libcudart|libcublas|libnccl|libamdhip64|libhip|libhsa-runtime|nvcuda\.dll$|(cudart|cublaslt|cublas)64_[^/\\]+\.dll$|(amdhip64|hipblas|rocblas)\.dll$|libvulkan|vulkan-1\.dll$|(lib)?ggml|(lib)?llama)|metal\.framework'

imports_of() {
    case "$(uname -s)" in
        Darwin) otool -L "$1" | tail -n +2 | awk '{print $1}' ;;
        Linux) objdump -p "$1" | awk '$1 == "NEEDED" {print $2}' ;;
        *)
            local rustc_host
            rustc_host="$(rustc -vV | sed -n 's/^host: //p')"
            "$(rustc --print sysroot)/lib/rustlib/$rustc_host/bin/llvm-objdump" -p "$1" |
                tr -d '\r' | sed -n 's/^ *DLL Name: //p'
            ;;
    esac
}

(($# == 1)) || { sed -n '2,9p' "$0" >&2; exit 2; }
if [[ "$1" == "--names" ]]; then
    names="$(cat)"
else
    names="$(imports_of "$1")"
fi
# A host that imports nothing means the listing failed, not that it is clean.
[[ -n "$names" ]] || { echo "check-host-imports: found no imports in $1" >&2; exit 1; }

echo "$names"
if rejected="$(grep -Ei "$backend" <<<"$names")"; then
    echo "check-host-imports: the host links a backend library:" >&2
    echo "$rejected" >&2
    exit 1
fi
