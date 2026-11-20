#!/usr/bin/env bash
# The MeshLLM host must not link an inference backend. The backend lives in
# the Skippy runtime, loaded at run time, so one host works with any of them.
# This lists the libraries a built host imports and fails if one of them is a
# backend library: CUDA, ROCm, Vulkan, Metal, ggml or llama. On Linux it also
# fails if the host needs a newer glibc than scripts/linux-glibc-floor.txt.
#
#   scripts/check-host-imports.sh <host-binary> [--report <host-imports.json>]
#   scripts/check-host-imports.sh --names      check names read from stdin
#
# --report writes what was checked, with the SHA-256 of the host, only when
# the host passes. A report already at that path is removed first, so a failed
# check never leaves one behind.
#
# It reads imports with otool (macOS), objdump (Linux) or the llvm-objdump
# that ships with Rust's llvm-tools (Windows).
set -euo pipefail

# Anchored at the start of a name or after a path separator, except Metal.
backend='(^|[/\\])(libcuda(\.so|\.dylib|\.dll|$)|libcudart|libcublas|libnccl|libamdhip64|libhip|libhsa-runtime|nvcuda\.dll$|(cudart|cublaslt|cublas)64_[^/\\]+\.dll$|(amdhip64|hipblas|rocblas)\.dll$|libvulkan|vulkan-1\.dll$|(lib)?ggml|(lib)?llama)|metal\.framework'

fail() {
    echo "check-host-imports: $*" >&2
    exit 1
}

case "$(uname -s)" in
    Darwin) format=macho ;;
    Linux) format=elf ;;
    *) format=pe ;;
esac

imports_of() {
    case "$format" in
        macho) otool -L "$1" | tail -n +2 | awk '{print $1}' ;;
        elf) objdump -p "$1" | awk '$1 == "NEEDED" {print $2}' ;;
        pe)
            local rustc_host
            rustc_host="$(rustc -vV | sed -n 's/^host: //p')"
            "$(rustc --print sysroot)/lib/rustlib/$rustc_host/bin/llvm-objdump" -p "$1" |
                tr -d '\r' | sed -n 's/^ *DLL Name: //p'
            ;;
    esac
}

sha256_of() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{print $1}'
}

# The newest glibc symbol version the host needs. Refuses one above the floor.
glibc_needed() {
    local floor needed
    floor="$(grep -v '^#' "$(dirname "$0")/linux-glibc-floor.txt" | tr -d '[:space:]')"
    needed="$(objdump -T "$1" | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -n 1)"
    [[ -n "$floor" && -n "$needed" ]] || fail "could not read the glibc floor of $1"
    [[ "$(printf '%s\n%s\n' "$needed" "$floor" | sort -V | tail -n 1)" == "$floor" ]] ||
        fail "$1 needs glibc $needed but scripts/linux-glibc-floor.txt allows $floor"
    echo "$needed"
}

usage() {
    sed -n '8,9p' "$0" >&2
    exit 2
}

host="" report=""
case "$#" in
    1) host="$1" ;;
    3)
        [[ "$2" == "--report" ]] || usage
        host="$1" report="$3"
        rm -f "$report"
        ;;
    *) usage ;;
esac

sha=""
if [[ "$host" == "--names" ]]; then
    names="$(cat)"
else
    sha="$(sha256_of "$host")"
    names="$(imports_of "$host")"
fi
names="$(LC_ALL=C sort -u <<<"$names")"
# A host that imports nothing means the listing failed, not that it is clean.
[[ -n "$names" ]] || fail "found no imports in $host"

echo "$names"
# grep exits 1 for "no match". Anything else is a failed check, not a clean host.
status=0
rejected="$(grep -Ei "$backend" <<<"$names")" || status=$?
case "$status" in
    1) ;;
    0) fail "the host links a backend library:"$'\n'"$rejected" ;;
    *) fail "grep failed with status $status" ;;
esac

[[ "$host" != "--names" ]] || exit 0
glibc=null
[[ "$format" != elf ]] || glibc="\"$(glibc_needed "$host")\""
[[ -n "$report" ]] || exit 0

# Same fields as the report the removed xtask tool wrote, keys sorted.
[[ "$(sha256_of "$host")" == "$sha" ]] || fail "$host changed while it was being checked"
name="${host##*[/\\]}"
{
    printf '{\n  "binary": "%s",\n  "binary_sha256": "%s",\n  "format": "%s",\n' "$name" "$sha" "$format"
    printf '  "glibc_floor": %s,\n  "imports": [\n' "$glibc"
    sed -e 's/[\\"]/\\&/g' -e 's/.*/    "&"/' -e '$!s/$/,/' <<<"$names"
    printf '  ],\n  "policy": "mesh-llm-dynamic-host-v2",\n  "rejected_imports": []\n}\n'
} >"$report.tmp"
mv "$report.tmp" "$report"
