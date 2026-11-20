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

# Each tool's output is captured before it is parsed, and a tool that exits
# non-zero fails the check even if it printed something plausible first.
imports_of() {
    local raw rustc_host
    case "$format" in
        macho)
            raw="$(otool -L "$1")" || fail "otool failed on $1"
            names="$(tail -n +2 <<<"$raw" | awk '{print $1}')" || fail "could not parse otool output"
            ;;
        elf)
            raw="$(objdump -p "$1")" || fail "objdump -p failed on $1"
            names="$(awk '$1 == "NEEDED" {print $2}' <<<"$raw")" || fail "could not parse objdump output"
            ;;
        pe)
            rustc_host="$(rustc -vV | sed -n 's/^host: //p')" || fail "rustc -vV failed"
            raw="$("$(rustc --print sysroot)/lib/rustlib/$rustc_host/bin/llvm-objdump" -p "$1")" ||
                fail "llvm-objdump failed on $1"
            names="$(tr -d '\r' <<<"$raw" | sed -n 's/^ *DLL Name: //p')" || fail "could not parse llvm-objdump output"
            ;;
    esac
}

sha256_of() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{print $1}'
}

# Sets glibc to the newest glibc symbol version the host needs. Refuses one
# above the floor.
glibc_needed() {
    local floor raw needed newest
    # Exactly one version line, apart from comments and blank lines.
    floor="$(grep -Ev '^[[:blank:]]*(#|$)' "$(dirname "$0")/linux-glibc-floor.txt")" ||
        fail "could not read linux-glibc-floor.txt"
    [[ "$floor" =~ ^[[:blank:]]*([0-9]+(\.[0-9]+)*)[[:blank:]]*$ ]] ||
        fail "linux-glibc-floor.txt does not hold exactly one version"
    floor="${BASH_REMATCH[1]}"
    raw="$(objdump -T "$1")" || fail "objdump -T failed on $1"
    needed="$(grep -o 'GLIBC_[0-9][0-9.]*' <<<"$raw" | sed 's/GLIBC_//' | sort -V | tail -n 1)" ||
        fail "found no glibc symbol version in $1"
    [[ -n "$needed" ]] || fail "found no glibc symbol version in $1"
    newest="$(printf '%s\n%s\n' "$needed" "$floor" | sort -V | tail -n 1)" || fail "could not compare glibc versions"
    [[ "$newest" == "$floor" ]] || fail "$1 needs glibc $needed but scripts/linux-glibc-floor.txt allows $floor"
    glibc="\"$needed\""
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
    sha="$(sha256_of "$host")" || fail "could not hash $host"
    [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || fail "could not hash $host"
    imports_of "$host"
fi
names="$(LC_ALL=C sort -u <<<"$names")" || fail "could not sort the import names"
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
[[ "$format" != elf ]] || glibc_needed "$host"
[[ -n "$report" ]] || exit 0

# Same fields as the report the removed xtask tool wrote, keys sorted.
again="$(sha256_of "$host")" || fail "could not hash $host"
[[ "$again" == "$sha" ]] || fail "$host changed while it was being checked"
name="${host##*[/\\]}"
{
    printf '{\n  "binary": "%s",\n  "binary_sha256": "%s",\n  "format": "%s",\n' "$name" "$sha" "$format"
    printf '  "glibc_floor": %s,\n  "imports": [\n' "$glibc"
    sed -e 's/[\\"]/\\&/g' -e 's/.*/    "&"/' -e '$!s/$/,/' <<<"$names"
    printf '  ],\n  "policy": "mesh-llm-dynamic-host-v2",\n  "rejected_imports": []\n}\n'
} >"$report.tmp"
mv "$report.tmp" "$report"
