#!/usr/bin/env bash
# Download the Skippy native runtime published for the Skippy release pinned
# in Cargo.toml, check it against the release's checksums, and unpack it.
#
#   scripts/fetch-skippy-runtime.sh <out-dir> [backend]
#
# backend: cpu, metal, cuda, cuda12 or cuda13. Default: metal on macOS, cpu
# elsewhere. Plain `cuda` picks the package for the installed CUDA toolkit.
# <out-dir> ends up as a directory of packaged runtimes, the same shape a
# source build writes.
#
# Exit 3 means there is nothing to download: the pin is not a Skippy release,
# or that release has no package for this machine and backend. The caller can
# then build from source. Any other failure is a real error.
set -euo pipefail

(($# >= 1 && $# <= 2)) || { sed -n '2,14s/^# \{0,1\}//p' "${BASH_SOURCE[0]}" >&2; exit 2; }
out="$1"
backend="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pin="$("$REPO_ROOT/scripts/skippy-pin.sh")"
read -r url rev version <<<"$pin"
none() { echo "No Skippy runtime to download: $1" >&2; exit 3; }
command -v jq >/dev/null 2>&1 || none "jq is not installed"

case "$(uname -s)" in
    Darwin) os=macos ;;
    Linux) os=linux ;;
    *) none "Skippy v$version publishes no package for $(uname -s)" ;;
esac
arch="$(uname -m)"
[[ "$arch" != arm64 ]] || arch=aarch64
[[ -n "$backend" ]] || { [[ "$os" == macos ]] && backend=metal || backend=cpu; }
toolkit=""
case "$backend" in
    cuda12 | cuda13) toolkit="${backend#cuda}"; backend=cuda ;;
    cuda)
        toolkit="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1 || true)"
        [[ -n "$toolkit" ]] || none "cannot tell the CUDA toolkit version (no nvcc); name cuda12 or cuda13"
        ;;
esac

# The tag's commit; a peeled line is the commit behind an annotated tag.
tag_commit="$(git ls-remote "$url" "refs/tags/v$version" "refs/tags/v$version^{}" | sort -k2 | tail -1 | cut -f1)"
[[ "$tag_commit" == "$rev" ]] || none "the pinned commit ${rev:0:9} is not Skippy release v$version"

sha256_of() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | cut -d ' ' -f 1; }
get() { curl -fsSL --retry 3 --max-time 900 -o "$2" "$1"; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
base="${url%.git}/releases/download/v$version"
get "$base/native-runtimes.json" "$work/catalog.json"
get "$base/native-runtimes.json.sha256" "$work/catalog.sha256"
[[ "$(sha256_of "$work/catalog.json")" == "$(cut -d ' ' -f 1 "$work/catalog.sha256")" ]] ||
    { echo "native-runtimes.json does not match its published checksum" >&2; exit 1; }

package="$(jq -r --arg os "$os" --arg arch "$arch" --arg kind "$backend" --arg toolkit "$toolkit" '
    [.artifacts[] | select(.platform.os == $os and .platform.arch == $arch and .backend.kind == $kind
        and ($toolkit == "" or (.backend.cuda.toolkit_major | tostring) == $toolkit))]
    | if length == 1 then .[0] | [.url, .sha256, (.platform.min_glibc // "")] | @tsv else empty end' "$work/catalog.json")"
[[ -n "$package" ]] || none "Skippy v$version has no $backend$toolkit package for $os $arch"
IFS=$'\t' read -r package_url package_sha min_glibc <<<"$package"
if [[ -n "$min_glibc" ]]; then
    glibc_probe="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
    glibc="${glibc_probe##* }"
    [[ "$glibc_probe" == glibc\ * ]] || glibc=""
    [[ -n "$glibc" && "$(printf '%s\n' "$min_glibc" "$glibc" | sort -V | head -1)" == "$min_glibc" ]] ||
        none "the package needs glibc $min_glibc; this machine has ${glibc:-none}"
fi

echo "Downloading the Skippy v$version $backend$toolkit runtime: $package_url"
get "$package_url" "$work/package.tar.gz"
[[ "$(sha256_of "$work/package.tar.gz")" == "$package_sha" ]] ||
    { echo "$package_url does not match the checksum in native-runtimes.json" >&2; exit 1; }
mkdir "$work/unpacked"
tar -xzf "$work/package.tar.gz" -C "$work/unpacked"
compgen -G "$work/unpacked/native-runtimes/*/manifest.json" >/dev/null ||
    { echo "$package_url holds no packaged runtime" >&2; exit 1; }
rm -rf "$out"
mkdir -p "$out"
cp -R "$work/unpacked/native-runtimes/." "$out/"
echo "Skippy runtime: $out (v$version $backend$toolkit, downloaded)"
