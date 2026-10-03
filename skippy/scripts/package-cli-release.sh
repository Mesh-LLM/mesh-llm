#!/usr/bin/env bash
# Package one backend-neutral Skippy CLI separately from MeshLLM and runtimes.
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: skippy/scripts/package-cli-release.sh VERSION TARGET CLI_INPUT_DIR OUTPUT_DIR" >&2
    exit 2
fi

version="$1"
target="$2"
input_dir="$3"
output_dir="$4"
case "$version" in *[!A-Za-z0-9._+-]*|'') echo "invalid version: $version" >&2; exit 2;; esac
case "$target" in *[!A-Za-z0-9._-]*|'') echo "invalid target: $target" >&2; exit 2;; esac

binary=skippy
[[ "$target" == windows-* ]] && binary=skippy.exe
[[ -s "$input_dir/$binary" && -s "$input_dir/$binary.sha256" ]] || {
    echo "missing Skippy CLI input or checksum in $input_dir" >&2
    exit 1
}
if command -v sha256sum >/dev/null 2>&1; then
    (cd "$input_dir" && sha256sum --check "$binary.sha256")
else
    (cd "$input_dir" && shasum -a 256 --check "$binary.sha256")
fi

mkdir -p "$output_dir"
archive="skippy-${version}-${target}-cli.tar.gz"
COPYFILE_DISABLE=1 tar -czf "$output_dir/$archive" -C "$input_dir" "$binary" "$binary.sha256"
if command -v sha256sum >/dev/null 2>&1; then
    (cd "$output_dir" && sha256sum "$archive") > "$output_dir/$archive.sha256"
else
    (cd "$output_dir" && shasum -a 256 "$archive") > "$output_dir/$archive.sha256"
fi
echo "$output_dir/$archive"
