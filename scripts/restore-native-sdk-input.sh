#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "Usage: $0 <artifact-download-dir> <extract-dir> <target> <backend> <profile>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
download_dir="$1"
extract_dir="$2"
expected_target="$3"
expected_backend="$4"
expected_profile="$5"

case "$expected_target" in
    x86_64-*) expected_arch="x86_64" ;;
    aarch64-*) expected_arch="aarch64" ;;
    *)
        echo "unsupported native SDK target architecture: $expected_target" >&2
        exit 1
        ;;
esac
case "$(uname -m)" in
    x86_64|amd64) runner_arch="x86_64" ;;
    aarch64|arm64) runner_arch="aarch64" ;;
    *) runner_arch="$(uname -m)" ;;
esac
if [[ "$runner_arch" != "$expected_arch" ]]; then
    echo "native SDK target/runner architecture mismatch: $expected_target on $runner_arch" >&2
    exit 1
fi

if [[ ! -d "$download_dir" || -L "$download_dir" ]]; then
    echo "native SDK artifact download directory is invalid: $download_dir" >&2
    exit 1
fi
if [[ -L "$extract_dir" ]]; then
    echo "native SDK extraction directory must not be a symlink: $extract_dir" >&2
    exit 1
fi
mkdir -p "$extract_dir"

shopt -s nullglob dotglob
download_entries=("$download_dir"/*)
archives=("$download_dir"/*.tar.gz)
checksums=("$download_dir"/*.tar.gz.sha256)
extract_entries=("$extract_dir"/*)
if [[ "${#download_entries[@]}" -ne 2 ||
      "${#archives[@]}" -ne 1 ||
      "${#checksums[@]}" -ne 1 ]]; then
    echo "native SDK input must contain exactly one archive and checksum" >&2
    exit 1
fi
if [[ "${checksums[0]}" != "${archives[0]}.sha256" ]]; then
    echo "native SDK checksum sidecar does not match archive" >&2
    exit 1
fi
if [[ "${#extract_entries[@]}" -ne 0 ]]; then
    echo "native SDK extraction directory must be empty: $extract_dir" >&2
    exit 1
fi

"$REPO_ROOT/scripts/verify-native-sdk-package.sh" "${archives[0]}" >&2
tar -xzf "${archives[0]}" -C "$extract_dir"

artifact_entries=("$extract_dir"/*)
if [[ "${#artifact_entries[@]}" -ne 1 ||
      ! -d "${artifact_entries[0]}" ||
      -L "${artifact_entries[0]}" ]]; then
    echo "native SDK archive must extract one artifact directory" >&2
    exit 1
fi
artifact_dir="${artifact_entries[0]}"
"$REPO_ROOT/scripts/verify-native-sdk-package.sh" "$artifact_dir" >&2

jq -e --arg t "$expected_target" --arg b "$expected_backend" --arg p "$expected_profile" \
    '.target_triple == $t and .backend == $b and .cargo_profile == $p' "$artifact_dir/manifest.json" >/dev/null ||
    { echo "native SDK identity does not match $expected_target/$expected_backend/$expected_profile" >&2; exit 1; }

printf '%s\n' "$artifact_dir"
