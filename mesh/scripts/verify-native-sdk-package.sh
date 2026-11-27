#!/usr/bin/env bash
set -euo pipefail

TMP_ROOT=""
trap 'rm -rf "$TMP_ROOT"' EXIT

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <native-sdk-artifact-dir-or-tar.gz>..." >&2
    exit 1
fi

for input in "$@"; do
    if [[ -d "$input" ]]; then
        artifact_dir="$input"
    else
        case "$input" in
            *.tar.gz|*.tgz) ;;
            *) echo "unsupported native SDK artifact input: $input" >&2; exit 1 ;;
        esac
        (cd "$(dirname "$input")" && { shasum -a 256 -c "$(basename "$input").sha256" 2>/dev/null || sha256sum -c "$(basename "$input").sha256"; }) >/dev/null ||
            { echo "checksum mismatch: $input" >&2; exit 1; }
        if [[ -z "$TMP_ROOT" ]]; then
            TMP_ROOT="$(mktemp -d)"
        fi
        extract_dir="$(mktemp -d "$TMP_ROOT/artifact.XXXXXX")"
        tar -xzf "$input" -C "$extract_dir"
        shopt -s nullglob dotglob
        entries=("$extract_dir"/*)
        if [[ "${#entries[@]}" != "1" || ! -d "${entries[0]}" || -L "${entries[0]}" ]]; then
            echo "expected archive to contain one top-level artifact directory: $input" >&2
            exit 1
        fi
        artifact_dir="${entries[0]}"
    fi
    manifest="$artifact_dir/manifest.json"
    jq -e '.schema_version == 1 and .artifact_id and .sdk_version and .target_triple and .library and .uniffi_library and .library_sha256 and
        (.features | type == "array" and all(.[]; type == "string"))' "$manifest" >/dev/null ||
        { echo "invalid native SDK manifest: $manifest" >&2; exit 1; }
    for rel in $(jq -r '.library, .uniffi_library' "$manifest"); do
        [[ "$rel" != /* && "/$rel/" != *"/../"* && -f "$artifact_dir/$rel" ]] || { echo "manifest library missing or unsafe: $rel" >&2; exit 1; }
    done
    for rel in $(jq -r '.library, .uniffi_library' "$manifest"); do
        actual="$(shasum -a 256 "$artifact_dir/$rel" 2>/dev/null || sha256sum "$artifact_dir/$rel")"
        [[ "${actual%% *}" == "$(jq -r .library_sha256 "$manifest")" ]] || { echo "library_sha256 mismatch for $rel: $manifest" >&2; exit 1; }
    done
    echo "verified native SDK artifact: $artifact_dir"
done
