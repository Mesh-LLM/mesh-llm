#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/automation.sh"
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
        mesh_automation artifact verify-checksum "$input"
        if [[ -z "$TMP_ROOT" ]]; then
            TMP_ROOT="$(mktemp -d)"
        fi
        extract_dir="$(mktemp -d "$TMP_ROOT/artifact.XXXXXX")"
        mesh_automation artifact extract-tar "$input" "$extract_dir"
        shopt -s nullglob dotglob
        entries=("$extract_dir"/*)
        if [[ "${#entries[@]}" != "1" || ! -d "${entries[0]}" || -L "${entries[0]}" ]]; then
            echo "expected archive to contain one top-level artifact directory: $input" >&2
            exit 1
        fi
        artifact_dir="${entries[0]}"
    fi
    mesh_automation prepared-input native-sdk-manifest "$artifact_dir" "$artifact_dir/manifest.json"
    echo "verified native SDK artifact: $artifact_dir"
done
