#!/usr/bin/env bash
set -euo pipefail

TMP_ROOT=""
trap 'rm -rf "$TMP_ROOT"' EXIT

usage() {
    cat >&2 <<'EOF'
Usage: scripts/verify-native-runtime-package.sh <artifact-dir-or-tar.gz> [...]

Verifies MeshLLM native runtime artifacts:
  - manifest schema and resolver fields
  - artifact directory name matches runtime.id
  - all runtime.libraries exist
  - library_sha256 matches the primary library
  - archive checksum sidecar when present
EOF
}

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        echo "shasum or sha256sum is required" >&2
        exit 1
    fi
}

verify_sidecar_checksum() {
    local archive="$1"
    local sidecar="$archive.sha256"
    if [[ ! -f "$sidecar" ]]; then
        return 0
    fi
    local expected actual
    expected="$(awk '{print $1}' "$sidecar")"
    actual="$(sha256_file "$archive")"
    if [[ "$expected" != "$actual" ]]; then
        echo "archive checksum mismatch: $archive" >&2
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        exit 1
    fi
}

artifact_dir_for_input() {
    local input="$1"
    if [[ -d "$input" ]]; then
        printf '%s\n' "$input"
        return 0
    fi
    case "$input" in
        *.tar.gz|*.tgz) ;;
        *)
            echo "unsupported native runtime artifact input: $input" >&2
            exit 1
            ;;
    esac
    verify_sidecar_checksum "$input"
    if [[ -z "$TMP_ROOT" ]]; then
        TMP_ROOT="$(mktemp -d)"
    fi
    local extract_dir
    extract_dir="$TMP_ROOT/$(basename "$input" | tr -cd 'A-Za-z0-9_.-')"
    mkdir -p "$extract_dir"
    tar -C "$extract_dir" -xzf "$input"
    local count
    count="$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
    if [[ "$count" != "1" ]]; then
        echo "expected archive to contain one top-level artifact directory: $input" >&2
        exit 1
    fi
    find "$extract_dir" -mindepth 1 -maxdepth 1 -type d -print -quit
}

verify_artifact_dir() {
    local artifact_dir="$1"
    local manifest="$artifact_dir/manifest.json"
    if [[ ! -f "$manifest" ]]; then
        echo "missing manifest: $manifest" >&2
        exit 1
    fi
    python3 - "$artifact_dir" "$manifest" <<'PY'
import hashlib
import json
import os
import sys

artifact_dir, manifest_path = sys.argv[1:3]
with open(manifest_path, encoding="utf-8") as fh:
    manifest = json.load(fh)

if "runtime" not in manifest:
    raise SystemExit("missing manifest field: runtime")
runtime = manifest["runtime"]
required = {
    "id",
    "mesh_version",
    "skippy_abi",
    "platform",
    "backend",
    "libraries",
}
missing = sorted(required - runtime.keys())
if missing:
    raise SystemExit(f"missing runtime manifest field(s): {', '.join(missing)}")
if os.path.basename(os.path.normpath(artifact_dir)) != runtime["id"]:
    raise SystemExit("artifact directory name must match runtime id")
if not isinstance(runtime["libraries"], list) or not runtime["libraries"]:
    raise SystemExit("runtime libraries must be a non-empty list")
platform = runtime["platform"]
if not isinstance(platform, dict) or not platform.get("os") or not platform.get("arch"):
    raise SystemExit("runtime platform must declare os and arch")
backend = runtime["backend"]
if not isinstance(backend, dict) or not backend.get("kind"):
    raise SystemExit("runtime backend must declare kind")

for rel_path in runtime["libraries"]:
    if os.path.isabs(rel_path) or ".." in rel_path.split(os.sep):
        raise SystemExit(f"library path must be relative inside the artifact: {rel_path}")
    path = os.path.join(artifact_dir, rel_path)
    if not os.path.isfile(path):
        raise SystemExit(f"missing library: {path}")

build = manifest.get("build") or {}
library_sha256 = build.get("library_sha256")
primary_library = build.get("primary_library") or runtime["libraries"][0]
if library_sha256:
    primary = os.path.join(artifact_dir, primary_library)
    with open(primary, "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    if actual != library_sha256:
        raise SystemExit(
            f"library_sha256 mismatch for {primary_library}: {actual} != {library_sha256}"
        )
PY
    echo "verified native runtime artifact: $artifact_dir"
}

if [[ "$#" -lt 1 ]]; then
    usage
    exit 1
fi

for input in "$@"; do
    artifact_dir="$(artifact_dir_for_input "$input")"
    verify_artifact_dir "$artifact_dir"
done
