#!/usr/bin/env bash
set -euo pipefail

TMP_ROOT=""
trap 'rm -rf "$TMP_ROOT"' EXIT

usage() {
    cat >&2 <<'EOF'
Usage: scripts/verify-native-sdk-package.sh <artifact-dir-or-tar.gz> [...]

Verifies MeshLLM native SDK runtime artifacts:
  - archive checksum sidecar when present
  - manifest schema and required fields
  - artifact directory name matches manifest artifact_id
  - native library exists
  - library_sha256 matches the primary library
  - artifact_id matches platform/flavor
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
            echo "unsupported native SDK artifact input: $input" >&2
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

required = [
    "schema_version",
    "artifact_id",
    "native_runtime_id",
    "sdk_version",
    "mesh_version",
    "target_triple",
    "platform",
    "os",
    "arch",
    "backend",
    "flavor",
    "library",
    "library_paths",
    "library_sha256",
    "requirements",
    "features",
]
missing = [key for key in required if key not in manifest]
if missing:
    raise SystemExit(f"missing manifest field(s): {', '.join(missing)}")

if manifest["schema_version"] != 1:
    raise SystemExit(f"unsupported schema_version: {manifest['schema_version']!r}")

expected_artifact_id = f"meshllm-native-{manifest['platform']}-{manifest['flavor']}"
if manifest["artifact_id"] != expected_artifact_id:
    raise SystemExit(
        f"artifact_id does not match platform/flavor: {manifest['artifact_id']} != {expected_artifact_id}"
    )
if manifest["native_runtime_id"] != manifest["artifact_id"]:
    raise SystemExit(
        f"native_runtime_id must match artifact_id: {manifest['native_runtime_id']} != {manifest['artifact_id']}"
    )
if manifest["mesh_version"] != manifest["sdk_version"]:
    raise SystemExit(
        f"mesh_version must match sdk_version: {manifest['mesh_version']} != {manifest['sdk_version']}"
    )

expected_os = {
    "aarch64-apple-darwin": "macos",
    "x86_64-apple-darwin": "macos",
    "x86_64-unknown-linux-gnu": "linux",
    "aarch64-unknown-linux-gnu": "linux",
    "aarch64-linux-android": "linux",
    "armv7-linux-androideabi": "linux",
    "x86_64-linux-android": "linux",
    "x86_64-pc-windows-msvc": "windows",
}.get(manifest["target_triple"])
expected_arch = {
    "aarch64-apple-darwin": "aarch64",
    "x86_64-apple-darwin": "x86_64",
    "x86_64-unknown-linux-gnu": "x86_64",
    "aarch64-unknown-linux-gnu": "aarch64",
    "aarch64-linux-android": "aarch64",
    "armv7-linux-androideabi": "arm",
    "x86_64-linux-android": "x86_64",
    "x86_64-pc-windows-msvc": "x86_64",
}.get(manifest["target_triple"])
if expected_os and manifest["os"] != expected_os:
    raise SystemExit(f"os does not match target_triple: {manifest['os']} != {expected_os}")
if expected_arch and manifest["arch"] != expected_arch:
    raise SystemExit(f"arch does not match target_triple: {manifest['arch']} != {expected_arch}")

dir_name = os.path.basename(os.path.normpath(artifact_dir))
if dir_name != manifest["artifact_id"]:
    raise SystemExit(f"artifact directory name does not match artifact_id: {dir_name} != {manifest['artifact_id']}")

library = manifest["library"]
if library not in manifest["library_paths"]:
    raise SystemExit("library_paths must include the primary library")
if not isinstance(manifest["requirements"], list):
    raise SystemExit("requirements must be a list")
for key, rel_path in (("library", library),):
    if os.path.isabs(rel_path) or ".." in rel_path.split(os.sep):
        raise SystemExit(f"{key} must be a relative path inside the artifact: {rel_path}")
    path = os.path.join(artifact_dir, rel_path)
    if not os.path.isfile(path):
        raise SystemExit(f"missing {key}: {path}")

library_path = os.path.join(artifact_dir, library)
with open(library_path, "rb") as fh:
    actual = hashlib.sha256(fh.read()).hexdigest()
if actual != manifest["library_sha256"]:
    raise SystemExit(
        f"library_sha256 mismatch for {library}: {actual} != {manifest['library_sha256']}"
    )

legacy_uniffi_library = manifest.get("uniffi_library")
if legacy_uniffi_library:
    if os.path.isabs(legacy_uniffi_library) or ".." in legacy_uniffi_library.split(os.sep):
        raise SystemExit(
            f"uniffi_library must be a relative path inside the artifact: {legacy_uniffi_library}"
        )
    legacy_path = os.path.join(artifact_dir, legacy_uniffi_library)
    if not os.path.isfile(legacy_path):
        raise SystemExit(f"missing uniffi_library: {legacy_path}")
    with open(legacy_path, "rb") as fh:
        legacy_actual = hashlib.sha256(fh.read()).hexdigest()
    if legacy_actual != actual:
        raise SystemExit(
            f"uniffi_library checksum mismatch: {legacy_actual} != {actual}"
        )

features = set(manifest["features"])
for feature in ("mesh-inference", "model-management", "local-serving", "chat", "responses"):
    if feature not in features:
        raise SystemExit(f"missing feature marker: {feature}")

platform = manifest["platform"]
library_name = os.path.basename(library)
if platform.startswith("darwin-") and not library_name.endswith(".dylib"):
    raise SystemExit(f"darwin artifact must contain a dylib: {library_name}")
if (platform.startswith("linux-") or platform.startswith("android-")) and not library_name.endswith(".so"):
    raise SystemExit(f"{platform} artifact must contain a .so: {library_name}")
if platform.startswith("windows-") and not library_name.endswith(".dll"):
    raise SystemExit(f"windows artifact must contain a .dll: {library_name}")
PY

    echo "verified native SDK artifact: $artifact_dir"
}

if [[ "$#" -lt 1 ]]; then
    usage
    exit 1
fi

for input in "$@"; do
    artifact_dir="$(artifact_dir_for_input "$input")"
    verify_artifact_dir "$artifact_dir"
done
