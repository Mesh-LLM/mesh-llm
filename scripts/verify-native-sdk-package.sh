#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT=""
ARTIFACT_DIR_RESULT=""
ARTIFACT_INDEX=0
trap 'rm -rf "$TMP_ROOT"' EXIT

usage() {
    cat >&2 <<'EOF'
Usage: scripts/verify-native-sdk-package.sh <artifact-dir-or-tar.gz> [...]

Verifies MeshLLM native SDK runtime artifacts:
  - required archive checksum sidecar
  - archive paths and links cannot escape the extraction directory
  - manifest schema and required fields
  - artifact directory name matches manifest artifact_id
  - native library exists
  - library_sha256 matches the primary library
  - artifact_id matches platform/flavor
EOF
}

verify_sidecar_checksum() {
    local archive="$1"
    python3 "$SCRIPT_DIR/verify-checksum-sidecar.py" "$archive"
}

artifact_dir_for_input() {
    local input="$1"

    if [[ -d "$input" ]]; then
        ARTIFACT_DIR_RESULT="$input"
        return 0
    fi

    case "$input" in
        *.tar.gz|*.tgz) ;;
        *)
            echo "unsupported native SDK artifact input: $input" >&2
            exit 1
            ;;
    esac

    verify_sidecar_checksum "$input" || return 1

    if [[ -z "$TMP_ROOT" ]]; then
        TMP_ROOT="$(mktemp -d)"
    fi

    local extract_dir
    extract_dir="$TMP_ROOT/artifact-$ARTIFACT_INDEX"
    ARTIFACT_INDEX=$((ARTIFACT_INDEX + 1))
    mkdir -p "$extract_dir"
    python3 "$SCRIPT_DIR/safe-extract-tar.py" "$input" "$extract_dir" ||
        return 1

    local count entry
    count="$(
        find "$extract_dir" -mindepth 1 -maxdepth 1 -print |
            wc -l |
            tr -d ' '
    )"
    entry="$(find "$extract_dir" -mindepth 1 -maxdepth 1 -print -quit)"
    if [[ "$count" != "1" || ! -d "$entry" || -L "$entry" ]]; then
        echo "expected archive to contain one top-level artifact directory: $input" >&2
        return 1
    fi

    ARTIFACT_DIR_RESULT="$entry"
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
from pathlib import Path, PurePosixPath
import re
import sys

artifact_dir, manifest_path = sys.argv[1:3]
artifact_root = Path(artifact_dir).resolve(strict=True)
windows_drive = re.compile(r"^[A-Za-z]:")


def artifact_file(label, raw_path):
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise SystemExit(f"{label} path must be a non-empty string")
    if "\\" in raw_path:
        raise SystemExit(
            f"{label} path must use forward slashes inside the artifact: "
            f"{raw_path}"
        )
    rel_path = PurePosixPath(raw_path)
    if (
        rel_path.is_absolute()
        or windows_drive.match(raw_path)
        or ".." in rel_path.parts
    ):
        raise SystemExit(
            f"{label} must be a relative path inside the artifact: {raw_path}"
        )
    candidate = artifact_root.joinpath(*rel_path.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise SystemExit(f"missing {label}: {candidate}") from None
    try:
        resolved.relative_to(artifact_root)
    except ValueError:
        raise SystemExit(
            f"{label} path resolves outside the artifact: {raw_path}"
        ) from None
    if not resolved.is_file():
        raise SystemExit(f"missing {label}: {candidate}")
    return resolved


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

string_fields = (
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
    "library_sha256",
)
for field in string_fields:
    if not isinstance(manifest[field], str) or not manifest[field]:
        raise SystemExit(f"{field} must be a non-empty string")

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

target_contracts = {
    "aarch64-apple-darwin": ("darwin-aarch64", "macos", "aarch64"),
    "x86_64-apple-darwin": ("darwin-x86_64", "macos", "x86_64"),
    "x86_64-unknown-linux-gnu": ("linux-x86_64", "linux", "x86_64"),
    "aarch64-unknown-linux-gnu": ("linux-aarch64", "linux", "aarch64"),
    "aarch64-linux-android": ("android-arm64-v8a", "linux", "aarch64"),
    "armv7-linux-androideabi": ("android-armeabi-v7a", "linux", "arm"),
    "x86_64-linux-android": ("android-x86_64", "linux", "x86_64"),
    "x86_64-pc-windows-msvc": ("windows-x86_64", "windows", "x86_64"),
}
target_contract = target_contracts.get(manifest["target_triple"])
if target_contract is None:
    raise SystemExit(
        f"unsupported target_triple: {manifest['target_triple']}"
    )
expected_platform, expected_os, expected_arch = target_contract
if manifest["platform"] != expected_platform:
    raise SystemExit(
        "platform does not match target_triple: "
        f"{manifest['platform']} != {expected_platform}"
    )
if manifest["os"] != expected_os:
    raise SystemExit(f"os does not match target_triple: {manifest['os']} != {expected_os}")
if manifest["arch"] != expected_arch:
    raise SystemExit(f"arch does not match target_triple: {manifest['arch']} != {expected_arch}")

flavor_for_backend = {
    "cpu": "cpu",
    "metal": "metal",
    "cuda": "cuda",
    "cuda-blackwell": "cuda-blackwell",
    "rocm": "rocm",
    "hip": "rocm",
    "vulkan": "vulkan",
}
expected_flavor = flavor_for_backend.get(manifest["backend"])
if expected_flavor is None:
    raise SystemExit(f"unsupported native SDK backend: {manifest['backend']}")
if manifest["flavor"] != expected_flavor:
    raise SystemExit(
        "flavor does not match backend: "
        f"{manifest['flavor']} != {expected_flavor}"
    )

dir_name = Path(artifact_dir).name
if dir_name != manifest["artifact_id"]:
    raise SystemExit(f"artifact directory name does not match artifact_id: {dir_name} != {manifest['artifact_id']}")

library = manifest["library"]
library_paths = manifest["library_paths"]
if not isinstance(library_paths, list) or not library_paths:
    raise SystemExit("library_paths must be a non-empty list")
if library not in library_paths:
    raise SystemExit("library_paths must include the primary library")
if not isinstance(manifest["requirements"], list):
    raise SystemExit("requirements must be a list")
for rel_path in library_paths:
    artifact_file("library_paths entry", rel_path)

library_path = artifact_file("library", library)
with open(library_path, "rb") as fh:
    actual = hashlib.sha256(fh.read()).hexdigest()
if actual != manifest["library_sha256"]:
    raise SystemExit(
        f"library_sha256 mismatch for {library}: {actual} != {manifest['library_sha256']}"
    )

legacy_uniffi_library = manifest.get("uniffi_library")
if legacy_uniffi_library:
    legacy_path = artifact_file("uniffi_library", legacy_uniffi_library)
    with open(legacy_path, "rb") as fh:
        legacy_actual = hashlib.sha256(fh.read()).hexdigest()
    if legacy_actual != actual:
        raise SystemExit(
            f"uniffi_library checksum mismatch: {legacy_actual} != {actual}"
        )

if not isinstance(manifest["features"], list) or not all(
    isinstance(feature, str) and feature for feature in manifest["features"]
):
    raise SystemExit("features must be a list of non-empty strings")
features = set(manifest["features"])
for feature in ("mesh-inference", "model-management", "local-serving", "chat", "responses"):
    if feature not in features:
        raise SystemExit(f"missing feature marker: {feature}")

platform = manifest["platform"]
library_name = PurePosixPath(library).name
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
    artifact_dir_for_input "$input"
    verify_artifact_dir "$ARTIFACT_DIR_RESULT"
done
