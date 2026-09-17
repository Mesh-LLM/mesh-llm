#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=""
REPO="${GITHUB_REPOSITORY:-Mesh-LLM/mesh-llm}"
TAG="${RELEASE_TAG:-}"
RUNTIME_VERSION=""
TMP_ROOT=""
trap 'rm -rf "$TMP_ROOT"' EXIT

usage() {
    cat >&2 <<'EOF'
Usage: scripts/generate-native-runtime-release-manifest.sh --tag TAG --out FILE [--repo OWNER/REPO] [--runtime-version VERSION] <native-runtime.tar.gz> [...]

Generates native-runtimes.json for a GitHub release from packaged native
runtime artifacts. Each artifact archive must have its canonical .sha256
sidecar and contain a manifest.json with the native runtime resolver fields
emitted by package-native-runtime.sh. The tag locates the published archives;
--runtime-version selects their required runtime release (defaults to Skippy
RUNTIME_VERSION), independently of the product publication tag.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --out)
            OUT="${2:?missing output file}"
            shift 2
            ;;
        --repo)
            REPO="${2:?missing repo}"
            shift 2
            ;;
        --runtime-version)
            RUNTIME_VERSION="${2:?missing runtime version}"
            shift 2
            ;;
        --tag)
            TAG="${2:?missing release tag}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ -z "$OUT" || -z "$TAG" || "$#" -lt 1 ]]; then
    usage
    exit 1
fi

if [[ -z "$RUNTIME_VERSION" ]]; then
    RUNTIME_VERSION="$(cat "$SCRIPT_DIR/../crates/skippy-native-runtime/RUNTIME_VERSION")"
fi
if [[ ! "$RUNTIME_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    echo "invalid Skippy runtime release version: $RUNTIME_VERSION" >&2
    exit 1
fi

if [[ -z "$TMP_ROOT" ]]; then
    TMP_ROOT="$(mktemp -d)"
fi

for archive in "$@"; do
    "$SCRIPT_DIR/verify-native-runtime-package.sh" --portable "$archive"
done

python3 - \
    "$OUT" \
    "$REPO" \
    "$TAG" \
    "$RUNTIME_VERSION" \
    "$TMP_ROOT" \
    "$SCRIPT_DIR/safe-extract-tar.py" \
    "$@" <<'PY'
import hashlib
import json
import os
import subprocess
import sys

(
    out,
    repo,
    tag,
    release_version,
    tmp_root,
    safe_extractor,
    *archives,
) = sys.argv[1:]
artifacts = []
catalog_runtime_version = None
skippy_abi = None

required = {
    "id",
    "release_version",
    "skippy_abi",
    "platform",
    "backend",
    "libraries",
    "files",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for index, archive in enumerate(archives):
    archive = os.path.abspath(archive)
    archive_sha256 = sha256_file(archive)

    extract_dir = os.path.join(tmp_root, f"archive-{index}")
    extraction_result = subprocess.run(
        [sys.executable, safe_extractor, archive, extract_dir],
        check=False,
    )
    if extraction_result.returncode != 0:
        raise SystemExit(
            f"unsafe or invalid native runtime archive: {archive}"
        )

    manifest_paths = []
    for root, _, files in os.walk(extract_dir):
        if "manifest.json" in files:
            manifest_paths.append(os.path.join(root, "manifest.json"))
    if len(manifest_paths) != 1:
        raise SystemExit(f"expected exactly one manifest.json in {archive}, found {len(manifest_paths)}")

    with open(manifest_paths[0], encoding="utf-8") as fh:
        manifest = json.load(fh)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 2:
        raise SystemExit(f"{archive} requires native runtime schema_version 2; import legacy caches explicitly")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise SystemExit(f"{archive} is missing runtime manifest")
    missing = sorted(required - runtime.keys())
    if missing:
        raise SystemExit(f"{archive} is missing native runtime field(s): {', '.join(missing)}")

    runtime_version = runtime["release_version"]
    normalized_runtime_version = (
        runtime_version[1:]
        if runtime_version.startswith("v")
        else runtime_version
    )
    if normalized_runtime_version != release_version:
        raise SystemExit(
            f"{archive} release_version {runtime_version} does not match "
            f"requested runtime release {release_version}"
        )

    if catalog_runtime_version is None:
        catalog_runtime_version = runtime_version
    elif runtime_version != catalog_runtime_version:
        raise SystemExit(
            f"mixed runtime releases in native runtime artifacts: {runtime_version} != {catalog_runtime_version}"
        )
    if skippy_abi is None:
        skippy_abi = runtime["skippy_abi"]
    elif runtime["skippy_abi"] != skippy_abi:
        raise SystemExit(
            f"mixed Skippy ABI versions in native runtime artifacts: {runtime['skippy_abi']} != {skippy_abi}"
        )

    artifact = dict(runtime)
    artifact["url"] = (
        f"https://github.com/{repo}/releases/download/{tag}/{os.path.basename(archive)}"
    )
    artifact["sha256"] = archive_sha256
    artifacts.append(artifact)

if catalog_runtime_version is None:
    raise SystemExit("no native runtime artifacts supplied")

artifacts.sort(key=lambda item: item["id"])
release_manifest = {
    "schema_version": 2,
    "release_version": catalog_runtime_version,
    "skippy_abi": skippy_abi,
    "artifacts": artifacts,
}
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(release_manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY

echo "generated native runtime release manifest: $OUT"
