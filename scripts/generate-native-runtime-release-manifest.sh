#!/usr/bin/env bash
set -euo pipefail

OUT=""
REPO="${GITHUB_REPOSITORY:-Mesh-LLM/mesh-llm}"
TAG="${RELEASE_TAG:-}"
TMP_ROOT=""
trap 'rm -rf "$TMP_ROOT"' EXIT

usage() {
    cat >&2 <<'EOF'
Usage: scripts/generate-native-runtime-release-manifest.sh --tag TAG --out FILE [--repo OWNER/REPO] <native-runtime.tar.gz> [...]

Generates native-runtimes.json for a GitHub release from packaged native
runtime artifacts. Each artifact archive must contain a manifest.json with the
native runtime resolver fields emitted by package-native-runtime.sh.
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

if [[ -z "$TMP_ROOT" ]]; then
    TMP_ROOT="$(mktemp -d)"
fi

python3 - "$OUT" "$REPO" "$TAG" "$TMP_ROOT" "$@" <<'PY'
import hashlib
import json
import os
import sys
import tarfile

out, repo, tag, tmp_root, *archives = sys.argv[1:]
artifacts = []
mesh_version = None
skippy_abi = None

required = {
    "id",
    "mesh_version",
    "skippy_abi",
    "platform",
    "backend",
    "libraries",
}

for archive in archives:
    archive = os.path.abspath(archive)
    with open(archive, "rb") as fh:
        archive_sha256 = hashlib.sha256(fh.read()).hexdigest()

    extract_dir = os.path.join(tmp_root, os.path.basename(archive).replace(os.sep, "_"))
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)

    manifest_paths = []
    for root, _, files in os.walk(extract_dir):
        if "manifest.json" in files:
            manifest_paths.append(os.path.join(root, "manifest.json"))
    if len(manifest_paths) != 1:
        raise SystemExit(f"expected exactly one manifest.json in {archive}, found {len(manifest_paths)}")

    with open(manifest_paths[0], encoding="utf-8") as fh:
        manifest = json.load(fh)
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise SystemExit(f"{archive} is missing runtime manifest")
    missing = sorted(required - runtime.keys())
    if missing:
        raise SystemExit(f"{archive} is missing native runtime field(s): {', '.join(missing)}")

    if mesh_version is None:
        mesh_version = runtime["mesh_version"]
    elif runtime["mesh_version"] != mesh_version:
        raise SystemExit(
            f"mixed mesh versions in native runtime artifacts: {runtime['mesh_version']} != {mesh_version}"
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

if mesh_version is None:
    raise SystemExit("no native runtime artifacts supplied")

artifacts.sort(key=lambda item: item["id"])
release_manifest = {
    "mesh_version": mesh_version,
    "skippy_abi": skippy_abi,
    "artifacts": artifacts,
}
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(release_manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY

echo "generated native runtime release manifest: $OUT"
