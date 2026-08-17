#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Usage: $0 <mesh-llm-binary> <out-dir> [backend]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH_LLM="$1"
OUT_DIR="$2"
BACKEND="${3:-cpu}"
RUNTIME_CACHE="${MESH_LLM_NATIVE_RUNTIME_CACHE_DIR:-${RUNNER_TEMP:-/tmp}/mesh-llm-native-runtime-cache}"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "Missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi

cd "$REPO_ROOT"

native_runtime_dir="$(scripts/ci-prepare-native-runtime.sh "$OUT_DIR" "$BACKEND")"

echo "Installing CI native runtime:" >&2
echo "  runtime: $native_runtime_dir" >&2
echo "  cache:   $RUNTIME_CACHE" >&2
python3 - "$native_runtime_dir" "$RUNTIME_CACHE" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
cache = Path(sys.argv[2])
manifest_path = source / "manifest.json"

with manifest_path.open("r", encoding="utf-8") as fh:
    manifest = json.load(fh)

runtime = manifest["runtime"]
runtime_id = runtime["id"]
mesh_version = runtime.get("mesh_version") or "unknown"
libraries = runtime.get("libraries") or []
if not runtime_id.strip():
    raise SystemExit(f"native runtime id is empty in {manifest_path}")
if not mesh_version.strip():
    raise SystemExit(f"native runtime mesh_version is empty in {manifest_path}")
if not libraries:
    raise SystemExit(f"native runtime libraries are empty in {manifest_path}")

for library in libraries:
    library_path = source / library
    if not library_path.is_file():
        raise SystemExit(f"native runtime library is missing: {library_path}")

target = cache / mesh_version / runtime_id
if target.exists():
    shutil.rmtree(target)
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, target)
print(f"Installed CI native runtime: {target}", file=sys.stderr)
PY

printf '%s\n' "$RUNTIME_CACHE"
