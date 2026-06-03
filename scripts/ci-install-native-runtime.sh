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
MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$RUNTIME_CACHE" \
    "$MESH_LLM" runtime install \
        --bundle-dir "$native_runtime_dir" \
        --cache-dir "$RUNTIME_CACHE" >&2

printf '%s\n' "$RUNTIME_CACHE"
