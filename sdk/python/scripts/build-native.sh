#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="$REPO_ROOT/sdk/python/src/meshllm/_generated"
FEATURE_ARGS=()
if [ "${MESH_PYTHON_EMBEDDED_RUNTIME:-0}" = "1" ]; then
    FEATURE_ARGS=(--features embedded-runtime)
fi

cd "$REPO_ROOT"
CARGO_INCREMENTAL=0 just with-lld cargo build --locked --release -p mesh-llm-ffi "${FEATURE_ARGS[@]}"
mkdir -p "$OUT_DIR"

case "$(uname -s)" in
    Darwin)
        cp "$REPO_ROOT/target/release/libmeshllm_ffi.dylib" "$OUT_DIR/libuniffi.dylib"
        ;;
    Linux)
        cp "$REPO_ROOT/target/release/libmeshllm_ffi.so" "$OUT_DIR/libuniffi.so"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        cp "$REPO_ROOT/target/release/meshllm_ffi.dll" "$OUT_DIR/uniffi.dll"
        ;;
    *)
        echo "Unsupported host platform: $(uname -s)" >&2
        exit 1
        ;;
esac

echo "Staged the Python SDK native library in $OUT_DIR"
