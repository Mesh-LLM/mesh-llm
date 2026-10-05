#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
UDL="$REPO_ROOT/mesh/crates/mesh-llm-ffi/src/mesh_ffi.udl"
OUT_DIR="$REPO_ROOT/mesh/sdk/python/src/meshllm/_generated"
BINDGEN_ROOT="${MESH_UNIFFI_BINDGEN_ROOT:-$HOME/.cache/mesh-llm/uniffi-bindgen-0.32.0}"
BINDGEN="$BINDGEN_ROOT/bin/uniffi-bindgen"

if [ ! -x "$BINDGEN" ]; then
    echo "Installing uniffi-bindgen 0.32.0 into $BINDGEN_ROOT ..."
    CARGO_INCREMENTAL=0 cargo install uniffi --version 0.32.0 --features cli \
        --bin uniffi-bindgen --root "$BINDGEN_ROOT"
fi

mkdir -p "$OUT_DIR"
"$BINDGEN" generate "$UDL" --language python --out-dir "$OUT_DIR" --no-format
perl -pi -e 's/[ \t]+$//' "$OUT_DIR/mesh_ffi.py"

echo "Regenerated $OUT_DIR/mesh_ffi.py"
