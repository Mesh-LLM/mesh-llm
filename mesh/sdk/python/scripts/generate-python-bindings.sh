#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == /* && -x "$1" && "$2" == /* && -f "$2" && ! -L "$2" ]] || { echo 'usage: generate-python-bindings.sh ABS_UNIFFI_0_32_BINDGEN ABS_PREBUILT_MESH_FFI_LIBRARY' >&2; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH_ROOT="$(cd "$ROOT/../../.." && pwd)"
[[ -f "$MESH_ROOT/mesh/crates/mesh-llm-ffi/src/mesh_ffi.udl" && ! -L "$MESH_ROOT/mesh/crates/mesh-llm-ffi/src/mesh_ffi.udl" ]] || exit 1
[[ "$("$1" --version)" == 'uniffi-bindgen 0.32.0' ]] || { echo 'UniFFI 0.32.0 required' >&2; exit 1; }
# Resolve the matching Mesh crate/UDL while deriving checksums from the actual library metadata.
cd "$MESH_ROOT"
binding_language=python
"$1" generate "$2" --language "$binding_language" --out-dir "$ROOT/src/meshllm/_generated" --no-format --crate meshllm_ffi --config "$ROOT/uniffi.toml" --metadata-no-deps
perl -pi -e 's/[ \t]+$//' "$ROOT/src/meshllm/_generated/mesh_ffi.py"
