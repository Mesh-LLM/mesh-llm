#!/usr/bin/env bash
set -euo pipefail
# Production only: explicitly supplied, already built Mesh FFI bridge.
[[ $# == 1 && "$1" == /* && -f "$1" && ! -L "$1" ]] || { echo 'usage: build-native.sh ABS_PREBUILT_MESH_FFI_LIBRARY' >&2; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$(basename "$1")" in
  libmeshllm_ffi.dylib) leaf=libuniffi.dylib ;;
  libmeshllm_ffi.so) leaf=libuniffi.so ;;
  meshllm_ffi.dll) leaf=uniffi.dll ;;
  *) echo 'unsupported prebuilt Mesh FFI library name' >&2; exit 1 ;;
esac
cp "$1" "$ROOT/src/meshllm/_generated/$leaf"
