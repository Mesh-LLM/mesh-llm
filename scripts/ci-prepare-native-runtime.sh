#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    echo "Usage: $0 <out-dir> [backend]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$1"
BACKEND="${2:-cpu}"
BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-ci-runtime-${BACKEND}"

cd "$REPO_ROOT"

rm -rf "$OUT_DIR"
LLAMA_STAGE_LINK_MODE=dynamic \
LLAMA_STAGE_BACKEND="$BACKEND" \
LLAMA_STAGE_BUILD_DIR="$BUILD_DIR" \
LLAMA_BUILD_DIR="$BUILD_DIR" \
    scripts/package-native-runtime.sh \
        --build \
        --backend "$BACKEND" \
        --out "$OUT_DIR" >&2

scripts/verify-native-runtime-package.sh "$OUT_DIR"/meshllm-native-runtime-*.tar.gz >&2

runtime_dir="$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'meshllm-native-runtime-*' | sort | head -n 1)"
if [[ -z "$runtime_dir" ]]; then
    echo "native runtime artifact directory was not produced" >&2
    exit 1
fi

printf '%s\n' "$runtime_dir"
