#!/usr/bin/env bash
set -euo pipefail

SKIP_BUILD=0
if [[ "$#" -eq 4 ]]; then
    if [[ "$4" != "--skip-build" ]]; then
        echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> [--skip-build]" >&2
        exit 1
    fi
    SKIP_BUILD=1
elif [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> [--skip-build]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

scripts/check-sdk-contract.sh
if [[ "$SKIP_BUILD" == "1" ]]; then
    scripts/package-sdk-console-assets.sh --sdk node --skip-build
else
    scripts/package-sdk-console-assets.sh --sdk node
fi
scripts/verify-sdk-console-assets.sh --sdk node

native_runtime_dir="$(
    scripts/ci-prepare-native-runtime.sh \
        "$REPO_ROOT/target/rust-native-runtime" \
        cpu \
        --reuse-from-binary "$1"
)"
export MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR="$native_runtime_dir"

scripts/ci-sdk-fixture.sh "$1" "$2" "$3" -- \
    cargo test -p mesh-llm-ffi --test live_sdk_smoke -- --nocapture
