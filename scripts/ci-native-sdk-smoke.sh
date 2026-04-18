#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

scripts/ci-sdk-fixture.sh "$1" "$2" "$3" -- \
    cargo test -p mesh-api-ffi --test live_sdk_smoke -- --nocapture
