#!/bin/bash
# Sync mesh hook C++ files into the local llama.cpp checkout.
# Run from repo root: ./mesh-llm/llama-patches/sync.sh
#
# Direction: llama-patches/ → llama.cpp/  (default, before building)
#            llama.cpp/ → llama-patches/  (with --save, after editing C++)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCHES="$REPO_ROOT/mesh-llm/llama-patches"
LLAMA="$REPO_ROOT/llama.cpp"

FILES=(
    "tools/server/server-mesh-hook.h"
    "tools/server/server-context.cpp"
    "tools/server/server-task.h"
    "tools/server/server-task.cpp"
    "tools/server/server-common.h"
    "common/common.h"
    "common/arg.cpp"
)

if [[ "${1:-}" == "--save" ]]; then
    echo "Saving llama.cpp → llama-patches/"
    for f in "${FILES[@]}"; do
        cp "$LLAMA/$f" "$PATCHES/$f"
        echo "  $f"
    done
else
    echo "Syncing llama-patches/ → llama.cpp/"
    for f in "${FILES[@]}"; do
        cp "$PATCHES/$f" "$LLAMA/$f"
        echo "  $f"
    done
fi
