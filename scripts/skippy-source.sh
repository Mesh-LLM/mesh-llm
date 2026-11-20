#!/usr/bin/env bash
# MeshLLM consumes Skippy as an external project through `.deps/skippy`.
#
#   scripts/skippy-source.sh              show the linked checkout and its revision
#   scripts/skippy-source.sh <checkout>   link an existing Skippy checkout or worktree
#
# Any branch of Skippy can be built against by linking a worktree of it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINK="$REPO_ROOT/.deps/skippy"

if (($# == 0)); then
    if [[ ! -e "$LINK/crates/skippy-ffi/Cargo.toml" ]]; then
        echo "no Skippy checkout linked; run: scripts/skippy-source.sh <path-to-skippy-checkout>" >&2
        exit 1
    fi
    target="$(cd "$LINK" && pwd -P)"
    echo "$target $(git -C "$target" rev-parse HEAD)$(git -C "$target" diff --quiet HEAD -- 2>/dev/null || echo ' (modified)')"
    exit 0
fi

source_dir="$(cd "$1" && pwd -P)"
if [[ ! -f "$source_dir/crates/skippy-ffi/Cargo.toml" ]]; then
    echo "$source_dir is not a Skippy checkout (crates/skippy-ffi/Cargo.toml is missing)" >&2
    exit 1
fi
mkdir -p "$REPO_ROOT/.deps"
if [[ -e "$LINK" && ! -L "$LINK" ]]; then
    echo "$LINK exists and is not a link; remove it first" >&2
    exit 1
fi
ln -sfn "$source_dir" "$LINK"
echo "linked $LINK -> $source_dir"
