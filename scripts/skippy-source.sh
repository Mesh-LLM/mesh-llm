#!/usr/bin/env bash
# MeshLLM builds against the Skippy revision pinned in Cargo.toml and
# Cargo.lock. This script points the build at a local Skippy checkout instead,
# for work that spans both repositories.
#
#   scripts/skippy-source.sh              show what the build uses
#   scripts/skippy-source.sh <checkout>   build against that checkout or worktree
#   scripts/skippy-source.sh --pinned     go back to the pinned revision
#
# The override is .cargo/skippy-local.toml (not committed). While it exists,
# Cargo.lock no longer matches, so build without --locked; build-host.sh does
# that for you. Do not commit the Cargo.lock it produces.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERRIDE="$REPO_ROOT/.cargo/skippy-local.toml"
pin_url="$(sed -n 's/^skippy-ffi = .*git = "\([^"]*\)".*/\1/p' "$REPO_ROOT/Cargo.toml")"
pin_rev="$(sed -n 's/^skippy-ffi = .*rev = "\([^"]*\)".*/\1/p' "$REPO_ROOT/Cargo.toml")"
[[ -n "$pin_url" && -n "$pin_rev" ]] || { echo "no Skippy pin found in Cargo.toml" >&2; exit 1; }

if (($# == 0)); then
    if [[ -f "$OVERRIDE" ]]; then
        dir="$(sed -n 's/^# checkout: //p' "$OVERRIDE")"
        echo "local $dir $(git -C "$dir" rev-parse HEAD)$(git -C "$dir" diff --quiet HEAD -- 2>/dev/null || echo ' (modified)')"
    else
        echo "pinned $pin_url $pin_rev"
    fi
    exit 0
fi

if [[ "$1" == "--pinned" ]]; then
    rm -f "$OVERRIDE"
    git -C "$REPO_ROOT" checkout -- Cargo.lock
    echo "using pinned $pin_rev"
    exit 0
fi

source_dir="$(cd "$1" && pwd -P)"
[[ -f "$source_dir/crates/skippy-ffi/Cargo.toml" ]] || {
    echo "$source_dir is not a Skippy checkout (crates/skippy-ffi/Cargo.toml is missing)" >&2
    exit 1
}
{
    echo "# checkout: $source_dir"
    echo "[patch.\"$pin_url\"]"
    for manifest in "$source_dir"/crates/*/Cargo.toml; do
        name="$(sed -n 's/^name = "\(.*\)"/\1/p' "$manifest" | head -1)"
        echo "$name = { path = \"$(dirname "$manifest")\" }"
    done
} >"$OVERRIDE"
echo "building against $source_dir ($(git -C "$source_dir" rev-parse HEAD))"
