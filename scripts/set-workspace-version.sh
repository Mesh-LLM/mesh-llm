#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: scripts/set-workspace-version.sh <version>" >&2
    exit 1
fi

version="$1"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z][0-9A-Za-z.-]*)?$ ]]; then
    echo "invalid version: $version" >&2
    echo "expected semver like 0.53.1 or 0.53.1-deadbeef-123" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

files=(
    "$REPO_ROOT/mesh-llm/Cargo.toml"
    "$REPO_ROOT/mesh-llm/plugin/Cargo.toml"
    "$REPO_ROOT/mesh-llm/src/plugins/example/Cargo.toml"
)

for manifest in "${files[@]}"; do
    perl -0pi -e 's/^version = "[^"]+"/version = "'"$version"'"/m' "$manifest"
done

if command -v cargo >/dev/null 2>&1; then
    echo "Refreshing Cargo.lock workspace package versions..."
    (cd "$REPO_ROOT" && cargo metadata --format-version 1 >/dev/null)
else
    echo "cargo not available yet; skipping Cargo.lock refresh"
fi

echo "Updated workspace version to $version:"
for file in "${files[@]}" "$REPO_ROOT/Cargo.lock"; do
    echo "  $file"
done
