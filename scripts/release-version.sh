#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: scripts/release-version.sh <version|vversion>" >&2
    exit 1
fi

raw_version="$1"
version="${raw_version#v}"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "invalid version: $raw_version" >&2
    echo "expected semantic version like 0.49.0 or v0.49.0" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/set-workspace-version.sh" "$version"

echo "Updated release version to $version."
