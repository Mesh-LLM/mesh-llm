#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Usage: $0 <tag> <MeshLLMFFI.xcframework.zip> [Package.swift]" >&2
    exit 1
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: Swift package manifest verification must run on macOS" >&2
    exit 1
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/lib/automation.sh"
mesh_automation release swift-manifest verify "$1" "$2" \
    "${3:-$REPO_ROOT/Package.swift}" "$(command -v swift)" 30 65536
