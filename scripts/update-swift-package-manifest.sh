#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/update-swift-package-manifest.sh <tag> <MeshLLMFFI.xcframework.zip> [Package.swift]

Updates Package.swift so its remote SwiftPM binary target points at the release
XCFramework URL and checksum for the supplied artifact.
EOF
}

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    usage
    exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: Swift package manifest updates must run on macOS" >&2
    exit 1
fi

TAG="$1"
ARTIFACT_PATH="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_SWIFT="${3:-$REPO_ROOT/Package.swift}"
ARTIFACT_NAME="MeshLLMFFI.xcframework.zip"
ARTIFACT_URL="https://github.com/Mesh-LLM/mesh-llm/releases/download/$TAG/$ARTIFACT_NAME"

if [[ ! -f "$ARTIFACT_PATH" ]]; then
    echo "Swift release artifact does not exist: $ARTIFACT_PATH" >&2
    exit 1
fi

if [[ ! -f "$PACKAGE_SWIFT" ]]; then
    echo "Package.swift does not exist: $PACKAGE_SWIFT" >&2
    exit 1
fi

CHECKSUM="$(swift package compute-checksum "$ARTIFACT_PATH")"

perl -0pi -e 's#let remoteFFIXCFrameworkURL = ".*"#let remoteFFIXCFrameworkURL = "'"$ARTIFACT_URL"'"#' "$PACKAGE_SWIFT"
perl -0pi -e 's#let remoteFFIXCFrameworkChecksum = ".*"#let remoteFFIXCFrameworkChecksum = "'"$CHECKSUM"'"#' "$PACKAGE_SWIFT"

echo "updated Swift package manifest for $TAG"
echo "  url: $ARTIFACT_URL"
echo "  checksum: $CHECKSUM"
