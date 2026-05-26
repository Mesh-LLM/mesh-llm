#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: scripts/prepare-swift-package-release.sh <tag>" >&2
    exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: Swift package release artifacts must be prepared on macOS" >&2
    exit 1
fi

TAG="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_SWIFT="$REPO_ROOT/Package.swift"
ARTIFACT_DIR="$REPO_ROOT/dist"
ARTIFACT_NAME="MeshLLMFFI.xcframework.zip"
ARTIFACT_PATH="$ARTIFACT_DIR/$ARTIFACT_NAME"

mkdir -p "$ARTIFACT_DIR"

"$REPO_ROOT/sdk/swift/scripts/build-xcframework.sh"

rm -f "$ARTIFACT_PATH"
ditto -c -k --sequesterRsrc --keepParent \
    "$REPO_ROOT/sdk/swift/Generated/MeshLLMFFI.xcframework" \
    "$ARTIFACT_PATH"

"$REPO_ROOT/scripts/verify-swift-release-artifact.sh" "$ARTIFACT_PATH"
"$REPO_ROOT/scripts/update-swift-package-manifest.sh" "$TAG" "$ARTIFACT_PATH" "$PACKAGE_SWIFT"
"$REPO_ROOT/scripts/verify-swift-package-manifest.sh" "$TAG" "$ARTIFACT_PATH" "$PACKAGE_SWIFT"

echo "Prepared SwiftPM artifact:"
echo "  tag: $TAG"
echo "  path: $ARTIFACT_PATH"
