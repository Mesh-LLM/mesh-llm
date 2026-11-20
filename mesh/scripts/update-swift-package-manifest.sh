#!/usr/bin/env bash
# Point Package.swift's remote MeshLLMFFI binary target at a release artifact.
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Usage: $0 <tag> <MeshLLMFFI.xcframework.zip> [Package.swift]" >&2
    exit 1
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="$1"
ARTIFACT="$2"
MANIFEST="${3:-$REPO_ROOT/Package.swift}"
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]] || { echo "invalid tag: $TAG" >&2; exit 1; }
[[ -s "$ARTIFACT" ]] || { echo "missing artifact: $ARTIFACT" >&2; exit 1; }
CHECKSUM="$(swift package compute-checksum "$ARTIFACT")"
[[ "$CHECKSUM" =~ ^[0-9a-f]{64}$ ]] || { echo "unexpected checksum: $CHECKSUM" >&2; exit 1; }
URL="https://github.com/Mesh-LLM/mesh-llm/releases/download/$TAG/MeshLLMFFI.xcframework.zip"
perl -0pi -e 's#let remoteFFIXCFrameworkURL = "[^"]*"#let remoteFFIXCFrameworkURL = "'"$URL"'"#; s#let remoteFFIXCFrameworkChecksum = "[^"]*"#let remoteFFIXCFrameworkChecksum = "'"$CHECKSUM"'"#' "$MANIFEST"
"$REPO_ROOT/mesh/scripts/verify-swift-package-manifest.sh" "$TAG" "$ARTIFACT" "$MANIFEST"
printf 'updated Swift package manifest for %s\n  url: %s\n  checksum: %s\n' "$TAG" "$URL" "$CHECKSUM"
