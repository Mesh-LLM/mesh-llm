#!/usr/bin/env bash
# Check Package.swift's remote MeshLLMFFI binary target matches a release artifact.
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Usage: $0 <tag> <MeshLLMFFI.xcframework.zip> [Package.swift]" >&2
    exit 1
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="$1"
ARTIFACT="$2"
MANIFEST="${3:-$REPO_ROOT/Package.swift}"
EXPECTED_URL="https://github.com/Mesh-LLM/mesh-llm/releases/download/$TAG/MeshLLMFFI.xcframework.zip"
EXPECTED_CHECKSUM="$(swift package compute-checksum "$ARTIFACT")"
field() { perl -ne 'print "$1\n" if /^let '"$1"' = "([^"]*)"/' "$MANIFEST"; }
URL="$(field remoteFFIXCFrameworkURL)"
CHECKSUM="$(field remoteFFIXCFrameworkChecksum)"
if [[ "$URL" != "$EXPECTED_URL" || "$CHECKSUM" != "$EXPECTED_CHECKSUM" ]]; then
    echo "Package.swift does not match $TAG artifact" >&2
    echo "  url:      $URL (expected $EXPECTED_URL)" >&2
    echo "  checksum: $CHECKSUM (expected $EXPECTED_CHECKSUM)" >&2
    exit 1
fi
echo "Package.swift matches $TAG ($CHECKSUM)"
