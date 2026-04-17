#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_SWIFT="$REPO_ROOT/Package.swift"

perl -0pi -e 's#let remoteFFIXCFrameworkURL = ".*"#let remoteFFIXCFrameworkURL = "https://github.com/Mesh-LLM/mesh-llm/releases/download/__MESH_SWIFT_RELEASE_TAG__/MeshLLMFFI.xcframework.zip"#' "$PACKAGE_SWIFT"
perl -0pi -e 's#let remoteFFIXCFrameworkChecksum = ".*"#let remoteFFIXCFrameworkChecksum = "__MESH_SWIFT_RELEASE_CHECKSUM__"#' "$PACKAGE_SWIFT"
