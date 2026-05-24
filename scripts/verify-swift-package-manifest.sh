#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/verify-swift-package-manifest.sh <tag> <MeshLLMFFI.xcframework.zip> [Package.swift]

Verifies that the Swift package manifest points at the release XCFramework URL
and checksum for the supplied artifact. This must pass before cutting or
publishing a tag, because SwiftPM consumers resolve Package.swift from the tag.
EOF
}

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    usage
    exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: Swift package manifest verification must run on macOS" >&2
    exit 1
fi

TAG="$1"
ARTIFACT_PATH="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_SWIFT="${3:-$REPO_ROOT/Package.swift}"
ARTIFACT_NAME="MeshLLMFFI.xcframework.zip"
EXPECTED_URL="https://github.com/Mesh-LLM/mesh-llm/releases/download/$TAG/$ARTIFACT_NAME"

if [[ ! -f "$ARTIFACT_PATH" ]]; then
    echo "Swift release artifact does not exist: $ARTIFACT_PATH" >&2
    exit 1
fi

if [[ ! -f "$PACKAGE_SWIFT" ]]; then
    echo "Package.swift does not exist: $PACKAGE_SWIFT" >&2
    exit 1
fi

EXPECTED_CHECKSUM="$(swift package compute-checksum "$ARTIFACT_PATH")"

python3 - "$PACKAGE_SWIFT" "$EXPECTED_URL" "$EXPECTED_CHECKSUM" <<'PY'
import re
import sys

manifest_path, expected_url, expected_checksum = sys.argv[1:4]
manifest = open(manifest_path, encoding="utf-8").read()

def swift_string(name: str) -> str:
    match = re.search(rf'let\s+{re.escape(name)}\s*=\s*"([^"]*)"', manifest)
    if not match:
        raise SystemExit(f"missing {name} in {manifest_path}")
    return match.group(1)

actual_url = swift_string("remoteFFIXCFrameworkURL")
actual_checksum = swift_string("remoteFFIXCFrameworkChecksum")

if "__MESH_SWIFT_RELEASE_TAG__" in actual_url:
    raise SystemExit("Package.swift still contains the Swift release URL placeholder")
if "__MESH_SWIFT_RELEASE_CHECKSUM__" in actual_checksum:
    raise SystemExit("Package.swift still contains the Swift release checksum placeholder")
if actual_url != expected_url:
    raise SystemExit(f"Swift release URL mismatch: {actual_url} != {expected_url}")
if actual_checksum != expected_checksum:
    raise SystemExit(
        f"Swift release checksum mismatch: {actual_checksum} != {expected_checksum}"
    )
PY

echo "verified Swift package manifest for $TAG"
