#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

scripts/check-sdk-contract.sh

if [[ "${MESH_SWIFT_FULL_XCFRAMEWORK_SMOKE:-0}" == "1" ]]; then
    ./sdk/swift/scripts/build-xcframework.sh
else
    ./sdk/swift/scripts/build-host-macos-xcframework.sh
fi

scripts/verify-swift-privacy-manifest.sh \
    sdk/swift/PrivacyInfo.xcprivacy \
    sdk/swift/Generated/MeshLLMFFI.xcframework

SWIFT_ARTIFACT_SMOKE_DIR="$(mktemp -d)"
trap 'rm -rf "$SWIFT_ARTIFACT_SMOKE_DIR"' EXIT
ditto -c -k --sequesterRsrc --keepParent \
    sdk/swift/Generated/MeshLLMFFI.xcframework \
    "$SWIFT_ARTIFACT_SMOKE_DIR/MeshLLMFFI.xcframework.zip"
scripts/verify-swift-release-artifact.sh \
    "$SWIFT_ARTIFACT_SMOKE_DIR/MeshLLMFFI.xcframework.zip"

scripts/ci-sdk-fixture.sh "$1" "$2" "$3" -- \
    bash -lc '
        set -euo pipefail
        cd '"$REPO_ROOT"'
        swift run \
            --package-path sdk/swift/example/MeshExampleApp \
            MeshExampleApp \
            "$MESH_SDK_INVITE_TOKEN"
    '
