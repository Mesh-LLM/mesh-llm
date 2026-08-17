#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> <xcframework-zip> <host-only|full> <generated-mesh_ffi.swift>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SWIFT_INPUT_ARCHIVE="$4"
SWIFT_INPUT_MODE="$5"
SWIFT_INPUT_BINDING="$6"
SWIFT_TRACKED_BINDING="sdk/swift/Sources/MeshLLM/Generated/mesh_ffi.swift"

if [[ ! -f "$SWIFT_INPUT_BINDING" || -L "$SWIFT_INPUT_BINDING" ]]; then
    echo "immutable generated Swift binding is missing or unsafe: $SWIFT_INPUT_BINDING" >&2
    exit 1
fi
install -m 0644 "$SWIFT_INPUT_BINDING" "$SWIFT_TRACKED_BINDING"
cmp "$SWIFT_INPUT_BINDING" "$SWIFT_TRACKED_BINDING"

scripts/check-sdk-contract.sh
scripts/package-sdk-console-assets.sh --sdk swift
scripts/verify-sdk-console-assets.sh --sdk swift

scripts/verify-swift-release-artifact.sh \
    "$SWIFT_INPUT_ARCHIVE" \
    "$SWIFT_INPUT_MODE"

SWIFT_EXTRACT_DIR="$(mktemp -d)"
trap 'rm -rf "$SWIFT_EXTRACT_DIR"' EXIT
scripts/safe-extract-zip.py "$SWIFT_INPUT_ARCHIVE" "$SWIFT_EXTRACT_DIR"

SWIFT_GENERATED_DIR="sdk/swift/Generated"
if [[ -L "$SWIFT_GENERATED_DIR" ]] \
    || [[ -e "$SWIFT_GENERATED_DIR" && ! -d "$SWIFT_GENERATED_DIR" ]]; then
    echo "Swift generated artifact directory is unsafe: $SWIFT_GENERATED_DIR" >&2
    exit 1
fi
mkdir -p "$SWIFT_GENERATED_DIR"
SWIFT_XCFRAMEWORK="$SWIFT_GENERATED_DIR/MeshLLMFFI.xcframework"
if [[ ! -d "$SWIFT_EXTRACT_DIR/MeshLLMFFI.xcframework" ]]; then
    echo "verified Swift SDK input did not restore MeshLLMFFI.xcframework" >&2
    exit 1
fi
rm -rf "$SWIFT_XCFRAMEWORK"
mv "$SWIFT_EXTRACT_DIR/MeshLLMFFI.xcframework" "$SWIFT_XCFRAMEWORK"

scripts/verify-swift-privacy-manifest.sh \
    sdk/swift/PrivacyInfo.xcprivacy \
    "$SWIFT_XCFRAMEWORK"

native_runtime_dir="$(
    scripts/ci-prepare-native-runtime.sh \
        "$REPO_ROOT/target/swift-native-runtime" \
        cpu \
        --reuse-from-binary "$1"
)"
export MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR="$native_runtime_dir"

# shellcheck disable=SC2016 # The nested shell expands exported fixture variables.
scripts/ci-sdk-fixture.sh "$1" "$2" "$3" -- \
    bash -lc '
        set -euo pipefail
        cd '"$REPO_ROOT"'
        swift run \
            --package-path sdk/swift/example/MeshExampleApp \
            MeshExampleApp \
            "$MESH_SDK_INVITE_TOKEN"
    '
