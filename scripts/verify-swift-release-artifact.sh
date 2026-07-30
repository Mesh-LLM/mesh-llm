#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/verify-swift-release-artifact.sh <MeshLLMFFI.xcframework.zip> [host-only|full]

Verifies the SwiftPM release artifact shape by checking the zipped XCFramework,
its embedded privacy manifests, and a temporary Swift package consumer that
depends on the zipped binary target. When a mode is supplied, also verifies the
expected platform/slice contract.
EOF
}

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  usage
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: Swift release artifact verification must run on macOS" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ZIP="$1"
EXPECTED_MODE="${2:-}"

case "$EXPECTED_MODE" in
  ""|host-only|full) ;;
  *)
    echo "unsupported Swift SDK artifact mode: $EXPECTED_MODE" >&2
    exit 1
    ;;
esac

if [[ ! -f "$ARTIFACT_ZIP" ]]; then
  echo "Swift release artifact does not exist: $ARTIFACT_ZIP" >&2
  exit 1
fi

GENERATED_SWIFT_RELATIVE="sdk/swift/Sources/MeshLLM/Generated/mesh_ffi.swift"
GENERATED_SWIFT="$REPO_ROOT/$GENERATED_SWIFT_RELATIVE"

if [[ ! -f "$GENERATED_SWIFT" ]]; then
  echo "generated Swift UniFFI bindings are missing; run sdk/swift/scripts/build-xcframework.sh first" >&2
  exit 1
fi

if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git -C "$REPO_ROOT" ls-files --error-unmatch "$GENERATED_SWIFT_RELATIVE" >/dev/null 2>&1; then
    echo "generated Swift UniFFI bindings must be tracked for tagged SwiftPM consumers: $GENERATED_SWIFT_RELATIVE" >&2
    exit 1
  fi
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

EXTRACT_DIR="$TMP_ROOT/extract"
mkdir -p "$EXTRACT_DIR"
python3 "$REPO_ROOT/scripts/safe-extract-zip.py" \
  "$ARTIFACT_ZIP" \
  "$EXTRACT_DIR"

XCFRAMEWORK_COUNT="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -name '*.xcframework' -type d | wc -l | tr -d ' ')"
if [[ "$XCFRAMEWORK_COUNT" != "1" ]]; then
  echo "expected exactly one top-level XCFramework in $ARTIFACT_ZIP, found $XCFRAMEWORK_COUNT" >&2
  exit 1
fi

XCFRAMEWORK_PATH="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -name '*.xcframework' -type d -print -quit)"
if [[ "$(basename "$XCFRAMEWORK_PATH")" != "MeshLLMFFI.xcframework" ]]; then
  echo "unexpected XCFramework name: $(basename "$XCFRAMEWORK_PATH")" >&2
  exit 1
fi

plutil -lint "$XCFRAMEWORK_PATH/Info.plist" >/dev/null
"$REPO_ROOT/scripts/verify-swift-privacy-manifest.sh" \
  "$REPO_ROOT/sdk/swift/PrivacyInfo.xcprivacy" \
  "$XCFRAMEWORK_PATH"

xcframework_args=("$XCFRAMEWORK_PATH")
if [[ -n "$EXPECTED_MODE" ]]; then
  xcframework_args+=("$EXPECTED_MODE")
fi
python3 "$REPO_ROOT/scripts/verify-swift-xcframework.py" \
  "${xcframework_args[@]}"

CONSUMER_DIR="$TMP_ROOT/consumer"
mkdir -p "$CONSUMER_DIR/Sources" "$CONSUMER_DIR/Sources/Consumer"
cp "$ARTIFACT_ZIP" "$CONSUMER_DIR/MeshLLMFFI.xcframework.zip"
ln -s "$REPO_ROOT/sdk/swift/Sources/MeshLLM" "$CONSUMER_DIR/Sources/MeshLLM"

cat > "$CONSUMER_DIR/Package.swift" <<'EOF'
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MeshLLMReleaseArtifactConsumer",
    platforms: [
        .macOS(.v13),
    ],
    targets: [
        .binaryTarget(
            name: "MeshLLMFFI",
            path: "MeshLLMFFI.xcframework.zip"
        ),
        .target(
            name: "MeshLLM",
            dependencies: ["MeshLLMFFI"],
            path: "Sources/MeshLLM",
            resources: [
                .copy("Resources/Console"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("AppKit", .when(platforms: [.macOS])),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("SystemConfiguration"),
                .linkedLibrary("c++"),
            ]
        ),
        .executableTarget(
            name: "Consumer",
            dependencies: ["MeshLLM"],
            path: "Sources/Consumer",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("AppKit"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("SystemConfiguration"),
                .linkedLibrary("c++"),
            ]
        ),
    ]
)
EOF

cat > "$CONSUMER_DIR/Sources/Consumer/main.swift" <<'EOF'
import MeshLLM

let token = InviteToken("release-artifact-smoke")
let runtimeOptions = NativeRuntimeResolveOptions()
let ownerKeypair = generateOwnerKeypairHex()
precondition(!ownerKeypair.isEmpty)
print("consumer-ok \(token.value) \(runtimeOptions.searchDirectories.count) \(ownerKeypair.prefix(8))")
EOF

swift build --package-path "$CONSUMER_DIR"
swift run --package-path "$CONSUMER_DIR" Consumer

echo "verified Swift release artifact: $ARTIFACT_ZIP"
