#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/verify-swift-release-artifact.sh <MeshLLMFFI.xcframework.zip>

Verifies the SwiftPM release artifact shape by checking the zipped XCFramework,
its embedded privacy manifests, and a temporary Swift package consumer that
depends on the zipped binary target.
EOF
}

if [[ "$#" -ne 1 ]]; then
  usage
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: Swift release artifact verification must run on macOS" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ZIP="$1"

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
ditto -x -k "$ARTIFACT_ZIP" "$EXTRACT_DIR"

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

python3 - "$XCFRAMEWORK_PATH" <<'PY'
import os
import plistlib
import sys

xcframework = sys.argv[1]
with open(os.path.join(xcframework, "Info.plist"), "rb") as fh:
    info = plistlib.load(fh)

macos_frameworks = []
for library in info.get("AvailableLibraries", []):
    if library.get("SupportedPlatform") != "macos":
        continue
    library_path = library.get("LibraryPath")
    identifier = library.get("LibraryIdentifier")
    if not library_path or not identifier:
        raise SystemExit(f"invalid macOS library entry: {library!r}")
    macos_frameworks.append(os.path.join(xcframework, identifier, library_path))

if not macos_frameworks:
    raise SystemExit("XCFramework does not contain a macOS framework slice")

for framework in macos_frameworks:
    name = os.path.splitext(os.path.basename(framework))[0]
    expected = {
        "Versions/Current": "A",
        name: f"Versions/Current/{name}",
        "Headers": "Versions/Current/Headers",
        "Modules": "Versions/Current/Modules",
        "Resources": "Versions/Current/Resources",
    }
    for relative, target in expected.items():
        path = os.path.join(framework, relative)
        if not os.path.islink(path):
            raise SystemExit(f"macOS framework is not versioned; missing symlink: {path}")
        actual = os.readlink(path)
        if actual != target:
            raise SystemExit(f"unexpected symlink target for {path}: {actual!r} != {target!r}")

    required_paths = [
        os.path.join(framework, "Versions", "A", name),
        os.path.join(framework, "Versions", "A", "Headers"),
        os.path.join(framework, "Versions", "A", "Modules", "module.modulemap"),
        os.path.join(framework, "Versions", "A", "Resources", "Info.plist"),
        os.path.join(framework, "Versions", "A", "Resources", "PrivacyInfo.xcprivacy"),
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise SystemExit(f"macOS framework versioned layout is incomplete: {path}")

print(f"verified {len(macos_frameworks)} versioned macOS framework slice(s)")
PY

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
