#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SWIFT_DIR="$REPO_ROOT/mesh/sdk/swift"
automation=(just --justfile "$REPO_ROOT/Justfile" automation-run)
if [[ "${MESH_LLM_AUTOMATION_BIN+set}" == set ]]; then
  if [[ "$MESH_LLM_AUTOMATION_BIN" != /* || ! -f "$MESH_LLM_AUTOMATION_BIN" || ! -x "$MESH_LLM_AUTOMATION_BIN" ]]; then
    echo 'MESH_LLM_AUTOMATION_BIN must be an absolute executable' >&2
    exit 1
  fi
  automation=("$MESH_LLM_AUTOMATION_BIN")
fi
FFI_DIR="$SWIFT_DIR/Generated/FFI"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
XCFRAMEWORK_DIR="$SWIFT_DIR/Generated"
FRAMEWORK_NAME="MeshLLMFFI"
GENERATED_SWIFT="$SWIFT_DIR/Sources/MeshLLM/Generated/mesh_ffi.swift"

echo "Building host macOS $FRAMEWORK_NAME XCFramework..."
echo "Repo root: $REPO_ROOT"

if ! cargo metadata --no-deps --format-version 1 2>/dev/null | grep -q '"name":"mesh-llm-ffi"'; then
  echo "ERROR: mesh-llm-ffi crate not found. Ensure the workspace is configured."
  exit 1
fi

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
  arm64|aarch64)
    RUST_TARGET="aarch64-apple-darwin"
    CMAKE_ARCH="arm64"
    ;;
  *)
    echo "Unsupported macOS host architecture: $HOST_ARCH; the Swift SDK requires Apple Silicon because Intel Macs cannot run MeshLLM inference." >&2
    exit 1
    ;;
esac

rustup target add "$RUST_TARGET" 2>/dev/null || true

"$SWIFT_DIR/scripts/generate-swift-bindings.sh"

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
export LLAMA_STAGE_BACKEND="${LLAMA_STAGE_BACKEND:-metal}"
export LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-$REPO_ROOT/.deps/llama-build/build-stage-abi-$RUST_TARGET-metal}"

echo "Preparing embedded llama.cpp ABI libraries..."
"$REPO_ROOT/scripts/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"
"$REPO_ROOT/scripts/build-llama.sh" \
  -DCMAKE_OSX_SYSROOT=macosx \
  -DCMAKE_OSX_ARCHITECTURES="$CMAKE_ARCH" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET"

RUSTUP_RUSTC="$(rustup which --toolchain "${RUSTUP_TOOLCHAIN:-stable}" rustc)"
echo "Using rustc: $RUSTUP_RUSTC"
echo "Building for $RUST_TARGET..."
echo "macOS deployment target: $MACOSX_DEPLOYMENT_TARGET"
echo "llama.cpp backend: $LLAMA_STAGE_BACKEND"
echo "llama.cpp build dir: $LLAMA_STAGE_BUILD_DIR"
RUSTC="$RUSTUP_RUSTC" \
  cargo build --release -p mesh-llm-ffi --target "$RUST_TARGET" --no-default-features --features embedded-runtime

LIB_PATH="$TARGET_DIR/$RUST_TARGET/release/libmeshllm_ffi.a"

echo "Syncing UniFFI API checksums into generated Swift bindings..."
"${automation[@]}" prepared-input swift-api-checksum "$LIB_PATH" "$GENERATED_SWIFT"

FRAMEWORK_DIR="$TARGET_DIR/frameworks/macos-host/$FRAMEWORK_NAME.framework"
rm -rf "$FRAMEWORK_DIR"

# macOS requires a versioned bundle layout (Versions/A/); flat bundles are
# rejected by the macOS dynamic linker and cause xcframework load failures.
VERSION_DIR="$FRAMEWORK_DIR/Versions/A"
mkdir -p "$VERSION_DIR/Headers"
mkdir -p "$VERSION_DIR/Modules"
mkdir -p "$VERSION_DIR/Resources"

cp "$LIB_PATH" "$VERSION_DIR/$FRAMEWORK_NAME"
cp "$FFI_DIR/MeshLLMFFI.h" "$VERSION_DIR/Headers/MeshLLMFFI.h"
cp "$FFI_DIR/MeshLLMFFI.modulemap" "$VERSION_DIR/Modules/module.modulemap"

if [ -f "$SWIFT_DIR/PrivacyInfo.xcprivacy" ]; then
  cp "$SWIFT_DIR/PrivacyInfo.xcprivacy" "$VERSION_DIR/Resources/PrivacyInfo.xcprivacy"
  echo "  Embedded PrivacyInfo.xcprivacy in host macOS framework"
else
  echo "WARNING: PrivacyInfo.xcprivacy not found at $SWIFT_DIR/PrivacyInfo.xcprivacy"
fi

cat > "$VERSION_DIR/Resources/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>ai.meshllm.MeshLLMFFI</string>
    <key>CFBundleName</key>
    <string>MeshLLMFFI</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>13.0</string>
</dict>
</plist>
PLIST

ln -sfh A                          "$FRAMEWORK_DIR/Versions/Current"
ln -sfh "Versions/Current/$FRAMEWORK_NAME" "$FRAMEWORK_DIR/$FRAMEWORK_NAME"
ln -sfh "Versions/Current/Headers"         "$FRAMEWORK_DIR/Headers"
ln -sfh "Versions/Current/Modules"         "$FRAMEWORK_DIR/Modules"
ln -sfh "Versions/Current/Resources"       "$FRAMEWORK_DIR/Resources"

echo "Creating host macOS XCFramework..."
XCFW_OUT="$XCFRAMEWORK_DIR/$FRAMEWORK_NAME.xcframework"
rm -rf "$XCFW_OUT"

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "ERROR: xcodebuild is required to create the Swift SDK XCFramework." >&2
  exit 1
fi

xcodebuild -create-xcframework \
  -framework "$FRAMEWORK_DIR" \
  -output "$XCFW_OUT"

echo "XCFramework created at: $XCFW_OUT"

PRIVACY_COUNT=$(find "$XCFW_OUT" -name "PrivacyInfo.xcprivacy" | wc -l | tr -d ' ')
echo "Found $PRIVACY_COUNT PrivacyInfo.xcprivacy file(s) inside XCFramework"
if [ "$PRIVACY_COUNT" -lt 1 ]; then
  echo "ERROR: PrivacyInfo.xcprivacy not embedded in XCFramework!"
  exit 1
fi

echo "Host macOS XCFramework build complete!"
