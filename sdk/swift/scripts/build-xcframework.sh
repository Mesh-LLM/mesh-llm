#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWIFT_DIR="$REPO_ROOT/sdk/swift"
FFI_DIR="$SWIFT_DIR/Generated/FFI"
TARGET_DIR="$REPO_ROOT/target"
XCFRAMEWORK_DIR="$SWIFT_DIR/Generated"
FRAMEWORK_NAME="MeshLLMFFI"
GENERATED_SWIFT="$SWIFT_DIR/Sources/MeshLLM/Generated/mesh_ffi.swift"
RUST_FEATURES="host,embedded-runtime"

echo "Building $FRAMEWORK_NAME XCFramework..."
echo "Repo root: $REPO_ROOT"

if ! cargo metadata --no-deps --format-version 1 2>/dev/null | grep -q '"name":"mesh-llm-ffi"'; then
  echo "ERROR: mesh-llm-ffi crate not found. Ensure the workspace is configured."
  exit 1
fi

rustup target add \
  aarch64-apple-ios \
  aarch64-apple-ios-sim \
  x86_64-apple-ios \
  aarch64-apple-ios-macabi \
  x86_64-apple-ios-macabi \
  aarch64-apple-darwin \
  x86_64-apple-darwin \
  2>/dev/null || true

"$SWIFT_DIR/scripts/generate-swift-bindings.sh"
IPHONEOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-16.0}"
MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
export -n IPHONEOS_DEPLOYMENT_TARGET MACOSX_DEPLOYMENT_TARGET 2>/dev/null || true
"$REPO_ROOT/scripts/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

# Resolve stable rustc from rustup (avoids Homebrew rustc shadowing)
RUSTUP_RUSTC="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc"
if [ ! -x "$RUSTUP_RUSTC" ]; then
  # Fallback: find any stable toolchain
  STABLE_TOOLCHAIN=$(rustup toolchain list | grep stable | head -1 | awk '{print $1}')
  RUSTUP_RUSTC="$(rustup run "$STABLE_TOOLCHAIN" which rustc)"
fi
echo "Using rustc: $RUSTUP_RUSTC"

build_llama_for_target() {
  local RUST_TARGET="$1"
  local SDK="$2"
  local ARCH="$3"
  local PLATFORM_NAME="$4"
  shift 4

  local LLAMA_BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-$RUST_TARGET-metal"
  echo "Building llama.cpp ABI for $PLATFORM_NAME ($RUST_TARGET)..."
  LLAMA_STAGE_BACKEND=metal \
  LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
  "$REPO_ROOT/scripts/build-llama.sh" \
    -DCMAKE_OSX_SYSROOT="$SDK" \
    -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IPHONEOS_DEPLOYMENT_TARGET" \
    "$@"
}

build_rust_target() {
  local RUST_TARGET="$1"
  local PLATFORM_NAME="$2"
  local LLAMA_BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-$RUST_TARGET-metal"

  echo "Building for $RUST_TARGET ($PLATFORM_NAME)..."
  local -a CARGO_ENV=(
    "RUSTC=$RUSTUP_RUSTC"
    "LLAMA_STAGE_BACKEND=metal"
    "LLAMA_STAGE_BUILD_DIR=$LLAMA_BUILD_DIR"
  )

  case "$RUST_TARGET" in
    *-apple-darwin)
      CARGO_ENV+=("MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET")
      ;;
    *-apple-ios*)
      CARGO_ENV+=("IPHONEOS_DEPLOYMENT_TARGET=$IPHONEOS_DEPLOYMENT_TARGET")
      ;;
  esac

  env "${CARGO_ENV[@]}" \
    cargo build --release -p mesh-llm-ffi --target "$RUST_TARGET" --no-default-features --features "$RUST_FEATURES"
}

build_apple_target() {
  local RUST_TARGET="$1"
  local SDK="$2"
  local ARCH="$3"
  local PLATFORM_NAME="$4"
  shift 4

  build_llama_for_target "$RUST_TARGET" "$SDK" "$ARCH" "$PLATFORM_NAME" "$@"
  build_rust_target "$RUST_TARGET" "$PLATFORM_NAME"
}

echo "Building for aarch64-apple-ios..."
build_apple_target aarch64-apple-ios iphoneos arm64 iOS -DCMAKE_SYSTEM_NAME=iOS -DGGML_BLAS=OFF

echo "Building for aarch64-apple-ios-sim..."
build_apple_target aarch64-apple-ios-sim iphonesimulator arm64 "iOS simulator" -DCMAKE_SYSTEM_NAME=iOS -DGGML_BLAS=OFF

echo "Building for x86_64-apple-ios..."
build_apple_target x86_64-apple-ios iphonesimulator x86_64 "iOS simulator" -DCMAKE_SYSTEM_NAME=iOS -DGGML_BLAS=OFF

echo "Building for aarch64-apple-ios-macabi (Mac Catalyst)..."
build_apple_target aarch64-apple-ios-macabi macosx arm64 "Mac Catalyst" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DGGML_BLAS=OFF \
  -DCMAKE_C_FLAGS=-target\ arm64-apple-ios16.0-macabi \
  -DCMAKE_CXX_FLAGS=-target\ arm64-apple-ios16.0-macabi \
  -DCMAKE_EXE_LINKER_FLAGS=-target\ arm64-apple-ios16.0-macabi \
  -DCMAKE_SHARED_LINKER_FLAGS=-target\ arm64-apple-ios16.0-macabi \
  -DCMAKE_MODULE_LINKER_FLAGS=-target\ arm64-apple-ios16.0-macabi

echo "Building for x86_64-apple-ios-macabi (Mac Catalyst)..."
build_apple_target x86_64-apple-ios-macabi macosx x86_64 "Mac Catalyst" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DGGML_BLAS=OFF \
  -DCMAKE_C_FLAGS=-target\ x86_64-apple-ios16.0-macabi \
  -DCMAKE_CXX_FLAGS=-target\ x86_64-apple-ios16.0-macabi \
  -DCMAKE_EXE_LINKER_FLAGS=-target\ x86_64-apple-ios16.0-macabi \
  -DCMAKE_SHARED_LINKER_FLAGS=-target\ x86_64-apple-ios16.0-macabi \
  -DCMAKE_MODULE_LINKER_FLAGS=-target\ x86_64-apple-ios16.0-macabi

echo "Building for aarch64-apple-darwin (macOS)..."
build_apple_target aarch64-apple-darwin macosx arm64 macOS -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0

echo "Building for x86_64-apple-darwin (macOS)..."
build_apple_target x86_64-apple-darwin macosx x86_64 macOS -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0

echo "Syncing UniFFI API checksums into generated Swift bindings..."
python3 - "$TARGET_DIR/aarch64-apple-darwin/release/libmeshllm_ffi.a" "$GENERATED_SWIFT" <<'PY'
import pathlib
import re
import subprocess
import sys

lib_path = pathlib.Path(sys.argv[1])
swift_path = pathlib.Path(sys.argv[2])

disassembly = subprocess.run(
    ["otool", "-tvV", str(lib_path)],
    check=True,
    capture_output=True,
    text=True,
).stdout

pattern = re.compile(
    r"_uniffi_meshllm_ffi_(checksum_[A-Za-z0-9_]+):\n[0-9a-f]+\s+mov\s+w0, #0x([0-9a-f]+)\n[0-9a-f]+\s+ret",
    re.MULTILINE,
)
checksums = {name: int(value, 16) for name, value in pattern.findall(disassembly)}

swift = swift_path.read_text()

for name, value in checksums.items():
    call = f"{name}()"
    swift = re.sub(
        rf"({re.escape(call)} != )\d+",
        rf"\g<1>{value}",
        swift,
    )

swift_path.write_text(swift)
PY

echo "Creating fat library for iOS simulator..."
mkdir -p "$TARGET_DIR/ios-sim-fat"
lipo -create \
  "$TARGET_DIR/aarch64-apple-ios-sim/release/libmeshllm_ffi.a" \
  "$TARGET_DIR/x86_64-apple-ios/release/libmeshllm_ffi.a" \
  -output "$TARGET_DIR/ios-sim-fat/libmeshllm_ffi.a"

echo "Creating fat library for macOS..."
mkdir -p "$TARGET_DIR/macos-fat"
lipo -create \
  "$TARGET_DIR/aarch64-apple-darwin/release/libmeshllm_ffi.a" \
  "$TARGET_DIR/x86_64-apple-darwin/release/libmeshllm_ffi.a" \
  -output "$TARGET_DIR/macos-fat/libmeshllm_ffi.a"

echo "Creating fat library for Mac Catalyst..."
mkdir -p "$TARGET_DIR/ios-macabi-fat"
lipo -create \
  "$TARGET_DIR/aarch64-apple-ios-macabi/release/libmeshllm_ffi.a" \
  "$TARGET_DIR/x86_64-apple-ios-macabi/release/libmeshllm_ffi.a" \
  -output "$TARGET_DIR/ios-macabi-fat/libmeshllm_ffi.a"

create_framework() {
  local ARCH="$1"
  local LIB_PATH="$2"
  local FRAMEWORK_DIR="$TARGET_DIR/frameworks/$ARCH/$FRAMEWORK_NAME.framework"

  if [ "$ARCH" = "macos" ]; then
    # macOS requires a versioned bundle layout (Versions/A/); flat bundles are
    # rejected by the macOS dynamic linker and cause xcframework load failures.
    local VERSION_DIR="$FRAMEWORK_DIR/Versions/A"
    mkdir -p "$VERSION_DIR/Headers"
    mkdir -p "$VERSION_DIR/Modules"
    mkdir -p "$VERSION_DIR/Resources"

    cp "$LIB_PATH" "$VERSION_DIR/$FRAMEWORK_NAME"
    cp "$FFI_DIR/MeshLLMFFI.h" "$VERSION_DIR/Headers/MeshLLMFFI.h"
    cp "$FFI_DIR/MeshLLMFFI.modulemap" "$VERSION_DIR/Modules/module.modulemap"

    if [ -f "$SWIFT_DIR/PrivacyInfo.xcprivacy" ]; then
      cp "$SWIFT_DIR/PrivacyInfo.xcprivacy" "$VERSION_DIR/Resources/PrivacyInfo.xcprivacy"
      echo "  Embedded PrivacyInfo.xcprivacy in $ARCH framework"
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
  else
    # iOS, simulator, and Mac Catalyst use flat (non-versioned) framework layout
    mkdir -p "$FRAMEWORK_DIR/Headers"
    mkdir -p "$FRAMEWORK_DIR/Modules"

    cp "$LIB_PATH" "$FRAMEWORK_DIR/$FRAMEWORK_NAME"
    cp "$FFI_DIR/MeshLLMFFI.h" "$FRAMEWORK_DIR/Headers/MeshLLMFFI.h"
    cp "$FFI_DIR/MeshLLMFFI.modulemap" "$FRAMEWORK_DIR/Modules/module.modulemap"

    if [ -f "$SWIFT_DIR/PrivacyInfo.xcprivacy" ]; then
      cp "$SWIFT_DIR/PrivacyInfo.xcprivacy" "$FRAMEWORK_DIR/PrivacyInfo.xcprivacy"
      echo "  Embedded PrivacyInfo.xcprivacy in $ARCH framework"
    else
      echo "WARNING: PrivacyInfo.xcprivacy not found at $SWIFT_DIR/PrivacyInfo.xcprivacy"
    fi

    cat > "$FRAMEWORK_DIR/Info.plist" << 'PLIST'
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
    <string>16.0</string>
</dict>
</plist>
PLIST
  fi

  echo "  Created framework bundle for $ARCH"
}

echo "Assembling framework bundles..."
create_framework "ios"     "$TARGET_DIR/aarch64-apple-ios/release/libmeshllm_ffi.a"
create_framework "ios-sim" "$TARGET_DIR/ios-sim-fat/libmeshllm_ffi.a"
create_framework "ios-macabi" "$TARGET_DIR/ios-macabi-fat/libmeshllm_ffi.a"
create_framework "macos"   "$TARGET_DIR/macos-fat/libmeshllm_ffi.a"

echo "Creating XCFramework..."
rm -rf "$XCFRAMEWORK_DIR/$FRAMEWORK_NAME.xcframework"
mkdir -p "$XCFRAMEWORK_DIR"

XCFW_OUT="$XCFRAMEWORK_DIR/$FRAMEWORK_NAME.xcframework"

if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "ERROR: xcodebuild is required to create the Swift SDK XCFramework." >&2
  exit 1
fi

xcodebuild -create-xcframework \
  -framework "$TARGET_DIR/frameworks/ios/$FRAMEWORK_NAME.framework" \
  -framework "$TARGET_DIR/frameworks/ios-sim/$FRAMEWORK_NAME.framework" \
  -framework "$TARGET_DIR/frameworks/ios-macabi/$FRAMEWORK_NAME.framework" \
  -framework "$TARGET_DIR/frameworks/macos/$FRAMEWORK_NAME.framework" \
  -output "$XCFW_OUT"

echo ""
echo "XCFramework created at: $XCFW_OUT"

echo "Verifying PrivacyInfo.xcprivacy embedding..."
PRIVACY_COUNT=$(find "$XCFW_OUT" -name "PrivacyInfo.xcprivacy" | wc -l | tr -d ' ')
echo "Found $PRIVACY_COUNT PrivacyInfo.xcprivacy file(s) inside XCFramework"
if [ "$PRIVACY_COUNT" -lt 1 ]; then
  echo "ERROR: PrivacyInfo.xcprivacy not embedded in XCFramework!"
  exit 1
fi

echo ""
echo "Build complete!"
