#!/usr/bin/env bash
# Verify that Android .so LOAD segments have 16KB (2**14) page alignment.
# Required for Android 15+ and Google Play (mandatory from August 2025).
#
# Usage:
#   ./verify-alignment.sh [path/to/libmesh_ffi.so]
#
# Defaults to arm64-v8a if no path is given.
set -euo pipefail

SO_PATH="${1:-sdk/kotlin/src/main/jniLibs/arm64-v8a/libmesh_ffi.so}"

if [ ! -f "$SO_PATH" ]; then
    echo "ERROR: .so not found at: $SO_PATH"
    echo "Build the Android library first: cargo build --target aarch64-linux-android --release"
    exit 1
fi

# Prefer llvm-objdump from NDK if available (cross-platform, accurate output)
OBJDUMP=""
if command -v llvm-objdump &>/dev/null; then
    OBJDUMP=llvm-objdump
elif [ -n "${ANDROID_NDK_HOME:-}" ]; then
    NDK_OBJDUMP=$(find "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt" -name "llvm-objdump" 2>/dev/null | head -1 || true)
    if [ -n "$NDK_OBJDUMP" ]; then
        OBJDUMP="$NDK_OBJDUMP"
    fi
fi

if [ -z "$OBJDUMP" ]; then
    echo "WARNING: llvm-objdump not found; falling back to readelf"
    if ! command -v readelf &>/dev/null; then
        echo "ERROR: neither llvm-objdump nor readelf found"
        echo "Install binutils or set ANDROID_NDK_HOME"
        exit 1
    fi
    echo "Checking LOAD alignment via readelf for: $SO_PATH"
    readelf -l "$SO_PATH" | grep -E 'LOAD|Align'
    echo ""
    echo "Verify that each LOAD segment shows Align = 0x4000 (16384 = 16KB)"
    exit 0
fi

echo "Using: $OBJDUMP"
echo "Checking LOAD alignment for: $SO_PATH"
echo ""

LOAD_LINES=$("$OBJDUMP" -p "$SO_PATH" | grep "^    LOAD")

if [ -z "$LOAD_LINES" ]; then
    echo "ERROR: no LOAD segments found in $SO_PATH"
    exit 1
fi

FAIL=0
while IFS= read -r line; do
    ALIGN=$(echo "$line" | awk '{print $9}')
    if [ "$ALIGN" = "2**14" ]; then
        echo "  PASS: LOAD align=$ALIGN (16KB) — $line"
    else
        echo "  FAIL: LOAD align=$ALIGN (expected 2**14) — $line"
        FAIL=1
    fi
done <<< "$LOAD_LINES"

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "RESULT: PASS — all LOAD segments have 2**14 (16KB) alignment ✓"
    echo "This .so is compatible with Android 15+ 16KB page size requirements."
else
    echo "RESULT: FAIL — one or more LOAD segments do not have 16KB alignment"
    echo "Ensure rustflags include: -C link-arg=-Wl,-z,max-page-size=16384"
    exit 1
fi
