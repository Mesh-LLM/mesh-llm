#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_PATH="$APPLE_ROOT"
OUTPUT_ROOT="$REPO_ROOT/target/apple-runtime/package"
BUNDLE_ID="meshllm-apple-runtime-darwin-arm64"
BUNDLE_DIR="$OUTPUT_ROOT/$BUNDLE_ID"
IDENTITY="${MESH_APPLE_RUNTIME_CODESIGN_IDENTITY:--}"
ENTITLEMENTS="${MESH_APPLE_RUNTIME_ENTITLEMENTS:-}"
ENTITLEMENT_PROVISIONING_VALIDATED="${MESH_APPLE_RUNTIME_ENTITLEMENT_PROVISIONING_VALIDATED:-0}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "Apple runtime packaging requires Apple silicon macOS" >&2
    exit 2
fi

if [[ -n "$ENTITLEMENTS" ]]; then
    [[ -f "$ENTITLEMENTS" ]] || {
        echo "entitlements file does not exist: $ENTITLEMENTS" >&2
        exit 2
    }
    if [[ "$ENTITLEMENT_PROVISIONING_VALIDATED" != "1" ]]; then
        echo "refusing entitlement-bearing package without validated Apple provisioning" >&2
        echo "set MESH_APPLE_RUNTIME_ENTITLEMENT_PROVISIONING_VALIDATED=1 only after the signing profile grants it" >&2
        exit 2
    fi
fi

swift build -c release --package-path "$PACKAGE_PATH"
BIN_DIR="$(swift build -c release --show-bin-path --package-path "$PACKAGE_PATH")"
SOURCE_BINARY="$BIN_DIR/mesh-apple-runtime"
[[ -x "$SOURCE_BINARY" ]] || {
    echo "missing Apple runtime executable: $SOURCE_BINARY" >&2
    exit 2
}

rm -rf "$OUTPUT_ROOT"
mkdir -p "$BUNDLE_DIR/bin" "$BUNDLE_DIR/Resources"
cp "$SOURCE_BINARY" "$BUNDLE_DIR/bin/mesh-apple-runtime"
cp "$PACKAGE_PATH/README.md" "$BUNDLE_DIR/README.md"
cp "$PACKAGE_PATH/Packaging/Entitlements/background-inference.entitlements" \
    "$BUNDLE_DIR/Resources/background-inference.entitlements"

codesign_args=(--force --sign "$IDENTITY")
if [[ "$IDENTITY" != "-" ]]; then
    codesign_args+=(--options runtime --timestamp)
fi
if [[ -n "$ENTITLEMENTS" ]]; then
    codesign_args+=(--entitlements "$ENTITLEMENTS")
fi
codesign "${codesign_args[@]}" "$BUNDLE_DIR/bin/mesh-apple-runtime"
codesign --verify --strict --verbose=2 "$BUNDLE_DIR/bin/mesh-apple-runtime"

BINARY_SHA="$(shasum -a 256 "$BUNDLE_DIR/bin/mesh-apple-runtime" | awk '{print $1}')"
OS_VERSION="$(sw_vers -productVersion)"
OS_BUILD="$(sw_vers -buildVersion)"
XCODE_VERSION="$(xcodebuild -version | awk 'NR == 1 { print $2 }')"
SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
SIGNING_IDENTITY="$(codesign -dv --verbose=2 "$BUNDLE_DIR/bin/mesh-apple-runtime" 2>&1 | awk -F= '/^Authority=/ && !found {print $2; found=1}')"
if [[ -z "$SIGNING_IDENTITY" ]]; then
    SIGNING_IDENTITY="ad-hoc"
fi

python3 - \
    "$BUNDLE_DIR/manifest.json" \
    "$BUNDLE_ID" \
    "$BINARY_SHA" \
    "$OS_VERSION" \
    "$OS_BUILD" \
    "$XCODE_VERSION" \
    "$SDK_VERSION" \
    "$SIGNING_IDENTITY" \
    "$ENTITLEMENTS" <<'PY'
import json
import sys

(
    output,
    bundle_id,
    binary_sha,
    os_version,
    os_build,
    xcode_version,
    sdk_version,
    signing_identity,
    entitlements,
) = sys.argv[1:]

manifest = {
    "runtime": {
        "id": bundle_id,
        "kind": "apple",
        "runtime_protocol": "0.1",
        "platform": {
            "os": "macos",
            "arch": "arm64",
            "minimum_os": "27.0",
        },
        "hosting_modes": ["process"],
        "entrypoint": "bin/mesh-apple-runtime",
        "model_ids": ["apple/system"],
        "features": [
            "availability",
            "streaming",
            "cancellation",
            "usage",
            "guided_generation",
            "tool_calling",
            "loopback_rest",
        ],
        "files": {
            "bin/mesh-apple-runtime": f"sha256:{binary_sha}",
        },
        "build": {
            "macos": os_version,
            "macos_build": os_build,
            "xcode": xcode_version,
            "sdk": sdk_version,
        },
        "signing": {
            "identity": signing_identity,
            "entitlements_source": entitlements or None,
        },
    }
}

with open(output, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

ARCHIVE="$OUTPUT_ROOT/$BUNDLE_ID.zip"
ditto -c -k --keepParent "$BUNDLE_DIR" "$ARCHIVE"
(cd "$OUTPUT_ROOT" && shasum -a 256 "$BUNDLE_ID.zip" > "$BUNDLE_ID.zip.sha256")

echo "packaged experimental Apple runtime:"
echo "  bundle: $BUNDLE_DIR"
echo "  archive: $ARCHIVE"
echo "  signing: $SIGNING_IDENTITY"
