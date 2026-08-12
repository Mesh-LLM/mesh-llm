#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_PATH="$APPLE_ROOT"
OUTPUT_ROOT="${MESH_APPLE_RUNTIME_OUTPUT_ROOT:-$REPO_ROOT/target/apple-runtime/package}"
BUNDLE_ID="meshllm-apple-runtime-darwin-arm64"
BUNDLE_DIR="$OUTPUT_ROOT/$BUNDLE_ID"
IDENTITY="${MESH_APPLE_RUNTIME_CODESIGN_IDENTITY:--}"
ENTITLEMENTS="${MESH_APPLE_RUNTIME_ENTITLEMENTS:-}"
ENTITLEMENT_PROVISIONING_VALIDATED="${MESH_APPLE_RUNTIME_ENTITLEMENT_PROVISIONING_VALIDATED:-0}"
RUNTIME_VERSION="${MESH_APPLE_RUNTIME_VERSION:-0.1.0}"
ARCHIVE_URL="${MESH_APPLE_RUNTIME_ARCHIVE_URL:-}"
RELEASE_MODE="${MESH_APPLE_RUNTIME_RELEASE:-0}"
NOTARY_PROFILE="${MESH_APPLE_RUNTIME_NOTARY_PROFILE:-}"
COREAI_MODEL_ROOT="${MESH_APPLE_COREAI_MODEL_ROOT:-}"
COREAI_MODEL_REF="${MESH_APPLE_COREAI_MODEL_REF:-}"
COREAI_MODEL_ID="${MESH_APPLE_COREAI_MODEL_ID:-}"
COREAI_MODEL_VERSION="${MESH_APPLE_COREAI_MODEL_VERSION:-}"
COREAI_CONTEXT_SIZE="${MESH_APPLE_COREAI_CONTEXT_SIZE:-4096}"
COREAI_LANGUAGES="${MESH_APPLE_COREAI_LANGUAGES:-en}"

if [[ -n "$COREAI_MODEL_ROOT" || -n "$COREAI_MODEL_REF" || -n "$COREAI_MODEL_ID" || -n "$COREAI_MODEL_VERSION" ]]; then
    [[ -n "$COREAI_MODEL_ROOT" || -n "$COREAI_MODEL_REF" ]] || {
        echo "configure MESH_APPLE_COREAI_MODEL_ROOT or MESH_APPLE_COREAI_MODEL_REF" >&2
        exit 2
    }
    if [[ -n "$COREAI_MODEL_ROOT" ]]; then
        [[ -d "$COREAI_MODEL_ROOT" ]] || {
            echo "MESH_APPLE_COREAI_MODEL_ROOT must point to a published .aimodel resource directory" >&2
            exit 2
        }
    fi
    [[ -z "$COREAI_MODEL_ROOT" || -z "$(find "$COREAI_MODEL_ROOT" -type l -print -quit)" ]] || {
        echo "Core AI model resources must not contain symlinks" >&2
        exit 2
    }
    if [[ -z "$COREAI_MODEL_ID" && -n "$COREAI_MODEL_REF" ]]; then
        COREAI_MODEL_ID="${COREAI_MODEL_REF%@*}"
    fi
    if [[ -z "$COREAI_MODEL_VERSION" && "$COREAI_MODEL_REF" == *@* ]]; then
        COREAI_MODEL_VERSION="${COREAI_MODEL_REF##*@}"
    fi
    [[ "$COREAI_MODEL_ID" == apple/coreai/* || "$COREAI_MODEL_ID" == */* ]] || {
        echo "MESH_APPLE_COREAI_MODEL_ID must be apple/coreai/<name> or an owner/repository HF identity" >&2
        exit 2
    }
    [[ -n "$COREAI_MODEL_VERSION" ]] || {
        echo "MESH_APPLE_COREAI_MODEL_VERSION is required for a Core AI artifact" >&2
        exit 2
    }
fi

if [[ "$RELEASE_MODE" != "0" && "$RELEASE_MODE" != "1" ]]; then
    echo "MESH_APPLE_RUNTIME_RELEASE must be 0 or 1" >&2
    exit 2
fi
if [[ "$RELEASE_MODE" == "1" ]]; then
    [[ "$IDENTITY" != "-" ]] || {
        echo "release packaging requires a Developer ID Application signing identity" >&2
        exit 2
    }
    [[ -n "$NOTARY_PROFILE" ]] || {
        echo "release packaging requires MESH_APPLE_RUNTIME_NOTARY_PROFILE" >&2
        exit 2
    }
    [[ -n "$ARCHIVE_URL" ]] || {
        echo "release packaging requires MESH_APPLE_RUNTIME_ARCHIVE_URL" >&2
        exit 2
    }
fi

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

swift package resolve --package-path "$PACKAGE_PATH"
"$PACKAGE_PATH/Packaging/prepare-coreai.sh"
swift build -c release --package-path "$PACKAGE_PATH"
BIN_DIR="$(swift build -c release --show-bin-path --package-path "$PACKAGE_PATH")"
SOURCE_BINARY="$BIN_DIR/mesh-apple-runtime"
[[ -x "$SOURCE_BINARY" ]] || {
    echo "missing Apple runtime executable: $SOURCE_BINARY" >&2
    exit 2
}

mkdir -p "$OUTPUT_ROOT"
rm -rf "$BUNDLE_DIR"
rm -f \
    "$OUTPUT_ROOT/$BUNDLE_ID.zip" \
    "$OUTPUT_ROOT/$BUNDLE_ID.zip.sha256" \
    "$OUTPUT_ROOT/provider-runtimes.json" \
    "$OUTPUT_ROOT/provider-runtimes.json.sha256" \
    "$OUTPUT_ROOT/notarization.json"
mkdir -p "$BUNDLE_DIR/bin" "$BUNDLE_DIR/Resources"
cp "$SOURCE_BINARY" "$BUNDLE_DIR/bin/mesh-apple-runtime"
cp "$PACKAGE_PATH/README.md" "$BUNDLE_DIR/README.md"
cp "$PACKAGE_PATH/Packaging/Entitlements/background-inference.entitlements" \
    "$BUNDLE_DIR/Resources/background-inference.entitlements"
if [[ -n "$COREAI_MODEL_ROOT" || -n "$COREAI_MODEL_REF" ]]; then
    if [[ -n "$COREAI_MODEL_ROOT" ]]; then
        mkdir -p "$BUNDLE_DIR/Models"
        cp -R "$COREAI_MODEL_ROOT" "$BUNDLE_DIR/Models/coreai-model"
    fi
    python3 - "$BUNDLE_DIR/Resources/coreai-model.json" "$COREAI_MODEL_ID" \
        "$COREAI_MODEL_VERSION" "$COREAI_CONTEXT_SIZE" "$COREAI_LANGUAGES" \
        "$COREAI_MODEL_ROOT" "$COREAI_MODEL_REF" <<'PY'
import json
import sys

output, model_id, version, context_size, languages, model_root, model_ref = sys.argv[1:]
configuration = {
    "id": model_id,
    "version": version,
    "contextSize": int(context_size),
    "languages": [item for item in languages.split(",") if item],
}
if model_root:
    configuration["path"] = "Models/coreai-model"
if model_ref:
    configuration["reference"] = model_ref
with open(output, "w", encoding="utf-8") as handle:
    json.dump(configuration, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
fi

codesign_args=(--force --sign "$IDENTITY")
if [[ "$RELEASE_MODE" == "1" ]]; then
    codesign_args+=(--options runtime --timestamp)
elif [[ "$IDENTITY" != "-" ]]; then
    codesign_args+=(--options runtime --timestamp=none)
fi
if [[ -n "$ENTITLEMENTS" ]]; then
    codesign_args+=(--entitlements "$ENTITLEMENTS")
fi
codesign "${codesign_args[@]}" "$BUNDLE_DIR/bin/mesh-apple-runtime"
codesign --verify --strict --verbose=2 "$BUNDLE_DIR/bin/mesh-apple-runtime"

BINARY_SHA="$(shasum -a 256 "$BUNDLE_DIR/bin/mesh-apple-runtime" | awk '{print $1}')"
README_SHA="$(shasum -a 256 "$BUNDLE_DIR/README.md" | awk '{print $1}')"
ENTITLEMENT_SHA="$(shasum -a 256 "$BUNDLE_DIR/Resources/background-inference.entitlements" | awk '{print $1}')"
OS_VERSION="$(sw_vers -productVersion)"
OS_BUILD="$(sw_vers -buildVersion)"
XCODE_VERSION="$(xcodebuild -version | awk 'NR == 1 { print $2 }')"
SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
CODESIGN_DETAILS="$(codesign -dv --verbose=4 "$BUNDLE_DIR/bin/mesh-apple-runtime" 2>&1)"
SIGNING_IDENTITY="$(printf '%s\n' "$CODESIGN_DETAILS" | awk -F= '/^Authority=/ && !found {print $2; found=1}')"
if [[ -z "$SIGNING_IDENTITY" ]]; then
    SIGNING_IDENTITY="ad-hoc"
fi
TEAM_IDENTIFIER="$(printf '%s\n' "$CODESIGN_DETAILS" | awk -F= '/^TeamIdentifier=/ {print $2; exit}')"
SIGNING_IDENTIFIER="$(printf '%s\n' "$CODESIGN_DETAILS" | awk -F= '/^Identifier=/ {print $2; exit}')"
if [[ "$TEAM_IDENTIFIER" == "not set" ]]; then
    TEAM_IDENTIFIER=""
fi
if [[ "$RELEASE_MODE" == "1" ]]; then
    [[ "$SIGNING_IDENTITY" == "Developer ID Application:"* && -n "$TEAM_IDENTIFIER" ]] || {
        echo "release signature is not a Developer ID Application signature" >&2
        exit 2
    }
fi
ENTITLEMENT_KEYS_JSON="[]"
if [[ -n "$ENTITLEMENTS" ]]; then
    ENTITLEMENT_KEYS_JSON="$(plutil -convert json -o - "$ENTITLEMENTS" | python3 -c \
        'import json, sys; print(json.dumps(sorted(json.load(sys.stdin).keys())))')"
fi

python3 - \
    "$BUNDLE_DIR/provider-runtime.json" \
    "$BUNDLE_ID" \
    "$RUNTIME_VERSION" \
    "$BINARY_SHA" \
    "$README_SHA" \
    "$ENTITLEMENT_SHA" \
    "$OS_VERSION" \
    "$OS_BUILD" \
    "$XCODE_VERSION" \
    "$SDK_VERSION" \
    "$SIGNING_IDENTITY" \
    "$TEAM_IDENTIFIER" \
    "$SIGNING_IDENTIFIER" \
    "$ENTITLEMENT_KEYS_JSON" \
    "$RELEASE_MODE" \
    "$BUNDLE_DIR" \
    "$COREAI_MODEL_ID" \
    "$COREAI_MODEL_VERSION" <<'PY'
import json
import hashlib
from pathlib import Path
import sys

(
    output,
    bundle_id,
    runtime_version,
    binary_sha,
    readme_sha,
    entitlement_sha,
    os_version,
    os_build,
    xcode_version,
    sdk_version,
    signing_identity,
    team_identifier,
    signing_identifier,
    entitlement_keys_json,
    release_mode,
    bundle_dir,
    coreai_model_id,
    coreai_model_version,
) = sys.argv[1:]

files = {
    "bin/mesh-apple-runtime": f"sha256:{binary_sha}",
    "README.md": f"sha256:{readme_sha}",
    "Resources/background-inference.entitlements": f"sha256:{entitlement_sha}",
}
bundle_path = Path(bundle_dir)
if coreai_model_id:
    config_path = bundle_path / "Resources/coreai-model.json"
    files["Resources/coreai-model.json"] = "sha256:" + hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    for path in sorted((bundle_path / "Models/coreai-model").rglob("*")):
        if path.is_file():
            relative = path.relative_to(bundle_path).as_posix()
            files[relative] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

models = [{"id": "apple/system", "kind": "system"}]
if coreai_model_id:
    models = [{
        "id": coreai_model_id,
        "kind": "coreai",
    }]

manifest = {
    "schema_version": 1,
    "runtime": {
        "id": bundle_id,
        "version": runtime_version,
        "provider_kind": "apple",
        "protocol_version": "0.1",
        "platform": {
            "os": "macos",
            "arch": "arm64",
            "target": "aarch64-apple-darwin",
            "minimum_os_version": "27.0",
        },
        "entrypoint": "bin/mesh-apple-runtime",
        "models": models,
        "features": [
            "availability",
            "streaming",
            "cancellation",
            "usage",
            "guided_generation",
            "tool_calling",
            "loopback_rest",
        ],
        "files": files,
        "build": {
            "macos": os_version,
            "macos_build": os_build,
            "xcode": xcode_version,
            "sdk": sdk_version,
        },
        "signature": {
            "identity": signing_identity,
            "team_identifier": team_identifier or None,
            "signing_identifier": signing_identifier or None,
            "entitlements": json.loads(entitlement_keys_json),
            "notarized": release_mode == "1",
        },
    }
}

with open(output, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

just --justfile "$REPO_ROOT/Justfile" with-lld \
    cargo run --quiet -p mesh-llm-provider-runtime --example inspect -- "$BUNDLE_DIR"

ARCHIVE="$OUTPUT_ROOT/$BUNDLE_ID.zip"
ditto -c -k --keepParent "$BUNDLE_DIR" "$ARCHIVE"

if [[ "$RELEASE_MODE" == "1" ]]; then
    xcrun notarytool submit "$ARCHIVE" \
        --keychain-profile "$NOTARY_PROFILE" \
        --wait \
        --output-format json \
        >"$OUTPUT_ROOT/notarization.json"
    python3 - "$OUTPUT_ROOT/notarization.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
if result.get("status") != "Accepted":
    raise SystemExit(f"notarization failed: {result.get('status', 'unknown status')}")
print(f"notarization accepted: {result.get('id', 'submission id unavailable')}")
PY
    spctl --assess --type execute --verbose=2 "$BUNDLE_DIR/bin/mesh-apple-runtime"
fi

ARCHIVE_SHA="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
printf '%s  %s\n' "$ARCHIVE_SHA" "$(basename "$ARCHIVE")" > "$ARCHIVE.sha256"

python3 - \
    "$BUNDLE_DIR/provider-runtime.json" \
    "$OUTPUT_ROOT/provider-runtimes.json" \
    "$ARCHIVE_SHA" \
    "$ARCHIVE_URL" <<'PY'
import json
import sys

bundle_manifest_path, release_manifest_path, archive_sha, archive_url = sys.argv[1:]
with open(bundle_manifest_path, encoding="utf-8") as handle:
    artifact = json.load(handle)["runtime"]
artifact["archive_sha256"] = f"sha256:{archive_sha}"
if archive_url:
    artifact["url"] = archive_url

release_manifest = {"schema_version": 1, "artifacts": [artifact]}
with open(release_manifest_path, "w", encoding="utf-8") as handle:
    json.dump(release_manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

RELEASE_MANIFEST="$OUTPUT_ROOT/provider-runtimes.json"
RELEASE_MANIFEST_SHA="$(shasum -a 256 "$RELEASE_MANIFEST" | awk '{print $1}')"
printf '%s  %s\n' "$RELEASE_MANIFEST_SHA" "$(basename "$RELEASE_MANIFEST")" \
    > "$RELEASE_MANIFEST.sha256"

just --justfile "$REPO_ROOT/Justfile" with-lld \
    cargo run --quiet -p mesh-llm-provider-runtime --example inspect_release -- \
    "$RELEASE_MANIFEST"
just --justfile "$REPO_ROOT/Justfile" with-lld \
    cargo run --quiet -p mesh-llm-provider-runtime --example inspect_archive -- \
    "$RELEASE_MANIFEST" "$ARCHIVE"

echo "packaged experimental Apple runtime:"
echo "  bundle: $BUNDLE_DIR"
echo "  archive: $ARCHIVE"
echo "  release manifest: $RELEASE_MANIFEST"
echo "  signing: $SIGNING_IDENTITY"
echo "  release eligible: $RELEASE_MODE"
