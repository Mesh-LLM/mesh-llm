#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"
OUTPUT_DIR="$REPO_ROOT/target/apple-runtime/sdk-carriers"
SWIFT_TARGET_DIR="$REPO_ROOT/target/swift-provider-host-debug"
SWIFT_EXAMPLE_DIR="$REPO_ROOT/sdk/swift/example/MeshExampleApp"
KOTLIN_EXAMPLE_DIR="$REPO_ROOT/sdk/kotlin/example/example-jvm"
ACTIVE_PID=""
ACTIVE_STOP_FILE=""

cleanup() {
    if [[ -n "$ACTIVE_STOP_FILE" ]]; then
        : > "$ACTIVE_STOP_FILE"
    fi
    if [[ "$ACTIVE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || {
    echo "Apple SDK carrier conformance requires Apple silicon macOS" >&2
    exit 2
}
[[ -f "$PACKAGE_ROOT/provider-runtime.json" ]] || {
    echo "missing Apple runtime package; run just apple::package" >&2
    exit 2
}

if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/java" ]]; then
    JDK_HOME="$JAVA_HOME"
elif [[ -x "/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home/bin/java" ]]; then
    JDK_HOME="/Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home"
else
    JDK_HOME="$(/usr/libexec/java_home -v 21 2>/dev/null || true)"
fi
[[ -x "$JDK_HOME/bin/java" ]] || {
    echo "Kotlin/JVM carrier conformance requires JDK 21" >&2
    exit 2
}

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Building Node/Electron provider carrier..."
just --justfile "$REPO_ROOT/Justfile" with-lld \
    cargo build -p mesh-llm-nodejs

echo "Building Kotlin/JVM provider carrier..."
just --justfile "$REPO_ROOT/Justfile" with-lld \
    cargo build -p mesh-llm-ffi --no-default-features --features host,embedded-runtime
JAVA_HOME="$JDK_HOME" \
    "$KOTLIN_EXAMPLE_DIR/gradlew" --no-daemon -p "$KOTLIN_EXAMPLE_DIR" classes

echo "Building Swift provider carrier..."
just --justfile "$REPO_ROOT/Justfile" with-lld \
    env -u RUSTFLAGS \
    CARGO_TARGET_DIR="$SWIFT_TARGET_DIR" \
    MESH_SWIFT_FFI_PROFILE=debug \
    bash "$REPO_ROOT/sdk/swift/scripts/build-host-macos-xcframework.sh"
xattr -cr "$REPO_ROOT/sdk/swift/Generated/MeshLLMFFI.xcframework"
swift build --package-path "$SWIFT_EXAMPLE_DIR" --product AppleSystemHost

SWIFT_BIN="$(swift build --package-path "$SWIFT_EXAMPLE_DIR" --show-bin-path)/AppleSystemHost"
NODE_ADDON="$REPO_ROOT/target/debug/libmesh_llm_nodejs.dylib"
KOTLIN_FFI_DIR="$OUTPUT_DIR/kotlin-native"
mkdir -p "$KOTLIN_FFI_DIR"
cp "$REPO_ROOT/target/debug/libmeshllm_ffi.dylib" \
    "$KOTLIN_FFI_DIR/libuniffi_mesh_ffi.dylib"

wait_until_ready() {
    local ready_file="$1"
    local stderr_file="$2"
    for _ in $(seq 1 600); do
        [[ -s "$ready_file" ]] && return 0
        if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
            echo "SDK carrier exited before its REST host became ready" >&2
            cat "$stderr_file" >&2
            return 1
        fi
        sleep 0.1
    done
    echo "timed out waiting for SDK carrier REST host" >&2
    cat "$stderr_file" >&2
    return 1
}

wait_for_apple_model() {
    local rest_base_url="$1"
    local models_file="$2"
    local stderr_file="$3"
    for _ in $(seq 1 300); do
        if curl --silent --show-error "$rest_base_url/v1/models" >"$models_file" \
            && python3 - "$models_file" <<'PY'
import json
import pathlib
import sys

models = json.loads(pathlib.Path(sys.argv[1]).read_text())
raise SystemExit(0 if any(
    model.get("id", "").startswith("apple/system@")
    for model in models.get("data", [])
) else 1)
PY
        then
            return 0
        fi
        if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
            echo "SDK carrier exited before apple/system became available" >&2
            cat "$stderr_file" >&2
            return 1
        fi
        sleep 0.1
    done
    echo "timed out waiting for apple/system through SDK carrier" >&2
    cat "$stderr_file" >&2
    return 1
}

run_carrier() {
    local carrier="$1"
    shift
    local carrier_dir="$OUTPUT_DIR/$carrier"
    local ready_file="$carrier_dir/ready.json"
    local stop_file="$carrier_dir/stop"
    local stdout_file="$carrier_dir/carrier.stdout"
    local stderr_file="$carrier_dir/carrier.stderr"
    mkdir -p "$carrier_dir"

    echo "Running $carrier carrier through shared REST conformance..."
    MESH_LLM_PROVIDER_RUNTIME_BUNDLE_DIR=/invalid/sdk-carrier-must-ignore-environment \
    MESH_LLM_PROVIDER_RUNTIME_INDEX=/invalid/sdk-carrier-must-ignore-environment.json \
    MESH_LLM_PROVIDER_RUNTIME_CACHE_DIR="$carrier_dir/provider-cache" \
    MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC=1 \
        "$@" >"$stdout_file" 2>"$stderr_file" &
    ACTIVE_PID=$!
    ACTIVE_STOP_FILE="$stop_file"
    wait_until_ready "$ready_file" "$stderr_file"

    local api_base_url
    api_base_url="$(python3 - "$ready_file" "$carrier" <<'PY'
import json
import pathlib
import sys

ready = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert ready["carrier"] == sys.argv[2], ready
assert ready["apiBaseUrl"].startswith("http://127.0.0.1:"), ready
print(ready["apiBaseUrl"])
PY
    )"
    local rest_base_url="${api_base_url%/v1}"
    wait_for_apple_model "$rest_base_url" "$carrier_dir/models-ready.json" "$stderr_file"
    MESH_APPLE_RUNTIME_BASE_URL="$rest_base_url" \
    MESH_APPLE_RUNTIME_REST_OUTPUT_DIR="$carrier_dir/rest" \
        "$APPLE_ROOT/QA/rest.sh" >"$carrier_dir/rest-output.json"

    : > "$stop_file"
    wait "$ACTIVE_PID"
    ACTIVE_PID=""
    ACTIVE_STOP_FILE=""
}

run_carrier swift \
    "$SWIFT_BIN" "$PACKAGE_ROOT" "$OUTPUT_DIR/swift/ready.json" "$OUTPUT_DIR/swift/stop"

run_carrier node-electron \
    env MESHLLM_NODE_NATIVE_PATH="$NODE_ADDON" \
    node "$REPO_ROOT/sdk/node/example/apple-system-host.js" \
    "$PACKAGE_ROOT" "$OUTPUT_DIR/node-electron/ready.json" "$OUTPUT_DIR/node-electron/stop"

run_carrier kotlin-jvm \
    env JAVA_HOME="$JDK_HOME" \
    JAVA_TOOL_OPTIONS="-Djna.library.path=$KOTLIN_FFI_DIR" \
    MESH_APPLE_PROVIDER_ROOT="$PACKAGE_ROOT" \
    MESH_APPLE_SDK_READY_FILE="$OUTPUT_DIR/kotlin-jvm/ready.json" \
    MESH_APPLE_SDK_STOP_FILE="$OUTPUT_DIR/kotlin-jvm/stop" \
    "$KOTLIN_EXAMPLE_DIR/gradlew" --no-daemon -q -p "$KOTLIN_EXAMPLE_DIR" run

python3 - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
carriers = []
for carrier in ("swift", "node-electron", "kotlin-jvm"):
    summary = json.loads((root / carrier / "rest" / "summary.json").read_text())
    assert summary["status"] == "pass", summary
    carriers.append({
        "carrier": carrier,
        "api_base_url": json.loads((root / carrier / "ready.json").read_text())["apiBaseUrl"],
        "model": summary["model"],
        "versioned_model": summary["versioned_model"],
        "completion": summary["completion"]["choices"][0]["message"]["content"],
        "tool_executions": summary["tool"]["mesh_tool_executions"],
        "stream_done": summary["stream_done"],
        "client_disconnect_cancelled": summary["client_disconnect_cancelled"],
        "typed_model_error": summary["typed_model_error"],
    })

result = {
    "status": "pass",
    "provider_process": "mesh-apple-runtime",
    "provider_artifact_shared_by_all_carriers": True,
    "carriers": carriers,
}
(root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY
