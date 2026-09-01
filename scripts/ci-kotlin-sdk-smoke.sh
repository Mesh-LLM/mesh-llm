#!/usr/bin/env bash
set -euo pipefail

SKIP_BUILD=0
if [[ "$#" -eq 8 ]]; then
    if [[ "$8" != "--skip-build" ]]; then
        echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> <native-sdk-input-dir> <target> <backend> <profile> [--skip-build]" >&2
        exit 1
    fi
    SKIP_BUILD=1
elif [[ "$#" -ne 7 ]]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> <native-sdk-input-dir> <target> <backend> <profile> [--skip-build]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

scripts/check-sdk-contract.sh
if [[ "$SKIP_BUILD" == "1" ]]; then
    scripts/package-sdk-console-assets.sh --sdk kotlin --skip-build
else
    scripts/package-sdk-console-assets.sh --sdk kotlin
fi
scripts/verify-sdk-console-assets.sh --sdk kotlin

native_sdk_tmp="$(mktemp -d)"
trap 'rm -rf "$native_sdk_tmp"' EXIT
native_sdk_artifact_dir="$(
    scripts/restore-native-sdk-input.sh \
        "$4" \
        "$native_sdk_tmp/extracted" \
        "$5" \
        "$6" \
        "$7"
)"
native_sdk_uniffi_library="$(
    python3 - "$native_sdk_artifact_dir/manifest.json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    manifest = json.load(fh)
print(os.path.dirname(manifest.get("uniffi_library") or manifest["library"]))
PY
)"
export MESHLLM_KOTLIN_JNA_LIBRARY_PATH="$native_sdk_artifact_dir/$native_sdk_uniffi_library"
native_runtime_dir="$(
    scripts/ci-prepare-native-runtime.sh \
        "$REPO_ROOT/target/kotlin-native-runtime" \
        cpu \
        --reuse-from-binary "$1"
)"
export MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR="$native_runtime_dir"

# shellcheck disable=SC2016 # The nested shell expands exported fixture variables.
scripts/ci-sdk-fixture.sh "$1" "$2" "$3" -- \
    bash -lc '
        set -euo pipefail
        if [ -x /usr/libexec/java_home ]; then
            JAVA_HOME="$(/usr/libexec/java_home -v 21 2>/dev/null || printf "%s" "${JAVA_HOME:-}")"
            export JAVA_HOME
        fi
        if [ -n "${JAVA_HOME:-}" ]; then
            export ORG_GRADLE_JAVA_HOME="${ORG_GRADLE_JAVA_HOME:-$JAVA_HOME}"
            export GRADLE_OPTS="${GRADLE_OPTS:-} -Dorg.gradle.java.installations.auto-detect=false -Dorg.gradle.java.installations.paths=$ORG_GRADLE_JAVA_HOME"
        fi
        export MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR="${MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR:?}"
        export MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="${MESH_LLM_NATIVE_RUNTIME_CACHE_DIR:?}"
        export JNA_LIBRARY_PATH="${MESHLLM_KOTLIN_JNA_LIBRARY_PATH}${JNA_LIBRARY_PATH:+:$JNA_LIBRARY_PATH}"
        export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djna.library.path=$MESHLLM_KOTLIN_JNA_LIBRARY_PATH"
        cd '"$REPO_ROOT"'/sdk/kotlin/example/example-jvm
        ./gradlew --no-daemon run --args="$MESH_SDK_INVITE_TOKEN"
    '
