#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CARGO_HTTP_MULTIPLEXING="${CARGO_HTTP_MULTIPLEXING:-false}"
export CARGO_NET_RETRY="${CARGO_NET_RETRY:-10}"

retry_transient() {
    local attempt=1
    local max_attempts=3

    while true; do
        if "$@"; then
            return 0
        fi
        if [ "$attempt" -ge "$max_attempts" ]; then
            return 1
        fi
        echo "command failed; retrying ($attempt/$max_attempts): $*" >&2
        sleep $((attempt * 5))
        attempt=$((attempt + 1))
    done
}

scripts/check-sdk-contract.sh

scripts/prepare-llama.sh "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"
LLAMA_STAGE_BACKEND=cpu \
LLAMA_STAGE_BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-ci-kotlin-cpu" \
LLAMA_BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-ci-kotlin-cpu" \
    scripts/build-llama.sh

LLAMA_STAGE_BACKEND=cpu \
LLAMA_STAGE_BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-ci-kotlin-cpu" \
    retry_transient cargo build -p mesh-llm-ffi --no-default-features --features host,embedded-runtime

native_sdk_out="$REPO_ROOT/target/kotlin-native-sdk"
LLAMA_STAGE_BACKEND=cpu \
LLAMA_STAGE_BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-ci-kotlin-cpu" \
    retry_transient scripts/package-native-sdk.sh \
        --backend cpu \
        --profile debug \
        --out "$native_sdk_out"
scripts/verify-native-sdk-package.sh "$native_sdk_out"/meshllm-native-*.tar.gz
native_runtime_dir="$(find "$native_sdk_out" -mindepth 1 -maxdepth 1 -type d -name 'meshllm-native-*' -print -quit)"
if [[ -z "$native_runtime_dir" ]]; then
    echo "native SDK runtime artifact directory was not produced" >&2
    exit 1
fi

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
        export MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR='"$native_runtime_dir"'
        cd '"$REPO_ROOT"'/sdk/kotlin/example/example-jvm
        ./gradlew --no-daemon run --args="$MESH_SDK_INVITE_TOKEN"
    '
