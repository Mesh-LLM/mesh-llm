#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

cargo build -p mesh-api-ffi

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
        if [ -f '"$REPO_ROOT"'/target/debug/libmesh_ffi.dylib ] && [ ! -e '"$REPO_ROOT"'/target/debug/libuniffi_mesh_ffi.dylib ]; then
            ln -sf libmesh_ffi.dylib '"$REPO_ROOT"'/target/debug/libuniffi_mesh_ffi.dylib
        fi
        if [ -f '"$REPO_ROOT"'/target/debug/libmesh_ffi.so ] && [ ! -e '"$REPO_ROOT"'/target/debug/libuniffi_mesh_ffi.so ]; then
            ln -sf libmesh_ffi.so '"$REPO_ROOT"'/target/debug/libuniffi_mesh_ffi.so
        fi
        export JAVA_TOOL_OPTIONS="-Djna.library.path='"$REPO_ROOT"'/target/debug"
        cd '"$REPO_ROOT"'/sdk/kotlin/example/example-jvm
        ./gradlew --no-daemon run --args="$MESH_SDK_INVITE_TOKEN"
    '
