#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/ci-prepare-native-runtime.sh <out-dir> [backend] [options]

Options:
  --reuse-from-binary PATH
      Prefer a compatible packaged runtime under PATH's adjacent
      native-runtimes directory. A present but incompatible bundle is an error.

Environment:
  MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK
      Set to 1 to build when --reuse-from-binary has no adjacent bundle, or 0
      to reject that fallback. The default is 0 in CI and 1 for local runs.
EOF
}

if [[ "$#" -lt 1 ]]; then
    usage
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/automation.sh"
OUT_DIR="$1"
shift

BACKEND="cpu"
if [[ "$#" -gt 0 && "$1" != --* ]]; then
    BACKEND="$1"
    shift
fi

REUSE_BINARY=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --reuse-from-binary)
            REUSE_BINARY="${2:?missing binary path}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-ci-runtime-${BACKEND}"
TEMP_ROOT=""

cleanup() {
    if [[ -n "$TEMP_ROOT" ]]; then
        rm -rf -- "$TEMP_ROOT"
    fi
}
trap cleanup EXIT

build_fallback_enabled() {
    local configured="${MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK:-}"
    if [[ -z "$configured" ]]; then
        [[ "${CI:-}" != "true" ]]
        return
    fi
    case "$configured" in
        1|true) return 0 ;;
        0|false) return 1 ;;
        *)
            echo "MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK must be 0, 1, false, or true" >&2
            exit 1
            ;;
    esac
}

select_compatible_runtime() {
    local binary="$1"
    local runtime_root="$2"
    local compatibility_json
    local expected_skippy_abi
    local runtime_dir

    if [[ ! -x "$binary" ]]; then
        echo "Missing executable mesh-llm binary for runtime reuse: $binary" >&2
        return 1
    fi

    TEMP_ROOT="$(mktemp -d "${RUNNER_TEMP:-/tmp}/mesh-sdk-runtime-compat.XXXXXX")"
    compatibility_json="$TEMP_ROOT/available.json"
    expected_skippy_abi="$(
        mesh_automation native package-source-version abi "$REPO_ROOT/skippy/crates/skippy-ffi/src/lib.rs"
    )"
    env \
        -u MESH_LLM_CONFIG \
        -u MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR \
        -u MESH_LLM_NATIVE_RUNTIME_CACHE_DIR \
        HOME="$TEMP_ROOT/home" \
        XDG_CACHE_HOME="$TEMP_ROOT/xdg-cache" \
        XDG_CONFIG_HOME="$TEMP_ROOT/xdg-config" \
        "$binary" \
        --log-format json \
        runtime list \
        --available \
        --bundle-dir "$runtime_root" \
        --cache-dir "$TEMP_ROOT/cache" \
        --json >"$compatibility_json"

    runtime_dir="$(
        mesh_automation prepared-input sdk-runtime-select \
            "$runtime_root" \
            "$BACKEND" \
            "$compatibility_json" \
            "$expected_skippy_abi"
    )"

    scripts/verify-native-runtime-package.sh "$runtime_dir" >&2
    echo "Reusing compatible native runtime beside the smoke binary:" >&2
    echo "  runtime: $runtime_dir" >&2
    printf '%s\n' "$runtime_dir"
}

resolve_reuse_binary() {
    local binary_dir
    binary_dir="$(cd "$(dirname "$REUSE_BINARY")" && pwd)"
    printf '%s/%s\n' "$binary_dir" "$(basename "$REUSE_BINARY")"
}

if [[ -n "$REUSE_BINARY" ]]; then
    REUSE_BINARY="$(resolve_reuse_binary)"
fi

cd "$REPO_ROOT"

if [[ -n "$REUSE_BINARY" ]]; then
    adjacent_runtime_root="$(dirname "$REUSE_BINARY")/native-runtimes"
    if [[ -d "$adjacent_runtime_root" ]]; then
        select_compatible_runtime "$REUSE_BINARY" "$adjacent_runtime_root"
        exit 0
    fi
    if ! build_fallback_enabled; then
        echo "Adjacent native runtime bundle is required in CI: $adjacent_runtime_root" >&2
        echo "Set MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK=1 only for an explicit standalone fallback." >&2
        exit 1
    fi
    echo "No adjacent native runtime bundle found; building the standalone fallback." >&2
fi

rm -rf "$OUT_DIR"
LLAMA_STAGE_LINK_MODE=dynamic \
LLAMA_STAGE_BACKEND="$BACKEND" \
LLAMA_STAGE_BUILD_DIR="$BUILD_DIR" \
LLAMA_BUILD_DIR="$BUILD_DIR" \
    scripts/package-native-runtime.sh \
        --build \
        --backend "$BACKEND" \
        --out "$OUT_DIR" >&2

scripts/verify-native-runtime-package.sh "$OUT_DIR"/meshllm-native-runtime-*.tar.gz >&2

runtime_dir="$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'meshllm-native-runtime-*' | sort | head -n 1)"
if [[ -z "$runtime_dir" ]]; then
    echo "native runtime artifact directory was not produced" >&2
    exit 1
fi

printf '%s\n' "$runtime_dir"
