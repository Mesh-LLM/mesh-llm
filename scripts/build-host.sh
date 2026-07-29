#!/usr/bin/env bash
# Build the backend-neutral MeshLLM host. Native code belongs in a separately
# packaged runtime; this script must never prepare or link a llama.cpp backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UI_DIR="$REPO_ROOT/crates/mesh-llm-ui"
BUILD_PROFILE="${MESH_LLM_BUILD_PROFILE:-debug}"

usage() {
    echo "usage: scripts/build-host.sh [--profile debug|dev|release]" >&2
}

while (($# > 0)); do
    case "$1" in
        --profile)
            BUILD_PROFILE="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

case "$BUILD_PROFILE" in
    debug|dev|release) ;;
    *)
        echo "unsupported host build profile: $BUILD_PROFILE" >&2
        exit 1
        ;;
esac

append_rustflag() {
    local flag="$1"
    case " ${RUSTFLAGS:-} " in
        *" $flag "*) ;;
        *) export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$flag" ;;
    esac
}

configure_lld_linker() {
    case "$(uname -s)" in
        Linux)
            command -v ld.lld >/dev/null 2>&1 || {
                echo "Error: LLVM ld.lld was not found; install lld and retry." >&2
                exit 1
            }
            append_rustflag "-C link-arg=-fuse-ld=lld"
            ;;
        Darwin)
            local lld=""
            if command -v ld64.lld >/dev/null 2>&1; then
                lld="$(command -v ld64.lld)"
            elif command -v brew >/dev/null 2>&1; then
                local prefix
                prefix="$(brew --prefix lld 2>/dev/null || true)"
                [[ -x "$prefix/bin/ld64.lld" ]] && lld="$prefix/bin/ld64.lld"
            fi
            [[ -n "$lld" ]] || {
                echo "Error: LLVM ld64.lld was not found; run 'brew install lld'." >&2
                exit 1
            }
            append_rustflag "-C link-arg=-fuse-ld=$lld"
            ;;
        *)
            echo "unsupported OS for a dynamic host build: $(uname -s)" >&2
            exit 1
            ;;
    esac
}

configure_rust_cache() {
    if [[ -z "${RUSTC_WRAPPER:-}" ]] && command -v sccache >/dev/null 2>&1; then
        RUSTC_WRAPPER="$(command -v sccache)"
        export RUSTC_WRAPPER
    fi
}

configure_lld_linker
configure_rust_cache

echo "Building backend-neutral MeshLLM host (profile: $BUILD_PROFILE)."
if [[ "${MESH_LLM_SKIP_UI:-0}" != "1" ]]; then
    MESH_LLM_BUILD_PROFILE="$BUILD_PROFILE" "$SCRIPT_DIR/build-ui.sh" "$UI_DIR"
else
    echo "Skipping mesh-llm UI build because MESH_LLM_SKIP_UI=1."
fi

cargo_args=(build --locked -p mesh-llm --bin mesh-llm --no-default-features \
    --features "web-ui,dynamic-native-runtime")
if [[ "$BUILD_PROFILE" == "release" ]]; then
    cargo_args=(build --release --locked -p mesh-llm --bin mesh-llm --no-default-features \
        --features "web-ui,dynamic-native-runtime")
fi
(cd "$REPO_ROOT" && cargo "${cargo_args[@]}")

if [[ "$BUILD_PROFILE" == "release" ]]; then
    echo "Mesh host: target/release/mesh-llm"
else
    echo "Mesh host: target/debug/mesh-llm"
fi
