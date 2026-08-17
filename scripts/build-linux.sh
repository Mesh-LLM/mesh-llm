#!/usr/bin/env bash
# Compatibility entry point for the unified Linux development product.
# Native ABI compilation remains solely in package-native-runtime.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND=""
CUDA_ARCH=""
ROCM_ARCH=""
PROFILE="${MESH_LLM_BUILD_PROFILE:-debug}"

usage() {
    echo "usage: scripts/build-linux.sh [--skip-ui] [--backend cpu|cuda|rocm|vulkan] [--cuda-arch SM_LIST] [--rocm-arch GFX_LIST]" >&2
}

while (($# > 0)); do
    case "$1" in
        --skip-ui) export MESH_LLM_SKIP_UI=1; shift ;;
        --backend) BACKEND="${2:-}"; shift 2 ;;
        --cuda-arch) CUDA_ARCH="${2:-}"; shift 2 ;;
        --rocm-arch) ROCM_ARCH="${2:-}"; shift 2 ;;
        --clean)
            echo "--clean is no longer supported by the unified build; remove only the specific target/native-runtime output you own." >&2
            exit 2
            ;;
        -h|--help) usage; exit 0 ;;
        -*)
            echo "unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            # Keep the former positional CUDA-architecture compatibility.
            [[ -z "$CUDA_ARCH" ]] || { usage; exit 2; }
            CUDA_ARCH="$1"
            shift
            ;;
    esac
done

exec "$SCRIPT_DIR/build-development-product.sh" \
    --profile "$PROFILE" \
    --backend "$BACKEND" \
    --cuda-arch "$CUDA_ARCH" \
    --rocm-arch "$ROCM_ARCH"
