#!/usr/bin/env bash
# Build the standalone Skippy CLI and its packageable native runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND=""
CUDA_ARCH=""
ROCM_ARCH=""

usage() {
    echo "usage: skippy/scripts/build-development-product.sh [--backend BACKEND] [--cuda-arch LIST] [--rocm-arch LIST]" >&2
}

while (($# > 0)); do
    case "$1" in
        --backend) BACKEND="${2:-}"; shift 2 ;;
        --cuda-arch) CUDA_ARCH="${2:-}"; shift 2 ;;
        --rocm-arch) ROCM_ARCH="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

normalize_recipe_argument() {
    local value="$1"
    shift
    local name
    for name in "$@"; do
        case "$value" in
            "$name="*) printf '%s\n' "${value#*=}"; return 0 ;;
        esac
    done
    printf '%s\n' "$value"
}

BACKEND="$(normalize_recipe_argument "$BACKEND" backend)"
CUDA_ARCH="$(normalize_recipe_argument "$CUDA_ARCH" cuda_arch cuda-arch)"
ROCM_ARCH="$(normalize_recipe_argument "$ROCM_ARCH" rocm_arch rocm-arch amd_arch amd-arch)"

if [[ -z "$BACKEND" ]]; then
    case "$(uname -s)" in
        Darwin) BACKEND=metal ;;
        Linux)
            if command -v nvidia-smi >/dev/null 2>&1 || command -v tegrastats >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1; then
                BACKEND=cuda
            elif command -v rocm-smi >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1 || command -v hipcc >/dev/null 2>&1 || [[ -x /opt/rocm/bin/hipcc ]]; then
                BACKEND=rocm
            elif command -v glslc >/dev/null 2>&1 && \
                { (command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1) || \
                  pkg-config --exists vulkan 2>/dev/null || [[ -n "${VULKAN_SDK:-}" ]]; }; then
                BACKEND=vulkan
            else
                BACKEND=cpu
            fi
            ;;
        *) BACKEND=cpu ;;
    esac
fi

runtime_out="$REPO_ROOT/target/debug/native-runtimes"
rm -rf "$runtime_out"
runtime_args=(--build --backend "$BACKEND" --out "$runtime_out")
[[ -n "$CUDA_ARCH" ]] && export LLAMA_STAGE_CUDA_ARCHITECTURES="$CUDA_ARCH"
[[ -n "$ROCM_ARCH" ]] && export LLAMA_STAGE_AMDGPU_TARGETS="$ROCM_ARCH"
"$SCRIPT_DIR/package-native-runtime.sh" "${runtime_args[@]}"
just skippy-cli-build

echo "Built standalone Skippy product:"
echo "  CLI:     $REPO_ROOT/target/debug/skippy"
echo "  runtime: $runtime_out"
echo "Run Skippy with --runtime-bundle $runtime_out."
