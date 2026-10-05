#!/usr/bin/env bash
# Build the two local products in dependency order: Skippy's native runtime
# and standalone CLI, then the MeshLLM host and console. The native llama
# build is confined to package-native-runtime.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Git never activates a committed hook on clone, so enable the repository hooks
# on the first local development build. Skipped in CI, and never overrides a
# hooks path the developer chose themselves.
enable_repo_git_hooks() {
    [[ -z "${CI:-}" ]] || return 0
    local current
    current="$(git -C "$REPO_ROOT" config --get core.hooksPath 2>/dev/null || true)"
    [[ -z "$current" ]] || return 0
    git -C "$REPO_ROOT" config core.hooksPath scripts/hooks 2>/dev/null || return 0
    echo "enabled repository git hooks (core.hooksPath=scripts/hooks); commit messages are now checked locally" >&2
}

enable_repo_git_hooks

BACKEND=""
CUDA_ARCH=""
ROCM_ARCH=""
PROFILE="${MESH_LLM_BUILD_PROFILE:-debug}"

usage() {
    echo "usage: scripts/build-development-product.sh [--backend BACKEND] [--cuda-arch LIST] [--rocm-arch LIST] [--profile debug|dev]" >&2
}

while (($# > 0)); do
    case "$1" in
        --backend) BACKEND="${2:-}"; shift 2 ;;
        --cuda-arch) CUDA_ARCH="${2:-}"; shift 2 ;;
        --rocm-arch) ROCM_ARCH="${2:-}"; shift 2 ;;
        --profile) PROFILE="${2:-}"; shift 2 ;;
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

# Just recipe parameters are positional, but the established public spelling is
# `just build backend=cuda cuda_arch=...`. Preserve that spelling while the
# recipe remains a thin wrapper around this script.
BACKEND="$(normalize_recipe_argument "$BACKEND" backend)"
CUDA_ARCH="$(normalize_recipe_argument "$CUDA_ARCH" cuda_arch cuda-arch)"
ROCM_ARCH="$(normalize_recipe_argument "$ROCM_ARCH" rocm_arch rocm-arch amd_arch amd-arch)"

case "$PROFILE" in
    debug|dev) ;;
    *) echo "development product profile must be debug or dev, got: $PROFILE" >&2; exit 1 ;;
esac

host_dir="$REPO_ROOT/target/debug"
runtime_out="$host_dir/native-runtimes"
just skippy "$BACKEND" "$CUDA_ARCH" "$ROCM_ARCH"

# MeshLLM consumes Skippy but remains a separate, backend-neutral product.
MESH_LLM_BUILD_PROFILE="$PROFILE" just mesh "$PROFILE"

echo "Built local products:"
echo "  Skippy CLI:     $host_dir/skippy"
echo "  Skippy runtime: $runtime_out"
echo "  MeshLLM host:   $host_dir/mesh-llm"
echo "Skippy accepts --runtime-bundle $runtime_out; MeshLLM discovers the adjacent runtime automatically."
