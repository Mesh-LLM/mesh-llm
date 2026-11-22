#!/usr/bin/env bash
# Build the Skippy native runtime this checkout needs, from source, and put it
# beside the host binary, where mesh-llm finds it with no settings.
#
#   scripts/build-skippy-runtime.sh [--backend NAME] [--cuda-arch LIST]
#       [--rocm-arch LIST] [--profile debug|dev|release]
#
# Source: the Skippy commit pinned in Cargo.toml, fetched into .deps/skippy, or
# the local checkout chosen with scripts/skippy-source.sh.
# Backend: --backend, else MESH_LLM_RUNTIME_BACKEND, else detected the way the
# in-tree build did: metal on macOS; on Linux cuda, rocm or vulkan when their
# tools are installed, else cpu; the same on Windows. Names are Skippy's.
# Output: target/<debug|release>/native-runtimes. A runtime already built from
# the same pinned commit and backend is kept; a local checkout always rebuilds.
# On Windows run this from Git Bash; it hands the build to Skippy's
# scripts/build-windows.ps1, which sets up MSVC.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="${MESH_LLM_RUNTIME_BACKEND:-}"
cuda_arch=""
rocm_arch=""
profile=debug
while (($# > 0)); do
    case "$1" in
        --backend) [[ -z "${2:-}" ]] || backend="$2"; shift 2 ;;
        --cuda-arch) cuda_arch="${2:-}"; shift 2 ;;
        --rocm-arch) rocm_arch="${2:-}"; shift 2 ;;
        --profile) profile="${2:-}"; shift 2 ;;
        *) sed -n '2,16s/^# \{0,1\}//p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
    esac
done
# `just build backend=cuda cuda_arch=...` is the established spelling, and just
# passes it through as the literal value.
backend="${backend#backend=}"
cuda_arch="${cuda_arch#cuda_arch=}"
rocm_arch="${rocm_arch#rocm_arch=}"
case "$profile" in
    debug | dev) out="$REPO_ROOT/target/debug/native-runtimes" ;;
    release) out="$REPO_ROOT/target/release/native-runtimes" ;;
    *) echo "unsupported profile: $profile" >&2; exit 2 ;;
esac
if [[ -z "$backend" ]]; then
    case "$(uname -s)" in
        Darwin) backend=metal ;;
        Linux)
            if command -v nvidia-smi >/dev/null 2>&1 || command -v tegrastats >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1; then
                backend=cuda
            elif command -v rocm-smi >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1 || command -v hipcc >/dev/null 2>&1 || [[ -x /opt/rocm/bin/hipcc ]]; then
                backend=rocm
            elif command -v glslc >/dev/null 2>&1 &&
                { (command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1) ||
                    pkg-config --exists vulkan 2>/dev/null || [[ -n "${VULKAN_SDK:-}" ]]; }; then
                backend=vulkan
            else
                backend=cpu
            fi
            ;;
        MINGW* | MSYS* | CYGWIN*)
            # The checks the in-tree Windows build made, in its order.
            pf="${PROGRAMFILES:-}"
            has_hip_config() { compgen -G "$1/lib/cmake/hip/hip-config.cmake" >/dev/null || compgen -G "$1/*/lib/cmake/hip/hip-config.cmake" >/dev/null; }
            if nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
                backend=cuda
            elif { [[ -n "${HIP_PATH:-}" ]] && has_hip_config "$HIP_PATH"; } ||
                { [[ -n "${ROCM_PATH:-}" ]] && has_hip_config "$ROCM_PATH"; } ||
                { [[ -n "$pf" ]] && [[ -d "$pf/AMD/ROCm" || -d "$pf/AMD/HIP" ]]; } ||
                command -v hipInfo >/dev/null 2>&1 || command -v hipconfig >/dev/null 2>&1; then
                backend=rocm
            elif vulkaninfo --summary >/dev/null 2>&1 ||
                [[ -n "${VULKAN_SDK:-}" && -d "${VULKAN_SDK:-}" ]] ||
                { [[ -n "$pf" ]] && compgen -G "$pf/VulkanSDK/*/" >/dev/null; }; then
                backend=vulkan
            else
                backend=cpu
            fi
            ;;
        *) backend=cpu ;;
    esac
fi
[[ "$backend" != hip ]] || backend=rocm

source_state="$("$REPO_ROOT/scripts/skippy-source.sh")"
if [[ "$source_state" == local\ * ]]; then
    checkout="$(sed -n 's/^# checkout: //p' "$REPO_ROOT/.cargo/skippy-local.toml")"
    stamp=""
else
    read -r _ url rev <<<"$source_state"
    checkout="$REPO_ROOT/.deps/skippy"
    stamp="$rev $backend $cuda_arch $rocm_arch"
    if [[ "$(git -C "$checkout" rev-parse HEAD 2>/dev/null || true)" != "$rev" ]]; then
        mkdir -p "$checkout"
        [[ -d "$checkout/.git" ]] || git -C "$checkout" init -q
        git -C "$checkout" fetch -q --depth 1 "$url" "$rev"
        git -C "$checkout" checkout -q --detach FETCH_HEAD
    fi
fi

if [[ -n "$stamp" && "$(cat "$out/.built-from" 2>/dev/null || true)" == "$stamp" ]]; then
    echo "Skippy runtime: $out ($backend, already built from ${rev:0:9})"
    exit 0
fi

echo "Building the Skippy $backend runtime from $checkout ($(git -C "$checkout" rev-parse --short=9 HEAD))."
rm -rf "$out"
mkdir -p "$out"
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*)
        (cd "$checkout" && powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 \
            -Backend "$backend" -CudaArch "$cuda_arch" -RocmArch "$rocm_arch" -RuntimeOnly)
        cp -R "$checkout/dist/native-runtimes/." "$out/"
        ;;
    *)
        [[ -z "$cuda_arch" ]] || export LLAMA_STAGE_CUDA_ARCHITECTURES="$cuda_arch"
        [[ -z "$rocm_arch" ]] || export LLAMA_STAGE_AMDGPU_TARGETS="$rocm_arch"
        (cd "$checkout" && scripts/package-native-runtime.sh --build --backend "$backend" --no-model-package-tool --out "$out")
        ;;
esac
[[ -z "$stamp" ]] || echo "$stamp" >"$out/.built-from"
echo "Skippy runtime: $out"
