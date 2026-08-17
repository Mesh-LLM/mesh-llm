#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT/scripts/lib/cuda-toolkit.sh"

LLAMA_WORKDIR="${LLAMA_WORKDIR:-$ROOT/.deps/llama.cpp}"
LLAMA_BUILD_ROOT="${MESH_LLM_LLAMA_BUILD_ROOT:-$ROOT/.deps/llama-build}"
LLAMA_BACKEND="${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-${LLAMA_BACKEND:-cpu}}}"
LLAMA_LINK_MODE="${LLAMA_STAGE_LINK_MODE:-${SKIPPY_LLAMA_LINK_MODE:-static}}"
PRINT_BUILD_DIR=0
PRINT_JOBS=0
REQUIRE_EXISTING=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --print-build-dir)
      PRINT_BUILD_DIR=1
      shift
      ;;
    --print-jobs)
      PRINT_JOBS=1
      shift
      ;;
    --require-existing)
      REQUIRE_EXISTING=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

case "$LLAMA_BACKEND" in
  cpu|cuda|rocm|hip|vulkan|metal) ;;
  *)
    echo "unsupported LLAMA_STAGE_BACKEND: $LLAMA_BACKEND" >&2
    echo "expected one of: cpu, cuda, rocm, hip, vulkan, metal" >&2
    exit 1
    ;;
esac

case "$LLAMA_LINK_MODE" in
  static|dynamic) ;;
  *)
    echo "unsupported LLAMA_STAGE_LINK_MODE: $LLAMA_LINK_MODE" >&2
    echo "expected one of: static, dynamic" >&2
    exit 1
    ;;
esac

if [[ -n "${MESH_LLM_LLAMA_BUILD_MEMORY_BYTES:-}" ]] &&
  ! [[ "$MESH_LLM_LLAMA_BUILD_MEMORY_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MESH_LLM_LLAMA_BUILD_MEMORY_BYTES must be a positive integer byte count" >&2
  echo "  got: $MESH_LLM_LLAMA_BUILD_MEMORY_BYTES" >&2
  exit 2
fi

sanitize_build_component() {
  printf '%s' "$1" | tr ';, /:' '_____' | tr -cd 'A-Za-z0-9_.-'
}

default_build_dir_for_backend() {
  local suffix="$LLAMA_BACKEND"
  case "$LLAMA_BACKEND" in
    cpu)
      suffix="cpu"
      ;;
    cuda)
      local cuda_arch="${LLAMA_STAGE_CUDA_ARCHITECTURES:-${SKIPPY_CUDA_ARCHITECTURES:-}}"
      suffix="cuda-sm$(sanitize_build_component "$cuda_arch")"
      ;;
    rocm|hip)
      local amdgpu_targets="${LLAMA_STAGE_AMDGPU_TARGETS:-${SKIPPY_AMDGPU_TARGETS:-}}"
      suffix="rocm-$(sanitize_build_component "$amdgpu_targets")"
      ;;
  esac
  printf '%s/build-stage-abi-%s-%s\n' "$LLAMA_BUILD_ROOT" "$LLAMA_LINK_MODE" "$suffix"
}

detect_cpu_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

# Memory this build may actually use, in bytes, or empty when unknown.
#
# nproc reports the host CPU count even inside a memory-limited container, so
# CPU count alone overcommits memory on containerized CI runners. Prefer the
# cgroup limit that will actually kill the compiler, then fall back to the
# kernel's view of available memory.
detect_memory_budget_bytes() {
  local value cgroup_limit
  if [[ -n "${MESH_LLM_LLAMA_BUILD_MEMORY_BYTES:-}" ]]; then
    printf '%s\n' "$MESH_LLM_LLAMA_BUILD_MEMORY_BYTES"
    return 0
  fi

  for cgroup_limit in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [[ -r "$cgroup_limit" ]]; then
      value="$(cat "$cgroup_limit" 2>/dev/null || true)"
      # cgroup v2 reports "max" when unlimited; v1 reports a saturated value.
      if [[ "$value" =~ ^[0-9]+$ ]] && ((value > 0)) && ((value < 1 << 62)); then
        printf '%s\n' "$value"
        return 0
      fi
    fi
  done

  if [[ -r /proc/meminfo ]]; then
    value="$(awk '/^MemTotal:/ { print $2 * 1024; exit }' /proc/meminfo)"
    if [[ "$value" =~ ^[0-9]+$ ]] && ((value > 0)); then
      printf '%s\n' "$value"
      return 0
    fi
  fi

  if command -v sysctl >/dev/null 2>&1; then
    value="$(sysctl -n hw.memsize 2>/dev/null || true)"
    if [[ "$value" =~ ^[0-9]+$ ]] && ((value > 0)); then
      printf '%s\n' "$value"
      return 0
    fi
  fi

  return 0
}

# Peak resident memory one compile job may need for this backend, in bytes.
#
# CUDA translation units run the host compiler plus one cicc frontend per
# target architecture, so they need several times what a CPU translation unit
# needs. These are deliberately conservative: the failure mode of a too-small
# number is a slower build, and of a too-large number is a SIGSEGV or an
# internal compiler error with no diagnostic about our code.
memory_per_job_bytes() {
  case "$LLAMA_BACKEND" in
    cuda|rocm|hip)
      echo $((3 * 1024 * 1024 * 1024))
      ;;
    *)
      echo $((1024 * 1024 * 1024))
      ;;
  esac
}

detect_jobs() {
  if [[ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
    echo "$CMAKE_BUILD_PARALLEL_LEVEL"
    return 0
  fi

  local cpu_jobs memory_budget per_job memory_jobs
  cpu_jobs="$(detect_cpu_jobs)"
  if ! [[ "$cpu_jobs" =~ ^[0-9]+$ ]] || ((cpu_jobs < 1)); then
    cpu_jobs=4
  fi

  memory_budget="$(detect_memory_budget_bytes)"
  if [[ -z "$memory_budget" ]]; then
    echo "$cpu_jobs"
    return 0
  fi

  per_job="$(memory_per_job_bytes)"
  memory_jobs=$((memory_budget / per_job))
  if ((memory_jobs < 1)); then
    memory_jobs=1
  fi

  if ((memory_jobs < cpu_jobs)); then
    echo "capping llama.cpp build parallelism at $memory_jobs of $cpu_jobs CPUs" >&2
    echo "  backend:        $LLAMA_BACKEND" >&2
    echo "  memory budget:  $((memory_budget / 1024 / 1024)) MiB" >&2
    echo "  assumed per job: $((per_job / 1024 / 1024)) MiB" >&2
    echo "$memory_jobs"
    return 0
  fi

  echo "$cpu_jobs"
}

required_static_archives() {
  printf '%s\n' \
    "$LLAMA_BUILD_DIR/src/libllama.a" \
    "$LLAMA_BUILD_DIR/common/libllama-common.a" \
    "$LLAMA_BUILD_DIR/common/libllama-common-base.a" \
    "$LLAMA_BUILD_DIR/ggml/src/libggml.a" \
    "$LLAMA_BUILD_DIR/ggml/src/libggml-base.a" \
    "$LLAMA_BUILD_DIR/tools/mtmd/libmtmd.a"
}

required_static_archives_exist() {
  local archive
  while IFS= read -r archive; do
    [[ -f "$archive" ]] || return 1
  done < <(required_static_archives)
  [[ -f "$LLAMA_BUILD_DIR/ggml/src/libggml-cpu.a" ||
     -f "$LLAMA_BUILD_DIR/ggml/src/ggml-cpu/libggml-cpu.a" ]]
}

dynamic_library_name_groups() {
  case "$(uname -s)" in
    Darwin)
      printf '%s\n' libllama.dylib libllama-common.dylib libmtmd.dylib
      ;;
    MINGW*|MSYS*|CYGWIN*)
      # CMake's MinGW generator normally prefixes these DLLs with "lib";
      # retain the unprefixed MSVC-compatible spelling as an accepted
      # alternative because both are valid runtime package inputs.
      printf '%s\n' \
        'libllama.dll|llama.dll' \
        'libllama-common.dll|llama-common.dll' \
        'libmtmd.dll|mtmd.dll'
      ;;
    *)
      printf '%s\n' libllama.so libllama-common.so libmtmd.so
      ;;
  esac
}

required_dynamic_libraries_exist() {
  local candidates name found
  local -a names
  while IFS= read -r candidates; do
    IFS='|' read -r -a names <<< "$candidates"
    found=""
    for name in "${names[@]}"; do
      found="$(find "$LLAMA_BUILD_DIR" -name "$name" -print -quit)"
      [[ -z "$found" ]] || break
    done
    [[ -n "$found" && -e "$found" ]] || return 1
  done < <(dynamic_library_name_groups)
}

required_outputs_exist() {
  if [[ "$LLAMA_LINK_MODE" == "dynamic" ]]; then
    required_dynamic_libraries_exist
  else
    required_static_archives_exist
  fi
}

if [[ -z "${LLAMA_BUILD_DIR:-}" ]]; then
  if [[ -n "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR"
  elif [[ -n "${SKIPPY_LLAMA_BUILD_DIR:-}" ]]; then
    LLAMA_BUILD_DIR="$SKIPPY_LLAMA_BUILD_DIR"
  else
    LLAMA_BUILD_DIR="$(default_build_dir_for_backend)"
  fi
fi

if [[ "$PRINT_BUILD_DIR" == "1" ]]; then
  printf '%s\n' "$LLAMA_BUILD_DIR"
  exit 0
fi

if [[ "$PRINT_JOBS" == "1" ]]; then
  printf '%s\n' "$(detect_jobs)"
  exit 0
fi

if [[ ! -d "$LLAMA_WORKDIR/.git" ]]; then
  echo "llama checkout not found: $LLAMA_WORKDIR" >&2
  echo "run: just llama-prepare" >&2
  exit 1
fi

# The LunarG SDK ships GCC runtime DLLs in its Bin directories. GitHub Actions
# prepends those directories to PATH, so a MinGW-built vulkan-shaders-gen can
# load the SDK's older libstdc++ instead of the runtime matching its compiler
# and fail with STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139). Keep the selected
# compiler's runtime first; glslc remains discoverable later on PATH.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    if [[ "$LLAMA_BACKEND" == "vulkan" ]]; then
      MINGW_CXX="$(command -v g++ || true)"
      if [[ -n "$MINGW_CXX" ]]; then
        MINGW_RUNTIME_DIR="$(dirname "$MINGW_CXX")"
        export PATH="$MINGW_RUNTIME_DIR:$PATH"
        echo "prioritizing MinGW runtime for Vulkan build: $MINGW_RUNTIME_DIR"
      fi
    fi
    ;;
esac

SCCACHE_BIN="${SCCACHE:-${SCCACHE_PATH:-}}"
if [[ -z "$SCCACHE_BIN" ]] && command -v sccache >/dev/null 2>&1; then
  SCCACHE_BIN="$(command -v sccache)"
fi

CMAKE_ARGS=(
  -S "$LLAMA_WORKDIR"
  -B "$LLAMA_BUILD_DIR"
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  -DBUILD_SHARED_LIBS="$(if [[ "$LLAMA_LINK_MODE" == "dynamic" ]]; then echo ON; else echo OFF; fi)"
  -DGGML_NATIVE="${LLAMA_STAGE_GGML_NATIVE:-${SKIPPY_GGML_NATIVE:-OFF}}"
  -DLLAMA_OPENSSL=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_CURL=OFF
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  # mtmd video shells out to ffmpeg via sheredom/subprocess.h, which calls
  # posix_spawn_file_actions_addchdir_np - unavailable on iOS, so it breaks the
  # Swift XCFramework build. mesh-llm does not use mtmd video; opt out on all
  # platforms.
  -DMTMD_VIDEO=OFF
)

# Static ABI inputs cross job and runner boundaries. Normalize compiler-
# embedded source/build paths so the archived link closure does not retain a
# producer-local workspace path.
if [[ "$LLAMA_LINK_MODE" == "static" ]]; then
  PREFIX_MAP_FLAGS="-ffile-prefix-map=$ROOT=/mesh-llm -fdebug-prefix-map=$ROOT=/mesh-llm -fmacro-prefix-map=$ROOT=/mesh-llm"
  CMAKE_ARGS+=(
    "-DCMAKE_C_FLAGS=$PREFIX_MAP_FLAGS"
    "-DCMAKE_CXX_FLAGS=$PREFIX_MAP_FLAGS"
  )
fi

if command -v ninja >/dev/null 2>&1; then
  CMAKE_ARGS=(-G Ninja "${CMAKE_ARGS[@]}")
  echo "using CMake generator: Ninja"
fi

case "$LLAMA_BACKEND" in
  cuda)
    configure_cuda_toolkit_env || true
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
    if [[ -n "${CUDACXX:-}" ]]; then
      CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER="$CUDACXX")
    fi
    if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
      CMAKE_ARGS+=(-DCUDAToolkit_ROOT="$CUDAToolkit_ROOT")
    fi
    CUDA_ARCHITECTURES="${LLAMA_STAGE_CUDA_ARCHITECTURES:-${SKIPPY_CUDA_ARCHITECTURES:-}}"
    if [[ -n "$CUDA_ARCHITECTURES" ]]; then
      CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES")
    fi
    if [[ "${GGML_CUDA_NO_VMM:-}" == "1" ]]; then
      CMAKE_ARGS+=(-DGGML_CUDA_NO_VMM=ON)
    fi
    ;;
  rocm|hip)
    CMAKE_ARGS+=(-DGGML_HIP=ON)
    AMDGPU_TARGETS="${LLAMA_STAGE_AMDGPU_TARGETS:-${SKIPPY_AMDGPU_TARGETS:-}}"
    if [[ -n "$AMDGPU_TARGETS" ]]; then
      CMAKE_ARGS+=(-DAMDGPU_TARGETS="$AMDGPU_TARGETS")
    fi
    ;;
  vulkan)
    CMAKE_ARGS+=(-DGGML_VULKAN=ON)
    ;;
  metal)
    CMAKE_ARGS+=(-DGGML_METAL=ON)
    ;;
esac

USE_SCCACHE="${LLAMA_STAGE_USE_SCCACHE:-${SKIPPY_USE_SCCACHE:-1}}"
if [[ "$USE_SCCACHE" != "0" && -n "$SCCACHE_BIN" ]] &&
   ! "$SCCACHE_BIN" --show-stats >/dev/null 2>&1; then
  if ! "$SCCACHE_BIN" --start-server >/dev/null 2>&1 ||
     ! "$SCCACHE_BIN" --show-stats >/dev/null 2>&1; then
    if [[ "${MESH_LLM_REQUIRE_SCCACHE:-0}" == "1" ]]; then
      echo "sccache is unavailable and MESH_LLM_REQUIRE_SCCACHE=1" >&2
      exit 1
    fi
    echo "sccache is unavailable; llama.cpp build will run without compiler caching" >&2
    USE_SCCACHE=0
  fi
fi

if [[ "$USE_SCCACHE" != "0" && -n "$SCCACHE_BIN" ]]; then
  CMAKE_ARGS+=(
    -DCMAKE_C_COMPILER_LAUNCHER="$SCCACHE_BIN"
    -DCMAKE_CXX_COMPILER_LAUNCHER="$SCCACHE_BIN"
  )
  case "$LLAMA_BACKEND" in
    cuda)
      CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER_LAUNCHER="$SCCACHE_BIN")
      ;;
    rocm|hip)
      CMAKE_ARGS+=(-DCMAKE_HIP_COMPILER_LAUNCHER="$SCCACHE_BIN")
      ;;
  esac
  echo "using sccache for llama.cpp C/C++ compilation: $SCCACHE_BIN"
elif [[ "$USE_SCCACHE" != "0" ]]; then
  if [[ "${MESH_LLM_REQUIRE_SCCACHE:-0}" == "1" ]]; then
    echo "sccache is required but was not found" >&2
    exit 1
  fi
  echo "sccache not found; llama.cpp build will run without compiler caching" >&2
else
  CMAKE_ARGS+=(-DGGML_CCACHE=OFF)
fi

if [[ "$#" -gt 0 ]]; then
  CMAKE_ARGS+=("$@")
fi

PATCHED_SHA="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-patched-sha" 2>/dev/null || git -C "$LLAMA_WORKDIR" rev-parse HEAD)"
BUILD_STAMP="$LLAMA_BUILD_DIR/.mesh-llm-build-stamp"

normalize_build_stamp_arg() {
  local value="$1"
  value="${value//"$LLAMA_BUILD_DIR"/@LLAMA_BUILD_DIR@}"
  value="${value//"$LLAMA_WORKDIR"/@LLAMA_WORKDIR@}"
  value="${value//"$ROOT"/@MESH_LLM_ROOT@}"
  if [[ -n "$SCCACHE_BIN" ]]; then
    value="${value//"$SCCACHE_BIN"/@SCCACHE@}"
  fi
  printf '%s\n' "$value"
}

CURRENT_BUILD_STAMP="$(
  printf 'stamp-version=3\n'
  printf 'patched-sha=%s\n' "$PATCHED_SHA"
  printf 'backend=%s\n' "$LLAMA_BACKEND"
  printf 'link-mode=%s\n' "$LLAMA_LINK_MODE"
  printf 'toolchain-epoch=%s\n' "${MESH_LLM_LLAMA_TOOLCHAIN_EPOCH:-local}"
  printf 'build-type=%s\n' "${CMAKE_BUILD_TYPE:-Release}"
  printf 'ggml-native=%s\n' "${LLAMA_STAGE_GGML_NATIVE:-${SKIPPY_GGML_NATIVE:-OFF}}"
  printf 'cuda-architectures=%s\n' "${LLAMA_STAGE_CUDA_ARCHITECTURES:-${SKIPPY_CUDA_ARCHITECTURES:-}}"
  printf 'amdgpu-targets=%s\n' "${LLAMA_STAGE_AMDGPU_TARGETS:-${SKIPPY_AMDGPU_TARGETS:-}}"
  printf 'cuda-no-vmm=%s\n' "${GGML_CUDA_NO_VMM:-}"
  printf 'use-sccache=%s\n' "$USE_SCCACHE"
  for arg in "${CMAKE_ARGS[@]}"; do
    printf 'cmake-arg=%s\n' "$(normalize_build_stamp_arg "$arg")"
  done
)"

if [[ "${LLAMA_STAGE_FORCE_BUILD:-${SKIPPY_FORCE_LLAMA_BUILD:-0}}" != "1" &&
      -f "$BUILD_STAMP" &&
      "$(cat "$BUILD_STAMP")" == "$CURRENT_BUILD_STAMP" ]] &&
   git -C "$LLAMA_WORKDIR" diff-index --quiet HEAD -- &&
   required_outputs_exist; then
  echo "patched llama.cpp ABI already built"
  echo "  backend:   $LLAMA_BACKEND"
  echo "  build dir: $LLAMA_BUILD_DIR"
  exit 0
fi

if [[ "$REQUIRE_EXISTING" == "1" ]]; then
  echo "prebuilt patched llama.cpp ABI did not match the requested build contract" >&2
  echo "  backend:         $LLAMA_BACKEND" >&2
  echo "  link mode:       $LLAMA_LINK_MODE" >&2
  echo "  toolchain epoch: ${MESH_LLM_LLAMA_TOOLCHAIN_EPOCH:-local}" >&2
  echo "  build dir:       $LLAMA_BUILD_DIR" >&2
  echo "refusing to rebuild because --require-existing was set" >&2
  exit 1
fi

cmake "${CMAKE_ARGS[@]}"

cmake --build "$LLAMA_BUILD_DIR" --config "${CMAKE_BUILD_TYPE:-Release}" --parallel "$(detect_jobs)" --target llama llama-common mtmd

printf '%s\n' "$CURRENT_BUILD_STAMP" > "$BUILD_STAMP"
if ! required_outputs_exist; then
  echo "patched llama.cpp build completed without the full link closure" >&2
  exit 1
fi

echo "built patched llama.cpp"
echo "  backend:   $LLAMA_BACKEND"
echo "  build dir: $LLAMA_BUILD_DIR"
