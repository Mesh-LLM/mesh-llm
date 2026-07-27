#!/usr/bin/env bash

# Resolve CUDA through the compiler's canonical path. Distribution packages may
# expose nvcc through a symlink such as /usr/local/bin/nvcc while the toolkit
# headers and libraries live under /usr/local/cuda/targets/<triple>. CMake can
# otherwise mistake /usr/local for the toolkit root.

cuda_canonical_path() {
  local path="$1"

  if command -v readlink >/dev/null 2>&1; then
    readlink -f "$path" 2>/dev/null && return 0
  fi
  if command -v realpath >/dev/null 2>&1; then
    realpath "$path" 2>/dev/null && return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$path" && return 0
  fi
  if command -v python >/dev/null 2>&1; then
    python -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$path" && return 0
  fi

  printf '%s\n' "$path"
}

cuda_toolkit_root_for_compiler() {
  local compiler="$1"
  local resolved=""
  local root=""

  [[ -n "$compiler" ]] || return 1
  resolved="$(cuda_canonical_path "$compiler")"
  root="$(cd "$(dirname "$resolved")/.." 2>/dev/null && pwd -P)" || return 1

  if [[ -f "$root/include/cuda_runtime.h" ]] ||
     compgen -G "$root/targets/*/include/cuda_runtime.h" >/dev/null; then
    printf '%s\n' "$root"
    return 0
  fi

  return 1
}

cuda_toolkit_library_dir() {
  local root="$1"
  local candidate=""

  for candidate in "$root/lib64" "$root/lib" "$root"/targets/*/lib; do
    if [[ -f "$candidate/libcudart.so" ]] &&
       [[ -f "$candidate/libcublas.so" ]] &&
       [[ -f "$candidate/libcublasLt.so" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

cuda_prepend_path_var() {
  local name="$1"
  local value="$2"
  local current="${!name:-}"

  case ":$current:" in
    *":$value:"*) ;;
    *)
      printf -v "$name" '%s' "${value}${current:+:$current}"
      export "$name"
      ;;
  esac
}

configure_cuda_toolkit_env() {
  local compiler="${CUDACXX:-}"
  local root="${CUDAToolkit_ROOT:-}"
  local library_dir=""

  if [[ -z "$compiler" ]]; then
    compiler="$(command -v nvcc 2>/dev/null || true)"
    [[ -n "$compiler" ]] || return 1
    compiler="$(cuda_canonical_path "$compiler")"
    export CUDACXX="$compiler"
  fi

  if [[ -z "$root" ]]; then
    root="$(cuda_toolkit_root_for_compiler "$compiler" || true)"
    if [[ -n "$root" ]]; then
      export CUDAToolkit_ROOT="$root"
    fi
  fi

  export NVCC="${NVCC:-$compiler}"
  cuda_prepend_path_var PATH "$(dirname "$compiler")"

  if [[ -n "$root" ]]; then
    export CUDA_HOME="${CUDA_HOME:-$root}"
    export CUDA_PATH="${CUDA_PATH:-$root}"
    library_dir="$(cuda_toolkit_library_dir "$root" || true)"
    if [[ -n "$library_dir" ]]; then
      export CUDA_LIBRARY_PATH="${CUDA_LIBRARY_PATH:-$library_dir}"
      cuda_prepend_path_var LIBRARY_PATH "$library_dir"
      cuda_prepend_path_var LD_LIBRARY_PATH "$library_dir"
    fi
  fi
  return 0
}
