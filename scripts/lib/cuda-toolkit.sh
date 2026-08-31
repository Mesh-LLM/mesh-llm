#!/usr/bin/env bash

# Resolve CUDA through the compiler's canonical path. Distribution packages may
# expose nvcc through a symlink such as /usr/local/bin/nvcc while the toolkit
# headers and libraries live under /usr/local/cuda/targets/<triple>. CMake can
# otherwise mistake /usr/local for the toolkit root.

_mesh_cuda_toolkit_manifest_major_cache=""

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

# Git Bash's `command -v nvcc` can report a Windows CUDA executable without
# its `.exe` suffix. The shell can execute that path through PATHEXT, but CMake
# requires the explicit existing compiler path when CUDACXX is set.
cuda_cmake_compiler_path() {
  local compiler="$1"

  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      case "$compiler" in
        *.[Ee][Xx][Ee]) ;;
        *)
          if [[ -f "${compiler}.exe" ]]; then
            printf '%s\n' "${compiler}.exe"
            return 0
          fi
          ;;
      esac
      ;;
  esac

  printf '%s\n' "$compiler"
}

# Resolve the CUDA compiler selected by the build. CUDACXX is the CMake
# spelling, CMAKE_CUDA_COMPILER is accepted for callers that pass the CMake
# cache variable through the environment, and NVCC is retained for the
# benchmark helper and Windows tooling. An explicit selector is authoritative:
# if it is set but cannot be resolved, do not silently fall back to PATH.
cuda_selected_compiler() {
  local requested=""
  local resolved=""

  if [[ -n "${CUDACXX:-}" ]]; then
    requested="$CUDACXX"
  elif [[ -n "${CMAKE_CUDA_COMPILER:-}" ]]; then
    requested="$CMAKE_CUDA_COMPILER"
  elif [[ -n "${NVCC:-}" ]]; then
    requested="$NVCC"
  else
    requested="nvcc"
  fi

  if [[ "$requested" == */* || "$requested" == *\\* ]]; then
    if [[ -x "$requested" ]]; then
      resolved="$requested"
    elif [[ -x "${requested}.exe" ]]; then
      resolved="${requested}.exe"
    else
      return 1
    fi
  else
    resolved="$(command -v "$requested" 2>/dev/null || true)"
    [[ -n "$resolved" && -x "$resolved" ]] || return 1
  fi

  resolved="$(cuda_cmake_compiler_path "$resolved")"
  [[ -x "$resolved" ]] || return 1
  cuda_canonical_path "$resolved"
}

cuda_toolkit_version_from_compiler() {
  local compiler="${1:-}"
  local output=""
  local version=""

  [[ -n "$compiler" ]] || return 1
  output="$("$compiler" --version 2>&1)" || return 1
  version="$(printf '%s\n' "$output" | sed -nE \
    's/.*[Rr]elease[[:space:]]+([0-9]+\.[0-9]+).*/\1/p' | head -n 1)"
  [[ -n "$version" ]] || return 1
  printf '%s\n' "$version"
}

cuda_toolkit_version_from_root() {
  local root="$1"
  local version=""

  [[ -n "$root" && -d "$root" ]] || return 1
  if [[ -f "$root/version.json" ]]; then
    version="$(sed -nE \
      's/.*"version"[[:space:]]*:[[:space:]]*"([0-9]+\.[0-9]+).*/\1/p' \
      "$root/version.json" | head -n 1)"
    [[ -n "$version" ]] && { printf '%s\n' "$version"; return 0; }
  fi
  if [[ -f "$root/version.txt" ]]; then
    version="$(sed -nE \
      's/.*CUDA Version[[:space:]]+([0-9]+\.[0-9]+).*/\1/p' \
      "$root/version.txt" | head -n 1)"
    [[ -n "$version" ]] && { printf '%s\n' "$version"; return 0; }
  fi
  return 1
}

# Return the installed CUDA toolkit version. A selected compiler is preferred
# because it is the executable CMake will use. If no compiler selector resolves,
# fall back only to toolkit-owned version metadata; driver-reported capability
# is deliberately not considered build-toolkit evidence.
cuda_toolkit_version() {
  local compiler=""
  local version=""
  local root=""

  # An explicit compiler selector must be the source of truth. If it is set
  # but invalid, do not let an unrelated toolkit symlink or metadata mask the
  # build failure.
  if [[ -n "${CUDACXX:-}" || -n "${CMAKE_CUDA_COMPILER:-}" ||
        -n "${NVCC:-}" ]]; then
    compiler="$(cuda_selected_compiler)" || return 1
    version="$(cuda_toolkit_version_from_compiler "$compiler" || true)"
    [[ -n "$version" ]] || return 1
    printf '%s\n' "$version"
    return 0
  fi

  if compiler="$(cuda_selected_compiler)"; then
    version="$(cuda_toolkit_version_from_compiler "$compiler" || true)"
    if [[ -n "$version" ]]; then
      printf '%s\n' "$version"
      return 0
    fi
    return 1
  fi

  for root in \
    "${CUDAToolkit_ROOT:-}" \
    "${CUDA_HOME:-}" \
    "${CUDA_PATH:-}" \
    /usr/local/cuda \
    /opt/cuda; do
    version="$(cuda_toolkit_version_from_root "$root" || true)"
    if [[ -n "$version" ]]; then
      printf '%s\n' "$version"
      return 0
    fi
  done

  return 1
}

cuda_toolkit_manifest_version() {
  local detected_version=""
  local detected_major=""
  local declared_major="${MESH_LLM_CUDA_TOOLKIT_MAJOR:-}"
  local declared_version="${MESH_CUDA_VERSION:-}"
  local declared_version_major=""
  local declared_version_minor=""
  local declared_version_major_minor=""

  if [[ -n "$declared_major" && ! "$declared_major" =~ ^[0-9]+$ ]]; then
    echo "MESH_LLM_CUDA_TOOLKIT_MAJOR must be digits-only (for example: 12)" >&2
    return 1
  fi
  if [[ -n "$declared_version" ]]; then
    if [[ ! "$declared_version" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]; then
      echo "MESH_CUDA_VERSION must be a numeric CUDA version (for example: 12.9.2)" >&2
      return 1
    fi
    declared_version_major="${declared_version%%.*}"
    if [[ "$declared_version" == *.* ]]; then
      declared_version_minor="${declared_version#*.}"
      declared_version_minor="${declared_version_minor%%.*}"
      declared_version_major_minor="${declared_version_major}.${declared_version_minor}"
    fi
  fi
  if [[ -n "$declared_major" && -n "$declared_version_major" &&
        "$declared_major" != "$declared_version_major" ]]; then
    printf '%s\n' \
      "MESH_LLM_CUDA_TOOLKIT_MAJOR=$declared_major does not match MESH_CUDA_VERSION=$declared_version" >&2
    return 1
  fi

  if ! detected_version="$(cuda_toolkit_version)"; then
    echo "CUDA toolkit version could not be detected; set CUDACXX/NVCC, put nvcc " \
      "on PATH, or provide CUDAToolkit_ROOT/version metadata (version.json/version.txt)." >&2
    return 1
  fi
  detected_major="${detected_version%%.*}"

  if [[ -n "$declared_major" && "$declared_major" != "$detected_major" ]]; then
    printf '%s\n' \
      "declared CUDA toolkit major $declared_major does not match the selected CUDA compiler/toolkit major $detected_major" >&2
    return 1
  fi
  if [[ -n "$declared_version_major" &&
        "$declared_version_major" != "$detected_major" ]]; then
    printf '%s\n' \
      "MESH_CUDA_VERSION=$declared_version does not match the selected CUDA compiler/toolkit major $detected_major" >&2
    return 1
  fi
  if [[ -n "$declared_version_major_minor" &&
        "$declared_version_major_minor" != "$detected_version" ]]; then
    printf '%s\n' \
      "MESH_CUDA_VERSION=$declared_version does not match the selected CUDA compiler/toolkit version $detected_version (major.minor)" >&2
    return 1
  fi

  if [[ -n "$declared_version" ]]; then
    printf '%s\n' "$declared_version"
  else
    printf '%s\n' "$detected_version"
  fi
}

cuda_toolkit_manifest_major() {
  local version=""

  if ! version="$(cuda_toolkit_manifest_version)"; then
    return 1
  fi
  printf '%s\n' "${version%%.*}"
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
      # shellcheck disable=SC2163
      export "$name"
      ;;
  esac
}

configure_cuda_toolkit_env() {
  local compiler=""
  local root="${CUDAToolkit_ROOT:-}"
  local library_dir=""

  compiler="$(cuda_selected_compiler)" || return 1
  export CUDACXX="$compiler"
  export NVCC="$compiler"

  if [[ -z "$root" ]]; then
    root="$(cuda_toolkit_root_for_compiler "$compiler" || true)"
    if [[ -n "$root" ]]; then
      export CUDAToolkit_ROOT="$root"
    fi
  fi

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
