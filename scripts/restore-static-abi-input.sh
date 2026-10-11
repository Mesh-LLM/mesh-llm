#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "Usage: $0 <artifact-download-dir> <build-dir> <target> <backend>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
automation=(cargo run --quiet --manifest-path "$REPO_ROOT/tools/xtask/Cargo.toml" --)
if [[ -n "${MESH_LLM_AUTOMATION_BIN:-}" ]]; then
    automation=("$MESH_LLM_AUTOMATION_BIN")
fi
download_dir="$1"
build_dir="$2"
expected_target="$3"
expected_backend="$4"
expected_basename="build-stage-abi-static"
expected_toolchain_epoch="${MESH_LLM_LLAMA_TOOLCHAIN_EPOCH:-}"
prepared_sha_file="${LLAMA_WORKDIR:-$REPO_ROOT/.deps/llama.cpp}/.mesh-llm-patched-sha"

if [[ -z "$expected_toolchain_epoch" ||
      ! "$expected_toolchain_epoch" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "MESH_LLM_LLAMA_TOOLCHAIN_EPOCH must identify the pinned build image" >&2
    exit 1
fi

case "$expected_target" in
    x86_64-unknown-linux-gnu) expected_arch="x86_64" ;;
    aarch64-unknown-linux-gnu) expected_arch="aarch64" ;;
    *)
        echo "unsupported static ABI target: $expected_target" >&2
        exit 1
        ;;
esac
case "$(uname -m)" in
    x86_64|amd64) runner_arch="x86_64" ;;
    aarch64|arm64) runner_arch="aarch64" ;;
    *) runner_arch="$(uname -m)" ;;
esac
if [[ "$runner_arch" != "$expected_arch" ]]; then
    echo "static ABI target/runner architecture mismatch: $expected_target on $runner_arch" >&2
    exit 1
fi
if [[ ! -d "$download_dir" || -L "$download_dir" ]]; then
    echo "static ABI artifact download directory is invalid: $download_dir" >&2
    exit 1
fi
if [[ "$(basename "$build_dir")" != "$expected_basename" ]]; then
    echo "static ABI destination must end in $expected_basename: $build_dir" >&2
    exit 1
fi
if [[ -e "$build_dir" || -L "$build_dir" ]]; then
    echo "static ABI destination must not already exist: $build_dir" >&2
    exit 1
fi

shopt -s nullglob dotglob
download_entries=("$download_dir"/*)
archive="$download_dir/mesh-llm-static-abi.tar.gz"
checksum="$archive.sha256"
if [[ "${#download_entries[@]}" -ne 2 ||
      ! -s "$archive" ||
      ! -s "$checksum" ]]; then
    echo "static ABI input must contain exactly its archive and checksum" >&2
    exit 1
fi
"${automation[@]}" artifact verify-checksum "$archive"

extract_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/mesh-static-abi.XXXXXX")"
trap 'rm -rf -- "$extract_root"' EXIT
"${automation[@]}" artifact extract-tar \
    "$archive" \
    "$extract_root"

extract_entries=("$extract_root"/*)
if [[ "${#extract_entries[@]}" -ne 1 ||
      ! -d "${extract_entries[0]}" ||
      -L "${extract_entries[0]}" ||
      "$(basename "${extract_entries[0]}")" != "$expected_basename" ]]; then
    echo "static ABI archive must contain exactly $expected_basename" >&2
    exit 1
fi
restored_dir="${extract_entries[0]}"
manifest="$restored_dir/.mesh-llm-static-abi-input.json"
build_stamp="$restored_dir/.mesh-llm-build-stamp"
cmake_cache="$restored_dir/CMakeCache.txt"
required_archives=(
    "$restored_dir/src/libllama.a"
    "$restored_dir/common/libllama-common.a"
    "$restored_dir/common/libllama-common-base.a"
    "$restored_dir/ggml/src/libggml.a"
    "$restored_dir/ggml/src/libggml-base.a"
    "$restored_dir/tools/mtmd/libmtmd.a"
    "$restored_dir/vendor/hash/libvendor-hash.a"
)
test -s "$manifest"
test -s "$build_stamp"
test -f "$cmake_cache"
for archive in "${required_archives[@]}"; do
    if [[ ! -s "$archive" ]]; then
        echo "static ABI is missing required archive: $archive" >&2
        exit 1
    fi
done
if [[ ! -s "$restored_dir/ggml/src/libggml-cpu.a" &&
      ! -s "$restored_dir/ggml/src/ggml-cpu/libggml-cpu.a" ]]; then
    echo "static ABI is missing libggml-cpu.a" >&2
    exit 1
fi

"${automation[@]}" prepared-input static-abi-manifest verify \
    "$manifest" \
    "$build_stamp" \
    "$expected_target" \
    "$expected_backend" \
    "$expected_basename" \
    "$expected_toolchain_epoch"
"${automation[@]}" prepared-input static-abi-stamp \
    "$build_stamp" \
    --backend "$expected_backend" \
    --link-mode static \
    --stamp-version 3 \
    --toolchain-epoch "$expected_toolchain_epoch"

if [[ ! -s "$prepared_sha_file" ]]; then
    echo "prepared llama.cpp patched SHA is missing: $prepared_sha_file" >&2
    exit 1
fi
prepared_sha="$(tr -d '[:space:]' < "$prepared_sha_file")"
if [[ ! "$prepared_sha" =~ ^[0-9a-f]{40}$ ]] ||
   ! grep -Fxq "patched-sha=$prepared_sha" "$build_stamp"; then
    echo "static ABI patched SHA does not match prepared llama.cpp" >&2
    exit 1
fi

mkdir -p "$(dirname "$build_dir")"
cp -a "$restored_dir" "$build_dir"
test -d "$build_dir"
