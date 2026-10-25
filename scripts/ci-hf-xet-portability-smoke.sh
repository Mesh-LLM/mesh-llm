#!/usr/bin/env bash
# ci-hf-xet-portability-smoke.sh — exercise the release binary's Xet HTTPS path.
#
# The fixture is a small metadata GGUF from the MeshLLM layer-package catalog.
# Hugging Face serves it through the Xet bridge, so this checks the same TLS
# provider path as a large model download without fetching a multi-gigabyte
# artifact.
#
# Usage:
#   scripts/ci-hf-xet-portability-smoke.sh /path/to/mesh-llm

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mesh-llm" >&2
  exit 2
fi

binary="$1"
if [[ ! -x "$binary" ]]; then
  echo "mesh-llm binary is not executable: $binary" >&2
  exit 2
fi

smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/mesh-llm-hf-xet-smoke.XXXXXX")"
trap 'rm -rf -- "$smoke_root"' EXIT

export HF_HOME="$smoke_root/hf-home"
export HF_HUB_CACHE="$smoke_root/hub-cache"
export HF_XET_CACHE="$smoke_root/xet-cache"
export MESH_LLM_DATA_DIR="$smoke_root/mesh-data"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$MESH_LLM_DATA_DIR"

fixture_ref="meshllm/gemma-4-26B-A4B-it-UD-Q4_K_M-layers/shared/metadata.gguf"
output_file="$smoke_root/download-output"

echo "=== MeshLLM Hugging Face/Xet portability smoke ==="
echo "  binary:  $binary"
echo "  fixture: $fixture_ref"
echo "  cache:   $smoke_root"
echo "  architecture: $(uname -m)"
if [[ -r /proc/cpuinfo ]]; then
  while IFS= read -r cpu_line; do
    case "$cpu_line" in
      Features* | flags*)
        echo "  cpu $cpu_line"
        break
        ;;
    esac
  done </proc/cpuinfo
elif command -v sysctl >/dev/null 2>&1; then
  echo "  arm_sha512_sysctl: $(sysctl -n hw.optional.armv8_2_sha512 2>/dev/null || echo unavailable)"
fi

command=("$binary" --log-format json models download --direct --json "$fixture_ref")
if ! command -v timeout >/dev/null 2>&1; then
  echo "::warning::Hugging Face/Xet advisory smoke skipped: timeout is unavailable" >&2
  exit 0
fi

error_file="$smoke_root/download-error"
download_succeeded=false
for attempt in 1 2 3; do
  echo "Hugging Face/Xet advisory smoke attempt $attempt/3" >&2
  set +e
  timeout --kill-after=10s 180s "${command[@]}" >"$output_file" 2>"$error_file"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    download_succeeded=true
    break
  fi
  if [[ $status -eq 132 ]]; then
    cat "$error_file" >&2
    echo "Hugging Face/Xet portability smoke failed with SIGILL" >&2
    exit 132
  fi
  if [[ $attempt -lt 3 ]]; then
    sleep "$attempt"
  fi
done

if [[ $download_succeeded != true ]]; then
  cat "$error_file" >&2
  echo "::warning::Hugging Face/Xet advisory smoke skipped after three failed live download attempts (last status: $status)" >&2
  exit 0
fi

automation=(cargo xtool)
if [[ -n "${MESH_LLM_AUTOMATION_BIN:-}" ]]; then
  automation=("$MESH_LLM_AUTOMATION_BIN")
fi
"${automation[@]}" automation hf-xet-smoke "$output_file" "$smoke_root"
