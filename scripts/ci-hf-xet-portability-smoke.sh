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

command=("$binary" --log-format json models download --direct --json "$fixture_ref")
if command -v timeout >/dev/null 2>&1; then
  timeout 180 "${command[@]}" >"$output_file"
else
  echo "timeout command is required for the bounded Hugging Face/Xet smoke" >&2
  exit 2
fi

python3 - "$output_file" "$smoke_root" <<'PY'
import json
import os
from pathlib import Path
import sys

output_path = Path(sys.argv[1])
smoke_root = Path(sys.argv[2]).resolve()
output = output_path.read_text(encoding="utf-8")

# --log-format json emits lifecycle records before the command payload. Find
# the payload by its stable path field instead of relying on line ordering.
payload = None
decoder = json.JSONDecoder()
for index, character in enumerate(output):
    if character != "{":
        continue
    try:
        candidate, _ = decoder.raw_decode(output[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict) and isinstance(candidate.get("path"), str):
        payload = candidate
        break

if payload is None:
    raise SystemExit(f"download output did not contain a JSON path payload:\n{output}")

downloaded = Path(payload["path"]).resolve()
try:
    downloaded.relative_to(smoke_root)
except ValueError as error:
    raise SystemExit(
        f"download escaped the isolated smoke cache: {downloaded} (root {smoke_root})"
    ) from error

if not downloaded.is_file():
    raise SystemExit(f"downloaded path is not a file: {downloaded}")

size = downloaded.stat().st_size
max_size = 64 * 1024 * 1024
if not 0 < size <= max_size:
    raise SystemExit(f"fixture size {size} is outside the 1..{max_size} byte bound")

print(f"Xet portability smoke passed: {downloaded} ({size} bytes)")
PY
