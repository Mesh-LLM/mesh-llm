#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH_BIN="${MESH_BIN:-$ROOT/target/debug/mesh-llm}"
SINGLE_REF="${SINGLE_REF:-unsloth/Qwen2.5-Coder-0.5B-Instruct-GGUF:Q4_K_M}"
MULTIPART_REF="${MULTIPART_REF:-Felladrin/gguf-sharded-Qwen1.5-0.5B-Chat:Q3_K_M.shard}"
RUN_REGRESSION=1
DRY_RUN=0

usage() {
  cat >&2 <<'EOF'
usage: scripts/qa-model-download-stats.sh [--dry-run] [--skip-regression]

Runs manual QA for model download transfer stats using the debug binary.

Default refs:
  SINGLE_REF     unsloth/Qwen2.5-Coder-0.5B-Instruct-GGUF:Q4_K_M
  MULTIPART_REF  Felladrin/gguf-sharded-Qwen1.5-0.5B-Chat:Q3_K_M.shard

Environment:
  MESH_BIN       path to debug binary; default target/debug/mesh-llm
  SINGLE_REF     override the single-file GGUF ref
  MULTIPART_REF  override the multipart GGUF ref

Examples:
  just build
  scripts/qa-model-download-stats.sh --dry-run
  scripts/qa-model-download-stats.sh
  SINGLE_REF='Qwen/Qwen1.5-0.5B-Chat-GGUF:q2_k' scripts/qa-model-download-stats.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-regression)
      RUN_REGRESSION=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -x "$MESH_BIN" ]]; then
  echo "debug binary not found or not executable: $MESH_BIN" >&2
  echo "run: just build" >&2
  exit 1
fi

LOG_DIR="${TMPDIR:-/tmp}/mesh-llm-download-qa-$(date +%Y%m%d-%H%M%S)"
HF_CACHE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mesh-llm-download-qa-hf.XXXXXX")"
export HF_HOME="$HF_CACHE_ROOT/home"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="$HF_CACHE_ROOT/xet"

cleanup_hf_cache() {
  rm -rf -- "$HF_CACHE_ROOT"
}

reset_hf_cache() {
  rm -rf -- "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE"
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE"
}

trap cleanup_hf_cache EXIT

run_cmd() {
  local label="$1"
  shift

  echo ""
  echo "=== $label ==="
  printf 'command:'
  printf ' %q' "$@"
  echo ""

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  mkdir -p "$LOG_DIR"
  local log_file="$LOG_DIR/${label//[^A-Za-z0-9_.-]/_}.log"
  "$@" 2>&1 | tee "$log_file"
}

show_ref() {
  local label="$1"
  local ref="$2"
  run_cmd "show $label ref" "$MESH_BIN" models show "$ref" --json
}

download_ref() {
  local label="$1"
  local ref="$2"
  shift 2
  run_cmd "$label download" "$MESH_BIN" models download "$ref" "$@"
  run_cmd "$label json download" "$MESH_BIN" models download "$ref" "$@" --json
}

echo "=== mesh-llm model download stats QA ==="
echo "  binary:        $MESH_BIN"
echo "  version:       $($MESH_BIN --version)"
echo "  single ref:    $SINGLE_REF"
echo "  multipart ref: $MULTIPART_REF"
echo "  hf home:       $HF_HOME"
echo "  hf hub cache:  $HF_HUB_CACHE"
echo "  hf xet cache:  $HF_XET_CACHE"
if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "  logs:          $LOG_DIR"
fi

show_ref "single" "$SINGLE_REF"
show_ref "multipart" "$MULTIPART_REF"

echo ""
echo "=== reset temporary Hugging Face cache before fetch phase ==="
if [[ "$DRY_RUN" -eq 0 ]]; then
  reset_hf_cache
else
  printf 'command: rm -rf %q %q %q && mkdir -p %q %q %q\n' \
    "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" \
    "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE"
fi

download_ref "single cold" "$SINGLE_REF"
download_ref "single cache-hit" "$SINGLE_REF"

download_ref "multipart cold" "$MULTIPART_REF" --direct
download_ref "multipart cache-hit" "$MULTIPART_REF" --direct

if [[ "$RUN_REGRESSION" -eq 1 ]]; then
  run_cmd "host-runtime transfer stats tests" \
    cargo test -p mesh-llm-host-runtime download_transfer_stats --lib
  run_cmd "host-runtime silent progress tests" \
    cargo test -p mesh-llm-host-runtime silent_download_progress_records_transfer_stats --lib
  run_cmd "host-runtime progress state tests" \
    cargo test -p mesh-llm-host-runtime download_progress_state --lib
  run_cmd "mesh-llm download headline tests" \
    cargo test -p mesh-llm downloaded_model_headline --lib
  run_cmd "mesh-llm cargo check" cargo check -p mesh-llm
fi

cleanup_hf_cache
trap - EXIT

echo ""
echo "QA complete. Inspect logs under: $LOG_DIR"
echo "Temporary Hugging Face cache cleaned up: $HF_CACHE_ROOT"
echo "Expected cache-hit behavior: no synthetic byte count or speed in the final summary."
echo "Expected multipart behavior: transferred bytes/speed aggregate across required shards that moved bytes."
