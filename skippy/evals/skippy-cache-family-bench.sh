#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
automation=(just --justfile "$ROOT/Justfile" automation-run)
if [[ "${MESH_LLM_AUTOMATION_BIN+set}" == set ]]; then
  if [[ "$MESH_LLM_AUTOMATION_BIN" != /* || ! -f "$MESH_LLM_AUTOMATION_BIN" || ! -x "$MESH_LLM_AUTOMATION_BIN" ]]; then
    echo 'MESH_LLM_AUTOMATION_BIN must be an absolute executable' >&2
    exit 1
  fi
  automation=("$MESH_LLM_AUTOMATION_BIN")
fi
: "${SKIPPY_CACHE_OPERATOR_INPUT:?Set SKIPPY_CACHE_OPERATOR_INPUT to an absolute native cache operator JSON file}"
if [[ "$SKIPPY_CACHE_OPERATOR_INPUT" != /* || ! -f "$SKIPPY_CACHE_OPERATOR_INPUT" ]]; then
  echo 'SKIPPY_CACHE_OPERATOR_INPUT must be an absolute regular input file' >&2
  exit 1
fi
RUN_ID="${SKIPPY_CACHE_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${1:-/tmp/skippy-cache-family-bench-${RUN_ID}}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
if [[ "${SKIPPY_CACHE_SKIP_BUILD:-0}" != 1 ]]; then
  (cd "$ROOT" && just skippy-workload-oracles-build "${SKIPPY_CACHE_BUILD_OUTPUT:-${OUTPUT_DIR}/tools}")
fi
USECASE_CORPUS="${SKIPPY_CACHE_USECASE_CORPUS:-${ROOT}/skippy/evals/skippy-usecase-corpus.json}"
COMMON_ARGS=(--use-case-corpus "$USECASE_CORPUS" --prefix-tokens "${PREFIX_TOKENS:-128}" --runtime-lane-count "${RUNTIME_LANE_COUNT:-1}" --llama-parallel "${LLAMA_PARALLEL:-1}" --llama-repeats "${LLAMA_REPEATS:-3}" --cache-hit-repeats "${CACHE_HIT_REPEATS:-3}")
"${automation[@]}" automation cache-family-run prepare-full --input "$SKIPPY_CACHE_OPERATOR_INPUT" --output "$OUTPUT_DIR/full-input" "${COMMON_ARGS[@]}"
"${automation[@]}" automation cache-family-run --input "$OUTPUT_DIR/full-input/cache-family-input.json" --output "$OUTPUT_DIR/full-gguf"
"${automation[@]}" automation cache-family-run prepare-use-cases --input "$SKIPPY_CACHE_OPERATOR_INPUT" --output "$OUTPUT_DIR/usecase-input" "${COMMON_ARGS[@]}"
"${automation[@]}" automation cache-family-run --input "$OUTPUT_DIR/usecase-input/cache-family-input.json" --output "$OUTPUT_DIR/use-cases"
"${automation[@]}" automation cache-family-report --input "$OUTPUT_DIR/full-gguf/production-cache-bench.json" --input "$OUTPUT_DIR/use-cases/production-cache-bench.json" --use-case-corpus "$USECASE_CORPUS" --output "$OUTPUT_DIR/readme-tables.md"
printf 'Wrote raw full-GGUF results: %s\n' "$OUTPUT_DIR/full-gguf/production-cache-bench.json"
printf 'Wrote raw use-case results: %s\n' "$OUTPUT_DIR/use-cases/production-cache-bench.json"
printf 'Wrote README tables: %s\n' "$OUTPUT_DIR/readme-tables.md"
