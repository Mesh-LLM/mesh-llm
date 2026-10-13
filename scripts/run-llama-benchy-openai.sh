#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:9337/v1}"
API_KEY="${API_KEY:-EMPTY}"
LLAMA_BENCHY_FROM="${LLAMA_BENCHY_FROM:-git+https://github.com/eugr/llama-benchy}"

MODEL="${MODEL:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
TOKENIZER="${TOKENIZER:-}"

PP="${PP:-128}"
TG="${TG:-16}"
DEPTH="${DEPTH:-0}"
RUNS="${RUNS:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
LATENCY_MODE="${LATENCY_MODE:-generation}"
FORMAT="${FORMAT:-md}"
SAVE_RESULT="${SAVE_RESULT:-}"
SKIP_COHERENCE="${SKIP_COHERENCE:-1}"
NO_ADAPT_PROMPT="${NO_ADAPT_PROMPT:-0}"
NO_CACHE="${NO_CACHE:-0}"

discover_model() {
  local -a automation=(cargo xtool)
  if [[ "${MESH_LLM_AUTOMATION_BIN+set}" == set ]]; then
    if [[ "$MESH_LLM_AUTOMATION_BIN" != /* || ! -f "$MESH_LLM_AUTOMATION_BIN" || ! -x "$MESH_LLM_AUTOMATION_BIN" ]]; then
      echo "MESH_LLM_AUTOMATION_BIN must be an absolute regular executable" >&2
      return 1
    fi
    automation=("$MESH_LLM_AUTOMATION_BIN")
  fi
  API_KEY="$API_KEY" "${automation[@]}" automation endpoint-model-discovery --base-url "$BASE_URL" --timeout-secs 10
}

if [[ -z "$MODEL" ]]; then
  MODEL="$(discover_model)"
fi

if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="$MODEL"
fi

read -r -a pp_values <<<"$PP"
read -r -a tg_values <<<"$TG"
read -r -a depth_values <<<"$DEPTH"
read -r -a concurrency_values <<<"$CONCURRENCY"

cmd=(
  uvx
  --from "$LLAMA_BENCHY_FROM"
  llama-benchy
  --base-url "$BASE_URL"
)
# The key display slot is owned by construction, never by parsing caller values.
api_key_index="$(( ${#cmd[@]} + 1 ))"
cmd+=(
  --api-key "$API_KEY"
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --pp "${pp_values[@]}"
  --tg "${tg_values[@]}"
  --depth "${depth_values[@]}"
  --runs "$RUNS"
  --concurrency "${concurrency_values[@]}"
  --latency-mode "$LATENCY_MODE"
  --format "$FORMAT"
)

if [[ -n "$TOKENIZER" ]]; then
  cmd+=(--tokenizer "$TOKENIZER")
fi
if [[ -n "$SAVE_RESULT" ]]; then
  cmd+=(--save-result "$SAVE_RESULT")
fi
if [[ "$SKIP_COHERENCE" == "1" ]]; then
  cmd+=(--skip-coherence)
fi
if [[ "$NO_ADAPT_PROMPT" == "1" ]]; then
  cmd+=(--no-adapt-prompt)
fi
if [[ "$NO_CACHE" == "1" ]]; then
  cmd+=(--no-cache)
fi

printf 'Running:'
# External benchy still requires its existing --api-key argv; never echo it.
display_cmd=("${cmd[@]}")
display_cmd[api_key_index]='<redacted>'
printf ' %q' "${display_cmd[@]}"
printf '\n'
exec "${cmd[@]}"
