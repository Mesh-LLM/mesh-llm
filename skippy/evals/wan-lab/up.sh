#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAB_DIR="${ROOT}/skippy/evals/wan-lab"
ENV_FILE="${ENV_FILE:-${LAB_DIR}/.env}"
LINK_ENV_FILE="${LINK_ENV_FILE:-${LAB_DIR}/.env.link}"

log() {
  printf '[skippy-wan-lab-up] %s\n' "$*" >&2
}

load_env_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$file"
    set +a
  fi
}

parse_hf_package_ref() {
  just --justfile "${ROOT}/Justfile" skippy-package-reference "$1"
}
hf_home_dir() {
  if [[ -n "${HF_HOME:-}" ]]; then
    printf '%s\n' "$HF_HOME"
  else
    printf '%s\n' "${HOME}/.cache/huggingface"
  fi
}

hf_snapshot_dir() {
  local result
  result="$(just --justfile "${ROOT}/Justfile" skippy-layer-package-cache \
    --reference "hf://${1}@${2}" --cache-root "${3}/hub")" || return $?
  jq -er '.snapshot_path' <<< "$result"
}
verify_package_cache() {
  local snapshot="$1"
  local args=(--package "$snapshot")
  if [[ -n "${LAYER_COUNT:-}" ]]; then args+=(--expected-layer-count "$LAYER_COUNT"); fi
  if [[ -n "${ACTIVATION_WIDTH:-}" ]]; then args+=(--expected-activation-width "$ACTIVATION_WIDTH"); fi
  just --justfile "${ROOT}/Justfile" skippy-layer-package-inspect "${args[@]}"
}

ensure_hf_package() {
  local package_ref="${MODEL_PACKAGE_REF:-hf://meshllm/gemma-4-26B-A4B-it-UD-Q4_K_M-layers}"
  local parsed_output
  parsed_output="$(parse_hf_package_ref "$package_ref")" || return $?
  local parsed=() parsed_line
  while IFS= read -r parsed_line; do
    parsed+=("$parsed_line")
  done <<< "$parsed_output"
  [[ "${#parsed[@]}" == 2 ]] || return 65
  local repo="${parsed[0]}"
  local revision="${parsed[1]}"
  local hf_home
  hf_home="$(hf_home_dir)"

  local hub_root="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${hf_home}/hub}}"
  local result args=(--reference "$package_ref" --cache-root "$hub_root")
  if [[ -n "${LAYER_COUNT:-}" ]]; then args+=(--expected-layer-count "$LAYER_COUNT"); fi
  if [[ -n "${ACTIVATION_WIDTH:-}" ]]; then args+=(--expected-activation-width "$ACTIVATION_WIDTH"); fi
  log "ensuring package with native commit-bound acquisition: ${repo}@${revision}"
  # Reuse the exact requested cached ref only when its complete package verifies.
  local cached cached_path
  if cached="$(just --justfile "${ROOT}/Justfile" skippy-layer-package-cache --reference "$package_ref" --cache-root "$hub_root" 2>/dev/null)" && \
     cached_path="$(jq -er '.snapshot_path' <<< "$cached")" && verify_package_cache "$cached_path" >&2; then
    result="$cached"
  else
    result="$(just --justfile "${ROOT}/Justfile" skippy-layer-package-fetch "${args[@]}")" || return $?
  fi
  local commit snapshot
  commit="$(jq -er '.commit' <<< "$result")" || return $?
  snapshot="$(jq -er '.snapshot_path' <<< "$result")" || return $?
  verify_package_cache "$snapshot" >&2 || return $?
  # Container lookup must use the exact commit just verified, even if a branch moves.
  export MODEL_PACKAGE_REF="hf://${repo}@${commit}"
  export HF_CACHE_MOUNT="$hub_root"

}

if [[ ! -f "$ENV_FILE" ]]; then
  log "creating ${ENV_FILE} from .env.example"
  cp "${LAB_DIR}/.env.example" "$ENV_FILE"
fi

load_env_file "$ENV_FILE"
load_env_file "$LINK_ENV_FILE"

ensure_hf_package

compose_args=(
  --env-file "$ENV_FILE"
)
if [[ -f "$LINK_ENV_FILE" ]]; then
  compose_args+=(--env-file "$LINK_ENV_FILE")
fi
compose_args+=(
  -f "${LAB_DIR}/docker-compose.yml"
)

log "starting Docker lab"
exec docker compose "${compose_args[@]}" up --build "$@"
