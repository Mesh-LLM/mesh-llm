#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[skippy-wan-lab] %s\n' "$*" >&2
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf 'required environment variable %s is not set\n' "$name" >&2
    exit 64
  fi
}

has_cli_arg() {
  local needle="$1"
  shift || true
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$needle" || "$arg" == "${needle}="* ]]; then
      return 0
    fi
  done
  return 1
}

healthcheck() {
  local port="${1:-19000}"
  nc -z 127.0.0.1 "$port"
}

float_half() {
  /usr/local/bin/mesh-llm-automation automation wan-observation delay "$1"
}

parse_hf_package_ref() {
  /usr/local/bin/skippy-package-builder parse-package-reference "$1"
}

hf_cache_snapshot_dir() {
  local result
  result="$(/usr/local/bin/skippy-package-builder resolve-layer-package-cache \
    --reference "hf://${1}@${2}" --cache-root "${HF_HUB_CACHE:-${HF_CACHE_ROOT:-/hf-cache}/hub}")" || return $?
  jq -er '.snapshot_path' <<< "$result"
}
check_cached_package_files() {
  local package_dir="$1"
  local manifest_path="$2"
  local stage_index="$3"
  local layer_start="$4"
  local layer_end="$5"
  local stage_count="$6"
  local missing=()
  local artifact_rows
  artifact_rows="$(package_required_artifacts "$manifest_path" "$stage_index" "$layer_start" "$layer_end" "$stage_count")" || return $?

  while IFS=$'\t' read -r remote_path expected_bytes; do
    local path="${package_dir}/${remote_path}"
    if [[ ! -f "$path" ]]; then
      missing+=("$remote_path")
      continue
    fi
    if [[ -n "$expected_bytes" ]]; then
      local actual_bytes
      actual_bytes="$(stat -c '%s' "$path")"
      if [[ "$actual_bytes" != "$expected_bytes" ]]; then
        missing+=("${remote_path} (expected ${expected_bytes} bytes, got ${actual_bytes})")
      fi
    fi
  done <<< "$artifact_rows"

  if ((${#missing[@]} > 0)); then
    printf '%s\n' "${missing[@]}"
    return 1
  fi
}

package_required_artifacts() {
  /usr/local/bin/skippy-package-builder plan-layer-package-artifacts \
    --manifest "$1" --stage-index "$2" --layer-start "$3" --layer-end "$4" --stage-count "$5"
}
prepare_hf_layer_package_from_host_cache() {
  local repo="$1"
  local revision="$2"
  local package_ref="$3"
  local stage_index="$4"
  local stage_count="$5"

  local package_dir
  if ! package_dir="$(hf_cache_snapshot_dir "$repo" "$revision")"; then
    log "requested HF package revision is unavailable in mounted host cache"
    return 67
  fi
  if [[ -z "$package_dir" || ! -f "${package_dir}/model-package.json" ]]; then
    log "HF package ${package_ref} is not present in the mounted host cache at ${HF_CACHE_ROOT:-/hf-cache}"
    log "download it on the host first:"
    log "  just skippy-layer-package-fetch --reference ${package_ref} --cache-root <hub-cache-root>"
    exit 67
  fi

  local manifest_path="${package_dir}/model-package.json"
  local layer_count activation_width
  layer_count="$(jq -r '.layer_count' "$manifest_path")"
  activation_width="$(jq -r '.activation_width // empty' "$manifest_path")"
  if [[ -z "$activation_width" ]]; then
    log "package manifest has no activation_width; set ACTIVATION_WIDTH"
    exit 65
  fi

  local stage_range
  stage_range="$(even_stage_range "$stage_index" "$stage_count" "$layer_count")" || return $?
  read -r layer_start layer_end <<< "$stage_range"

  local missing
  if ! missing="$(check_cached_package_files "$package_dir" "$manifest_path" "$stage_index" "$layer_start" "$layer_end" "$stage_count")"; then
    log "mounted host cache is missing files needed by stage ${stage_index}:"
    printf '%s\n' "$missing" >&2
    log "download the complete package on the host:"
    log "  just skippy-layer-package-fetch --reference ${package_ref} --cache-root <hub-cache-root>"
    exit 67
  fi

  export MODEL_PATH="$package_dir"
  export LOAD_MODE="layer-package"
  export LAYER_COUNT="${LAYER_COUNT:-$layer_count}"
  export ACTIVATION_WIDTH="${ACTIVATION_WIDTH:-$activation_width}"
  export LAYER_START="$layer_start"
  export LAYER_END="$layer_end"
  if [[ -z "${MODEL_ID:-}" || "${MODEL_ID:-}" == "skippy-wan-lab/model" ]]; then
    MODEL_ID="$(jq -r '.model_id' "$manifest_path")"
    export MODEL_ID
  fi
  log "using host HF cache package ${package_ref} at ${package_dir}"
}

prepare_hf_layer_package() {
  local package_ref="$1"
  local stage_index="$2"
  local stage_count="$3"

  local parsed_output
  parsed_output="$(parse_hf_package_ref "$package_ref")" || return $?
  local parsed=()
  mapfile -t parsed <<< "$parsed_output"
  [[ "${#parsed[@]}" == 2 ]] || return 65
  local repo="${parsed[0]}"
  local revision="${parsed[1]}"

  if [[ "${HF_PACKAGE_SOURCE:-host-cache}" == "host-cache" ]]; then
    prepare_hf_layer_package_from_host_cache "$repo" "$revision" "$package_ref" "$stage_index" "$stage_count" || return $?
    return 0
  fi

  [[ "${HF_PACKAGE_SOURCE:-host-cache}" == "download" ]] || return 65
  local fetched
  local fetch_args=(--reference "$package_ref" --cache-root "${PACKAGE_CACHE_DIR:-/package-cache}/hub"
    --stage-index "$stage_index" --stage-count "$stage_count")
  if [[ -n "${LAYER_COUNT:-}" ]]; then fetch_args+=(--expected-layer-count "$LAYER_COUNT"); fi
  if [[ -n "${ACTIVATION_WIDTH:-}" ]]; then fetch_args+=(--expected-activation-width "$ACTIVATION_WIDTH"); fi
  fetched="$(/usr/local/bin/skippy-package-builder fetch-layer-package "${fetch_args[@]}")" || return $?
  local package_dir manifest_path layer_count activation_width layer_start layer_end
  package_dir="$(jq -er '.snapshot_path' <<< "$fetched")" || return $?
  manifest_path="${package_dir}/model-package.json"
  layer_count="$(jq -er '.layer_count' <<< "$fetched")" || return $?
  activation_width="$(jq -er '.activation_width' <<< "$fetched")" || return $?
  layer_start="$(jq -er '.layer_start' <<< "$fetched")" || return $?
  layer_end="$(jq -er '.layer_end' <<< "$fetched")" || return $?

  export MODEL_PATH="$package_dir"
  export LOAD_MODE="layer-package"
  export LAYER_COUNT="${LAYER_COUNT:-$layer_count}"
  export ACTIVATION_WIDTH="${ACTIVATION_WIDTH:-$activation_width}"
  export LAYER_START="$layer_start"
  export LAYER_END="$layer_end"
  if [[ -z "${MODEL_ID:-}" || "${MODEL_ID:-}" == "skippy-wan-lab/model" ]]; then
    MODEL_ID="$(jq -r '.model_id' "$manifest_path")"
    export MODEL_ID
  fi
  log "prepared HF layer package ${package_ref} at ${package_dir}"
}

even_stage_range() {
  /usr/local/bin/skippy-package-builder even-layer-stage-range "$1" "$2" "$3"
}
route_iface_for_host() {
  local host="$1"
  local ip
  ip="$(getent hosts "$host" | awk '{print $1; exit}')"
  if [[ -z "$ip" ]]; then
    return 1
  fi
  ip route get "$ip" | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "dev") {
          print $(i + 1)
          exit
        }
      }
    }
  '
}

apply_linux_wan() {
  if [[ "${WAN_ENABLE:-1}" == "0" || "${WAN_ENABLE:-1}" == "false" ]]; then
    log "WAN shaping disabled"
    return 0
  fi

  local iface="${WAN_IFACE:-}"
  if [[ -z "$iface" && -n "${WAN_PROBE_HOST:-}" ]]; then
    iface="$(route_iface_for_host "$WAN_PROBE_HOST" || true)"
  fi
  iface="${iface:-eth0}"

  local delay="${WAN_DELAY_MS:-}"
  if [[ -z "$delay" && -n "${WAN_RTT_MS:-}" ]]; then
    delay="$(float_half "$WAN_RTT_MS")"
  fi
  delay="${delay:-0}"

  tc qdisc del dev "$iface" root >/dev/null 2>&1 || true

  local args=(qdisc replace dev "$iface" root netem delay "${delay}ms")
  if [[ -n "${WAN_JITTER_MS:-}" && "${WAN_JITTER_MS}" != "0" ]]; then
    args+=("${WAN_JITTER_MS}ms" distribution normal)
  fi
  if [[ -n "${WAN_LOSS_PERCENT:-}" && "${WAN_LOSS_PERCENT}" != "0" ]]; then
    args+=(loss "${WAN_LOSS_PERCENT}%")
  fi
  if [[ -n "${WAN_RATE_MBIT:-}" && "${WAN_RATE_MBIT}" != "0" ]]; then
    args+=(rate "${WAN_RATE_MBIT}mbit")
  fi

  log "applying Linux tc shaping on ${iface}: ${args[*]}"
  tc "${args[@]}"
  tc -s qdisc show dev "$iface" >&2 || true
}

write_stage_config() {
  if [[ -n "${CONFIG_PATH:-}" && -f "$CONFIG_PATH" ]]; then
    printf '%s\n' "$CONFIG_PATH"
    return 0
  fi

  require_env MODEL_PATH
  if [[ ! -f "$MODEL_PATH" ]]; then
    log "automatic admission requires a direct GGUF; supply CONFIG_PATH with an admitted config for a layer package"
    exit 65
  fi
  local stage_index="${STAGE_INDEX:?STAGE_INDEX is required}"
  local stage_count="${STAGE_COUNT:-4}"
  if [[ ! "$stage_index" =~ ^[0-9]+$ || ! "$stage_count" =~ ^[0-9]+$ ]] ||
    (( stage_count < 2 || stage_count > 10000 || stage_index >= stage_count )); then
    log "STAGE_COUNT must be at least two and STAGE_INDEX must identify one of its stages"
    exit 64
  fi

  local config_dir="${CONFIG_DIR:-/run/skippy-wan-lab}"
  local config_path="${CONFIG_PATH:-${config_dir}/stage-${stage_index}.json}"
  mkdir -p "$config_dir"
  local plan_root
  plan_root="$(mktemp -d "${config_dir}/admission.XXXXXX")"
  local args=(plan-split --model-path "$MODEL_PATH"
    --model-id "${MODEL_ID:-skippy-wan-lab/model}"
    --ctx-size "${CTX_SIZE:-512}" --lanes "${STAGE_LANES:-1}"
    --n-gpu-layers 0 --output-dir "${plan_root}/plan")
  local index
  for (( index = 0; index < stage_count; index++ )); do
    args+=(--worker "127.0.0.1:$((20000 + index))")
  done
  # Admission owns tensor closures, contracts and graph frontiers. Placeholder
  # addresses avoid requiring every container to be up while the plan is built.
  skippy "${args[@]}" >&2

  local deployment=(automation wan-stage-deployment
    --input "${plan_root}/plan/stage-${stage_index}.json" --output "$config_path"
    --stage-index "$stage_index" --stage-count "$stage_count"
    --bind-port "${STAGE_BIND_PORT:-19000}"
    --run-id "${RUN_ID:-skippy-docker-wan}"
    --topology-id "${TOPOLOGY_ID:-docker-wan-four-stage}"
    --cache-type-k "${CACHE_TYPE_K:-f16}" --cache-type-v "${CACHE_TYPE_V:-f16}"
    --flash-attn-type "${FLASH_ATTN_TYPE:-disabled}")
  if [[ -n "${N_BATCH:-}" ]]; then deployment+=(--n-batch "$N_BATCH"); fi
  if [[ -n "${N_UBATCH:-}" ]]; then deployment+=(--n-ubatch "$N_UBATCH"); fi
  /usr/local/bin/mesh-llm-automation "${deployment[@]}"


  printf '%s\n' "$config_path"
}

run_metrics() {
  exec metrics-server serve \
    --db "${METRICS_DB:-/data/metrics.duckdb}" \
    --http-addr "${METRICS_HTTP_ADDR:-0.0.0.0:18080}" \
    --otlp-grpc-addr "${METRICS_OTLP_GRPC_ADDR:-0.0.0.0:14317}"
}

run_stage() {
  require_env STAGE_INDEX

  local stage_index="$STAGE_INDEX"
  local stage_count="${STAGE_COUNT:-4}"
  if [[ -n "${CONFIG_PATH:-}" && -f "$CONFIG_PATH" ]]; then
    export MODEL_ID="${MODEL_ID:-$(jq -er '.model_id' "$CONFIG_PATH")}"
    export STAGE_LANES="${STAGE_LANES:-$(jq -er '.lane_count' "$CONFIG_PATH")}"
  elif [[ -n "${MODEL_PACKAGE_REF:-}" ]]; then
    prepare_hf_layer_package "$MODEL_PACKAGE_REF" "$stage_index" "$stage_count"
  else
    require_env MODEL_PATH
  fi

  if (( stage_index + 1 < stage_count )); then
    export WAN_PROBE_HOST="${WAN_PROBE_HOST:-stage$((stage_index + 1))}"
  else
    export WAN_PROBE_HOST="${WAN_PROBE_HOST:-stage$((stage_index - 1))}"
  fi
  apply_linux_wan

  local config_path
  config_path="$(write_stage_config)"
  local layer_start layer_end
  layer_start="$(jq -er '.layer_start' "$config_path")"
  layer_end="$(jq -er '.layer_end' "$config_path")"
  log "stage ${stage_index}/${stage_count}: layers ${layer_start}..${layer_end}, config=${config_path}"

  local activation_codec
  if [[ -n "${CONFIG_PATH:-}" && -f "$CONFIG_PATH" && -z "${ACTIVATION_WIRE_DTYPE:-}" ]]; then
    activation_codec="$(jq -r '.activation_codec // "raw-f32-v1"' "$config_path")"
  else
    case "${ACTIVATION_WIRE_DTYPE:-f16}" in
      f32) activation_codec=raw-f32-v1 ;;
      f16) activation_codec=f16-rne-v1 ;;
      bf16) activation_codec=bf16-rne-v1 ;;
      *)
        log "unsupported ACTIVATION_WIRE_DTYPE: ${ACTIVATION_WIRE_DTYPE}"
        exit 64
        ;;
    esac
  fi

  local args=(
    serve --stage-transport binary
    --config "$config_path"
    --activation-codec "$activation_codec"
    --metrics-otlp-grpc "${METRICS_OTLP_GRPC:-http://metrics:14317}"
    --telemetry-queue-capacity "${TELEMETRY_QUEUE_CAPACITY:-4096}"
    --telemetry-level "${TELEMETRY_LEVEL:-debug}"
    --max-inflight "${STAGE_MAX_INFLIGHT:-${STAGE_LANES:-1}}"
  )

  if [[ -n "${REPLY_CREDIT_LIMIT:-}" ]]; then
    args+=(--reply-credit-limit "$REPLY_CREDIT_LIMIT")
  fi
  if [[ "${ASYNC_PREFILL_FORWARD:-0}" == "1" || "${ASYNC_PREFILL_FORWARD:-0}" == "true" ]]; then
    args+=(--async-prefill-forward)
  fi
  if [[ "$stage_index" == "0" ]]; then
    args+=(
      --bind-addr "${OPENAI_BIND_ADDR:-0.0.0.0:9337}"
      --model-id "${MODEL_ID:-skippy-wan-lab/model}"
      --default-max-tokens "${OPENAI_DEFAULT_MAX_TOKENS:-32}"
      --generation-concurrency "${OPENAI_GENERATION_CONCURRENCY:-${STAGE_LANES:-1}}"
      --prefill-chunk-size "${OPENAI_PREFILL_CHUNK_SIZE:-256}"
      --prefill-chunk-policy "${OPENAI_PREFILL_CHUNK_POLICY:-adaptive-ramp}"
      --prefill-adaptive-start "${OPENAI_PREFILL_ADAPTIVE_START:-128}"
      --prefill-adaptive-step "${OPENAI_PREFILL_ADAPTIVE_STEP:-128}"
      --prefill-adaptive-max "${OPENAI_PREFILL_ADAPTIVE_MAX:-512}"
    )
  else
    args+=(--worker-only)
  fi

  exec skippy "${args[@]}"
}

run_prompt() {
  require_env STAGE_INDEX
  if [[ "$STAGE_INDEX" != "0" ]]; then
    log "interactive prompt should be attached from stage0; this is stage ${STAGE_INDEX}"
    exit 64
  fi

  local args=()
  if ! has_cli_arg "--endpoint" "$@"; then
    args+=(--endpoint "${PROMPT_OPENAI_ENDPOINT:-http://127.0.0.1:9337/v1}")
  fi
  if ! has_cli_arg "--max-new-tokens" "$@"; then
    args+=(--max-new-tokens "${PROMPT_MAX_NEW_TOKENS:-${OPENAI_DEFAULT_MAX_TOKENS:-32}}")
  fi
  if ! has_cli_arg "--history-path" "$@"; then
    args+=(--history-path "${PROMPT_HISTORY_PATH:-/tmp/skippy-wan-lab-prompt-history.txt}")
  fi

  log "attaching interactive prompt"
  exec skippy prompt "${args[@]}" "$@"
}

case "${1:-${APP_ROLE:-stage}}" in
  metrics)
    run_metrics
    ;;
  stage)
    run_stage
    ;;
  prompt)
    shift || true
    run_prompt "$@"
    ;;
  healthcheck)
    healthcheck "${2:-${STAGE_BIND_PORT:-19000}}"
    ;;
  metrics-healthcheck)
    healthcheck "${2:-18080}"
    ;;
  shell)
    shift || true
    exec bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
