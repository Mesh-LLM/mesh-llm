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
  python3 - "$1" <<'PY'
import sys
print(float(sys.argv[1]) / 2.0)
PY
}

infer_layer_count() {
  local model_path="$1"
  skippy-package-builder inspect "$model_path" \
    | jq -r '[.tensors[] | select(.role == "layer") | .layer_index] | max + 1'
}

infer_activation_width() {
  local model_path="$1"
  python3 - "$model_path" <<'PY'
import struct
import sys

path = sys.argv[1]

TYPE_UINT8 = 0
TYPE_INT8 = 1
TYPE_UINT16 = 2
TYPE_INT16 = 3
TYPE_UINT32 = 4
TYPE_INT32 = 5
TYPE_FLOAT32 = 6
TYPE_BOOL = 7
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT64 = 10
TYPE_INT64 = 11
TYPE_FLOAT64 = 12

def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError("unexpected EOF")
    return data

def read_u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]

def read_u64(f):
    return struct.unpack("<Q", read_exact(f, 8))[0]

def read_i32(f):
    return struct.unpack("<i", read_exact(f, 4))[0]

def read_string(f):
    n = read_u64(f)
    return read_exact(f, n).decode("utf-8", "replace")

def skip_value(f, kind):
    if kind in (TYPE_UINT8, TYPE_INT8, TYPE_BOOL):
        f.seek(1, 1)
    elif kind in (TYPE_UINT16, TYPE_INT16):
        f.seek(2, 1)
    elif kind in (TYPE_UINT32, TYPE_INT32, TYPE_FLOAT32):
        f.seek(4, 1)
    elif kind in (TYPE_UINT64, TYPE_INT64, TYPE_FLOAT64):
        f.seek(8, 1)
    elif kind == TYPE_STRING:
        f.seek(read_u64(f), 1)
    elif kind == TYPE_ARRAY:
        inner = read_u32(f)
        count = read_u64(f)
        for _ in range(count):
            skip_value(f, inner)
    else:
        raise ValueError(f"unknown GGUF metadata type {kind}")

def read_scalar(f, kind):
    if kind == TYPE_UINT32:
        return read_u32(f)
    if kind == TYPE_INT32:
        return read_i32(f)
    if kind == TYPE_UINT16:
        return struct.unpack("<H", read_exact(f, 2))[0]
    if kind == TYPE_INT16:
        return struct.unpack("<h", read_exact(f, 2))[0]
    if kind == TYPE_UINT64:
        return read_u64(f)
    if kind == TYPE_INT64:
        return struct.unpack("<q", read_exact(f, 8))[0]
    if kind == TYPE_STRING:
        return read_string(f)
    skip_value(f, kind)
    return None

with open(path, "rb") as f:
    if read_exact(f, 4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    version = read_u32(f)
    if version < 2:
        raise SystemExit(f"unsupported GGUF version {version}")
    _tensor_count = read_u64(f)
    kv_count = read_u64(f)
    metadata = {}
    for _ in range(kv_count):
        key = read_string(f)
        kind = read_u32(f)
        metadata[key] = read_scalar(f, kind)

arch = metadata.get("general.architecture")
if not arch:
    raise SystemExit("GGUF metadata is missing general.architecture")
width = metadata.get(f"{arch}.embedding_length")
if not width:
    raise SystemExit(f"GGUF metadata is missing {arch}.embedding_length")
print(int(width))
PY
}

parse_hf_package_ref() {
  local ref="$1"
  python3 - "$ref" <<'PY'
import sys

value = sys.argv[1]
if not value.startswith("hf://"):
    raise SystemExit("package ref must start with hf://")
rest = value[len("hf://"):]
revision = "main"
if "@" in rest:
    rest, revision = rest.rsplit("@", 1)
elif ":" in rest:
    rest, revision = rest.rsplit(":", 1)
if "/" not in rest or not rest or not revision:
    raise SystemExit(f"invalid HF package ref: {value}")
print(rest)
print(revision)
PY
}

download_hf_file() {
  local repo="$1"
  local revision="$2"
  local remote_path="$3"
  local output_path="$4"
  local expected_bytes="${5:-}"

  mkdir -p "$(dirname "$output_path")"
  if [[ -f "$output_path" && -n "$expected_bytes" ]]; then
    local actual_bytes
    actual_bytes="$(stat -c '%s' "$output_path")"
    if [[ "$actual_bytes" == "$expected_bytes" ]]; then
      return 0
    fi
  elif [[ -f "$output_path" && -z "$expected_bytes" ]]; then
    return 0
  fi

  local tmp_path="${output_path}.tmp.$$"
  local url="https://huggingface.co/${repo}/resolve/${revision}/${remote_path}"
  local curl_args=(-fL --retry 5 --retry-delay 2 --connect-timeout 20)
  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl_args+=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi
  log "downloading ${repo}@${revision}/${remote_path}"
  curl "${curl_args[@]}" "$url" -o "$tmp_path"
  if [[ -n "$expected_bytes" ]]; then
    local actual_bytes
    actual_bytes="$(stat -c '%s' "$tmp_path")"
    if [[ "$actual_bytes" != "$expected_bytes" ]]; then
      rm -f "$tmp_path"
      log "downloaded size mismatch for ${remote_path}: expected ${expected_bytes}, got ${actual_bytes}"
      exit 66
    fi
  fi
  mv "$tmp_path" "$output_path"
}

hf_cache_snapshot_dir() {
  local repo="$1"
  local revision="$2"
  local cache_root="${HF_CACHE_ROOT:-/hf-cache}"
  local repo_dir="${cache_root}/hub/models--${repo//\//--}"
  local commit=""

  if [[ -f "${repo_dir}/refs/${revision}" ]]; then
    commit="$(cat "${repo_dir}/refs/${revision}")"
  elif [[ -d "${repo_dir}/snapshots/${revision}" ]]; then
    commit="$revision"
  fi

  if [[ -n "$commit" && -d "${repo_dir}/snapshots/${commit}" ]]; then
    printf '%s\n' "${repo_dir}/snapshots/${commit}"
    return 0
  fi

  if [[ -d "${repo_dir}/snapshots" ]]; then
    find "${repo_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
  fi
}

check_cached_package_files() {
  local package_dir="$1"
  local manifest_path="$2"
  local stage_index="$3"
  local layer_start="$4"
  local layer_end="$5"
  local stage_count="$6"
  local missing=()

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
  done < <(package_required_artifacts "$manifest_path" "$stage_index" "$layer_start" "$layer_end" "$stage_count")

  if ((${#missing[@]} > 0)); then
    printf '%s\n' "${missing[@]}"
    return 1
  fi
}

package_required_artifacts() {
  local manifest_path="$1"
  local stage_index="$2"
  local layer_start="$3"
  local layer_end="$4"
  local stage_count="$5"

  python3 - "$manifest_path" "$stage_index" "$layer_start" "$layer_end" "$stage_count" <<'PY'
import json
import sys

manifest_path, stage_index, layer_start, layer_end, stage_count = sys.argv[1:]
stage_index = int(stage_index)
layer_start = int(layer_start)
layer_end = int(layer_end)
stage_count = int(stage_count)

with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

def emit(artifact):
    print(f"{artifact['path']}\t{artifact['artifact_bytes']}")

emit(manifest["shared"]["metadata"])
if stage_index == 0:
    emit(manifest["shared"]["embeddings"])
if stage_index + 1 == stage_count:
    emit(manifest["shared"]["output"])

layers = {layer["layer_index"]: layer for layer in manifest["layers"]}
for layer_index in range(layer_start, layer_end):
    emit(layers[layer_index])
for projector in manifest.get("projectors", []):
    emit(projector)
PY
}

prepare_hf_layer_package_from_host_cache() {
  local repo="$1"
  local revision="$2"
  local package_ref="$3"
  local stage_index="$4"
  local stage_count="$5"

  local package_dir
  package_dir="$(hf_cache_snapshot_dir "$repo" "$revision" || true)"
  if [[ -z "$package_dir" || ! -f "${package_dir}/model-package.json" ]]; then
    log "HF package ${package_ref} is not present in the mounted host cache at ${HF_CACHE_ROOT:-/hf-cache}"
    log "download it on the host first:"
    log "  hf download ${repo} --revision ${revision} --include model-package.json --include 'shared/*' --include 'layers/*' --include 'projectors/*'"
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

  read -r layer_start layer_end < <(even_stage_range "$stage_index" "$stage_count" "$layer_count")

  local missing
  if ! missing="$(check_cached_package_files "$package_dir" "$manifest_path" "$stage_index" "$layer_start" "$layer_end" "$stage_count")"; then
    log "mounted host cache is missing files needed by stage ${stage_index}:"
    printf '%s\n' "$missing" >&2
    log "download the complete package on the host:"
    log "  hf download ${repo} --revision ${revision} --include model-package.json --include 'shared/*' --include 'layers/*' --include 'projectors/*'"
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

  mapfile -t parsed < <(parse_hf_package_ref "$package_ref")
  local repo="${parsed[0]}"
  local revision="${parsed[1]}"

  if [[ "${HF_PACKAGE_SOURCE:-host-cache}" == "host-cache" ]]; then
    prepare_hf_layer_package_from_host_cache "$repo" "$revision" "$package_ref" "$stage_index" "$stage_count"
    return 0
  fi

  local safe_repo
  safe_repo="$(printf '%s__%s' "$repo" "$revision" | tr '/:@' '___')"
  local package_dir="${PACKAGE_CACHE_DIR:-/package-cache}/${safe_repo}"
  local manifest_path="${package_dir}/model-package.json"

  download_hf_file "$repo" "$revision" "model-package.json" "$manifest_path"

  local layer_count activation_width
  layer_count="$(jq -r '.layer_count' "$manifest_path")"
  activation_width="$(jq -r '.activation_width // empty' "$manifest_path")"
  if [[ -z "$activation_width" ]]; then
    log "package manifest has no activation_width; set ACTIVATION_WIDTH"
    exit 65
  fi

  read -r layer_start layer_end < <(even_stage_range "$stage_index" "$stage_count" "$layer_count")

  package_required_artifacts "$manifest_path" "$stage_index" "$layer_start" "$layer_end" "$stage_count" \
    | while IFS=$'\t' read -r remote_path expected_bytes; do
    download_hf_file "$repo" "$revision" "$remote_path" "${package_dir}/${remote_path}" "$expected_bytes"
  done

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
  local stage_index="$1"
  local stage_count="$2"
  local layer_count="$3"
  python3 - "$stage_index" "$stage_count" "$layer_count" <<'PY'
import sys
stage_index, stage_count, layer_count = map(int, sys.argv[1:])
base = layer_count // stage_count
rem = layer_count % stage_count
start = 0
for index in range(stage_count):
    width = base + (1 if index < rem else 0)
    end = start + width
    if index == stage_index:
        print(start, end)
        break
    start = end
PY
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

  python3 - "${plan_root}/plan/stage-${stage_index}.json" "$config_path" <<'PYCONFIG'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
stage_index = int(os.environ["STAGE_INDEX"])
stage_count = int(os.environ.get("STAGE_COUNT", "4"))
bind_port = int(os.environ.get("STAGE_BIND_PORT", "19000"))

def peer(index):
    return {
        "stage_id": f"stage-{index}",
        "stage_index": index,
        "endpoint": f"tcp://stage{index}:{bind_port}",
    }

# Only deployment identity, transport and ordinary runtime controls change.
# Keep the admitted source, tensor, execution-contract and frontier fields.
config.update({
    "run_id": os.environ.get("RUN_ID", "skippy-docker-wan"),
    "topology_id": os.environ.get("TOPOLOGY_ID", "docker-wan-four-stage"),
    "bind_addr": f"0.0.0.0:{bind_port}",
    "upstream": None if stage_index == 0 else peer(stage_index - 1),
    "downstream": None if stage_index + 1 >= stage_count else peer(stage_index + 1),
    "n_batch": int(os.environ["N_BATCH"]) if os.environ.get("N_BATCH") else None,
    "n_ubatch": int(os.environ["N_UBATCH"]) if os.environ.get("N_UBATCH") else None,
    "cache_type_k": os.environ.get("CACHE_TYPE_K", "f16"),
    "cache_type_v": os.environ.get("CACHE_TYPE_V", "f16"),
    "flash_attn_type": os.environ.get("FLASH_ATTN_TYPE", "disabled"),
})
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PYCONFIG

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
