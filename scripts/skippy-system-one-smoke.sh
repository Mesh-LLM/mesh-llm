#!/usr/bin/env bash
set -euo pipefail

# System One (OpenJEV) smoke for the llama.cpp upstream canary.
#
# Exercises `POST /systemone`, the Jev-compatible read introduced by the
# DiffusionGemma / System One proof of concept
# (docs/design/OPENJEV_SKIPPY_POC.md). It runs in two independent parts because
# they have different preconditions:
#
# * contract  - the request/response contract and every fail-closed boundary
#               decided by the Rust frontend. Driven through a pinned
#               non-DiffusionGemma fixture, so it needs no diffusion model and
#               is backend independent: it runs on any host with a patched
#               native build, including the Metal family-certify runner.
# * full-read - one real DiffusionGemma read, repeated and interleaved. Needs
#               the pinned DiffusionGemma GGUF in the offline model cache AND a
#               declared *qualified* execution backend.
#
# The proof of concept certifies only CUDA (the runbook records that a live
# Metal result "has not been certified"), so the read part is admitted by
# explicit declaration rather than by assuming whatever accelerator happens to
# be present. When it is not admitted the smoke records `unqualified`, prints
# why, and repeats it in the report instead of passing quietly. Set
# SYSTEMONE_SMOKE_REQUIRE_QUALIFIED=1 to make an unqualified read part fatal.
#
# Report: $REPORT_DIR/system-one.json
# Exit: 0 pass | unqualified, 1 fail (including an unusable environment).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_llama_build_dir() {
  if [[ -n "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    printf '%s\n' "$LLAMA_STAGE_BUILD_DIR"
  elif [[ -n "${SKIPPY_LLAMA_BUILD_DIR:-}" ]]; then
    printf '%s\n' "$SKIPPY_LLAMA_BUILD_DIR"
  else
    "$ROOT/scripts/build-llama.sh" --print-build-dir
  fi
}

SMOKE_MANIFEST="${SYSTEMONE_SMOKE_MANIFEST:-$ROOT/ci/model-artifacts/manifests/skippy-system-one-smoke.json}"
# The cadence this lane runs at. The manifest contract refuses an artifact that
# is not authorized for it.
SMOKE_CADENCE="${SYSTEMONE_SMOKE_CADENCE:-manual}"
CONTRACT_ARTIFACT_ID="${SYSTEMONE_SMOKE_CONTRACT_ARTIFACT_ID:-family-qwen3-dense}"
READ_ARTIFACT_ID="${SYSTEMONE_SMOKE_READ_ARTIFACT_ID:-family-diffusion-gemma}"
BUILD_BACKEND="${SYSTEMONE_SMOKE_BUILD_BACKEND:-${LLAMA_STAGE_BACKEND:-}}"
CERTIFIED_BACKENDS="${SYSTEMONE_SMOKE_CERTIFIED_BACKENDS:-cuda}"
REQUIRE_QUALIFIED="${SYSTEMONE_SMOKE_REQUIRE_QUALIFIED:-0}"
SKIP_CONTRACT="${SYSTEMONE_SMOKE_SKIP_CONTRACT:-0}"
ALIAS="${SYSTEMONE_SMOKE_ALIAS:-openjev-latest}"
CASES_DRIVER="${SYSTEMONE_SMOKE_DRIVER:-$ROOT/scripts/skippy-system-one-cases.py}"
TEST_MODEL_RESOLVER="$ROOT/scripts/resolve-test-model-manifest.py"
STAGE_SERVER_BIN="${STAGE_SERVER_BIN:-$ROOT/target/debug/skippy-server}"
MODEL_PACKAGE_BIN="${MODEL_PACKAGE_BIN:-$ROOT/target/debug/skippy-model-package}"
CTX_SIZE="${SYSTEMONE_SMOKE_CTX_SIZE:-8192}"
# The micro-batch must hold the model's fixed diffusion canvas. The published
# Q4_K_M canvas is larger than OpenJEV's vLLM default of 64; 512 clears it.
READ_N_BATCH="${SYSTEMONE_SMOKE_READ_N_BATCH:-512}"
CONTRACT_N_BATCH="${SYSTEMONE_SMOKE_CONTRACT_N_BATCH:-128}"
SERVER_STARTUP_TIMEOUT_SECS="${SYSTEMONE_SMOKE_STARTUP_TIMEOUT_SECS:-300}"
READ_REQUEST_TIMEOUT_SECS="${SYSTEMONE_SMOKE_READ_TIMEOUT_SECS:-900}"
CONTRACT_REQUEST_TIMEOUT_SECS="${SYSTEMONE_SMOKE_CONTRACT_TIMEOUT_SECS:-120}"
WORK_DIR="${WORK_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/skippy-system-one-smoke}"
REPORT_DIR="${REPORT_DIR:-${WORK_DIR}/reports}"
REPORT_PATH="${SYSTEMONE_SMOKE_REPORT:-$REPORT_DIR/system-one.json}"

SERVER_PID=""
REPORT_REASONS=""
# Full-model read cache state, filled in by read_part when the read backend is
# declared qualified: the resolved artifact path and whether the offline cache
# was actually checked (false on the intentionally unqualified Metal path).
READ_RESOLVED_ARTIFACT_PATH=""
READ_ARTIFACT_CACHE_CHECKED="false"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    return 2
  fi
}

require_file() {
  if [[ ! -x "$1" ]]; then
    echo "required executable not found: $1" >&2
    echo "$2" >&2
    return 2
  fi
}

pick_port() {
  python3 - <<'PY'
import socket
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    local children
    children="$(pgrep -P "$SERVER_PID" 2>/dev/null || true)"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    if [[ -n "$children" ]]; then
      printf '%s\n' "$children" | xargs kill >/dev/null 2>&1 || true
    fi
    sleep 1
    kill -9 "$SERVER_PID" >/dev/null 2>&1 || true
    if [[ -n "$children" ]]; then
      printf '%s\n' "$children" | xargs kill -9 >/dev/null 2>&1 || true
    fi
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT

record_reason() {
  if [[ -z "$REPORT_REASONS" ]]; then
    REPORT_REASONS="$1"
  else
    REPORT_REASONS="$REPORT_REASONS; $1"
  fi
}

# Resolves one artifact through the shared test-model manifest contract, which
# enforces the authorized cadence, single-file membership, and the pinned
# size + SHA-256. Prints the resolver's JSON summary.
artifact_summary() {
  python3 "$TEST_MODEL_RESOLVER" "$SMOKE_MANIFEST" \
    --artifact-id "$1" \
    --cadence "$SMOKE_CADENCE" \
    --require-single-file
}

# Resolves an artifact from the read-only offline cache without touching the
# network. Prints the path, or nothing on a cache miss.
cached_artifact_path() {
  local repo="$1" revision="$2" file="$3" root snapshot
  for root in "${HF_CACHE:-}/hub" "${HF_HUB_CACHE:-}"; do
    if [[ -n "$root" && -d "$root" ]]; then
      snapshot="$root/models--${repo//\//--}/snapshots/$revision/$file"
      if [[ -f "$snapshot" ]]; then
        printf '%s\n' "$snapshot"
        return 0
      fi
    fi
  done
  return 0
}

# Verifies the pinned size and SHA-256 of the artifact files in `dir`. A
# mismatch is a hard failure, never a skip.
verify_artifact_digest() {
  python3 "$TEST_MODEL_RESOLVER" "$SMOKE_MANIFEST" \
    --artifact-id "$1" \
    --cadence "$SMOKE_CADENCE" \
    --require-single-file \
    --verify-root "$2" >&2
}

write_stage_config() {
  python3 - "$@" <<'PY'
import json
import sys
from pathlib import Path

(
    config_path,
    model_id,
    model_path,
    source_sha256,
    layer_end,
    bind_addr,
    lane_count,
    ctx_size,
    n_batch,
) = sys.argv[1:]

config = {
    "run_id": "skippy-system-one-smoke",
    "topology_id": "skippy-system-one-smoke-single-stage",
    "model_id": model_id,
    "model_path": model_path,
    "stage_id": "stage-0",
    "stage_index": 0,
    "layer_start": 0,
    "layer_end": int(layer_end),
    "ctx_size": int(ctx_size),
    "lane_count": int(lane_count),
    "n_batch": int(n_batch),
    "n_ubatch": int(n_batch),
    "n_gpu_layers": 0,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "load_mode": "runtime-slice",
    "execution_contract": "",
    "bind_addr": bind_addr,
    "upstream": None,
    "downstream": None,
}
if source_sha256:
    config["source_model_sha256"] = source_sha256
with Path(config_path).open("w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY
}

# Both parts need a patched native build. Checked lazily so an unqualified
# read part (or a skipped contract part) does not need one.
require_smoke_binaries() {
  require_file "$MODEL_PACKAGE_BIN" \
    "run scripts/prepare-llama.sh pinned && scripts/build-llama.sh, then cargo build -p skippy-model-package" || return 2
  require_file "$STAGE_SERVER_BIN" \
    "run scripts/prepare-llama.sh pinned && scripts/build-llama.sh, then cargo build -p skippy-server" || return 2
  LLAMA_BUILD_DIR="$(resolve_llama_build_dir)"
  if [[ ! -d "$LLAMA_BUILD_DIR" ]]; then
    echo "patched llama.cpp build dir not found: $LLAMA_BUILD_DIR" >&2
    echo "run scripts/prepare-llama.sh pinned && scripts/build-llama.sh first" >&2
    return 2
  fi
  return 0
}

stage_layer_end() {
  LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
    "$MODEL_PACKAGE_BIN" inspect "$1" \
    | jq -r '[.tensors[] | select(.role == "layer") | .layer_index] | max + 1'
}

start_stage_server() {
  local label="$1" config="$2" port="$3" log="$4" deadline
  echo "system-one: starting ${label} on 127.0.0.1:${port}"
  LLAMA_STAGE_BUILD_DIR="$LLAMA_BUILD_DIR" \
    "$STAGE_SERVER_BIN" serve-openai \
      --config "$config" \
      --bind-addr "127.0.0.1:${port}" \
      >"$log" 2>&1 &
  SERVER_PID="$!"
  deadline=$(( $(date +%s) + SERVER_STARTUP_TIMEOUT_SECS ))
  while (( $(date +%s) <= deadline )); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "${label} server exited during startup; log follows" >&2
      sed -n '1,240p' "$log" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "${label} server did not become ready within ${SERVER_STARTUP_TIMEOUT_SECS}s; log follows" >&2
  sed -n '1,240p' "$log" >&2 || true
  return 1
}

# Shared body for one `serve-openai` + case-matrix run. Arguments:
# label, artifact id, explicit model path override, mode, n_batch, request
# timeout.
run_cases_against_stage() {
  local label="$1" artifact_id="$2" model_path="$3" mode="$4" n_batch="$5" \
    request_timeout="$6"
  local summary repo revision file model_id layer_end port config log rc
  require_smoke_binaries || return $?
  rc=0
  summary="$(artifact_summary "$artifact_id")" || rc=$?
  if (( rc != 0 )); then
    echo "system-one: could not resolve ${artifact_id} from ${SMOKE_MANIFEST} at cadence ${SMOKE_CADENCE}" >&2
    return 2
  fi
  repo="$(jq -r '.repo' <<<"$summary")"
  revision="$(jq -r '.revision' <<<"$summary")"
  file="$(jq -r '.file' <<<"$summary")"
  model_id="$(jq -r '.model_ref' <<<"$summary")"

  if [[ -z "$model_path" ]]; then
    model_path="$(cached_artifact_path "$repo" "$revision" "$file")"
  fi
  if [[ -z "$model_path" ]]; then
    echo "system-one: ${label} fixture ${repo}/${file} is not in the local model cache" >&2
    echo "set SYSTEMONE_SMOKE_CONTRACT_MODEL_PATH / SYSTEMONE_SMOKE_READ_MODEL_PATH, or pre-warm the cache with: scripts/skippy-system-one-smoke.sh --prewarm" >&2
    return 3
  fi
  rc=0
  verify_artifact_digest "$artifact_id" "$(dirname "$model_path")" || rc=$?
  if (( rc != 0 )); then
    echo "system-one: ${label} fixture failed its pinned size/SHA-256 check; refusing to load it" >&2
    return 2
  fi

  rc=0
  layer_end="$(stage_layer_end "$model_path")" || rc=$?
  if (( rc != 0 )) || [[ -z "$layer_end" || "$layer_end" == "null" || "$layer_end" -lt 2 ]]; then
    echo "failed to infer the layer count of $model_path" >&2
    return 2
  fi

  port="$(pick_port)"
  config="$WORK_DIR/${mode}-stage.json"
  log="$WORK_DIR/${mode}-server.log"
  write_stage_config "$config" "$model_id" "$model_path" "$(jq -r '.sha256' <<<"$summary")" \
    "$layer_end" "127.0.0.1:${port}" 1 "$CTX_SIZE" "$n_batch"
  start_stage_server "$label" "$config" "$port" "$log" || return 1
  rc=0
  python3 "$CASES_DRIVER" \
    --base-url "http://127.0.0.1:${port}" \
    --model "$model_id" \
    --alias "$ALIAS" \
    --mode "$mode" \
    --timeout "$request_timeout" \
    --json-out "$REPORT_DIR/system-one-${mode}.json" || rc=$?
  if (( rc != 0 )); then
    echo "System One ${label} cases failed (exit ${rc}); server log follows" >&2
    sed -n '1,240p' "$log" >&2 || true
    cleanup
    return 1
  fi
  cleanup
  return 0
}

prewarm_plan() {
  local summary repo revision file url size sha
  summary="$(artifact_summary "$READ_ARTIFACT_ID")"
  repo="$(jq -r '.repo' <<<"$summary")"
  revision="$(jq -r '.revision' <<<"$summary")"
  file="$(jq -r '.file' <<<"$summary")"
  url="$(jq -r '.url' <<<"$summary")"
  size="$(jq -r '.size_bytes' <<<"$summary")"
  sha="$(jq -r '.sha256' <<<"$summary")"
  cat <<EOF
System One smoke: pre-warm plan for $READ_ARTIFACT_ID

The family-certify model cache is offline and operator-owned, so no CI step
downloads this artifact. Run this once, on a host with network access and write
access to the cache:

    hf download "$repo" "$file" --revision "$revision"

The smoke resolves the artifact from the offline cache and verifies its pinned
size and SHA-256 before loading it:

    repository: $repo
    revision:   $revision
    file:       $file
    size_bytes: $size
    sha256:     $sha
    url:        $url

Then, on a host with a qualified execution backend, run:

    LLAMA_STAGE_BACKEND=cuda SYSTEMONE_SMOKE_BUILD_BACKEND=cuda \\
      scripts/skippy-system-one-smoke.sh

CUDA is the only backend the proof of concept certifies. With the cache entry
present but no declared qualified backend, the read part still reports NOT
CERTIFIED and the run exits 0. Qualifying another backend is a deliberate
one-time decision: run the artifact there, record the evidence, then widen
SYSTEMONE_SMOKE_CERTIFIED_BACKENDS (or the
LLAMA_CANARY_SYSTEMONE_CERTIFIED_BACKENDS repository variable).
EOF
}

contract_part() {
  if [[ "$SKIP_CONTRACT" == "1" || "$SKIP_CONTRACT" == "true" ]]; then
    echo "system-one: contract part skipped (SYSTEMONE_SMOKE_SKIP_CONTRACT=${SKIP_CONTRACT})"
    return 0
  fi
  run_cases_against_stage "contract" "$CONTRACT_ARTIFACT_ID" \
    "${SYSTEMONE_SMOKE_CONTRACT_MODEL_PATH:-}" contract "$CONTRACT_N_BATCH" \
    "$CONTRACT_REQUEST_TIMEOUT_SECS"
}

read_part() {
  if [[ "$READ_QUALIFIED" != "true" ]]; then
    echo "system-one: full-model read NOT CERTIFIED: build backend '${BUILD_BACKEND:-<unset>}' is not one of the declared qualified backends (${CERTIFIED_BACKENDS})" >&2
    return 3
  fi
  local summary cached
  summary="$(artifact_summary "$READ_ARTIFACT_ID")" || return 2
  cached="$(cached_artifact_path "$(jq -r '.repo' <<<"$summary")" \
    "$(jq -r '.revision' <<<"$summary")" "$(jq -r '.file' <<<"$summary")")"
  READ_RESOLVED_ARTIFACT_PATH="${SYSTEMONE_SMOKE_READ_MODEL_PATH:-$cached}"
  READ_ARTIFACT_CACHE_CHECKED="true"
  if [[ -z "$READ_RESOLVED_ARTIFACT_PATH" ]]; then
    # A declared-qualified backend must never satisfy this lane without
    # running the full-model read: a missing pinned artifact is a hard
    # failure, distinct from the intentionally unqualified path (exit 3).
    echo "system-one: full-model read FAILED: build backend '${BUILD_BACKEND:-<unset>}' is declared qualified (${CERTIFIED_BACKENDS}), but pinned $(jq -r '.repo' <<<"$summary")/$(jq -r '.file' <<<"$summary") is not in the offline model cache" >&2
    echo "pre-warm it with: scripts/skippy-system-one-smoke.sh --prewarm" >&2
    return 4
  fi
  run_cases_against_stage "full-model read" "$READ_ARTIFACT_ID" \
    "${SYSTEMONE_SMOKE_READ_MODEL_PATH:-}" full-read "$READ_N_BATCH" \
    "$READ_REQUEST_TIMEOUT_SECS"
}

main() {
  if [[ "${1:-}" == "--prewarm" ]]; then
    require_cmd python3 || exit 2
    if [[ ! -f "$SMOKE_MANIFEST" ]]; then
      echo "System One smoke manifest not found: $SMOKE_MANIFEST" >&2
      exit 2
    fi
    prewarm_plan
    exit 0
  fi

  require_cmd jq || exit 2
  require_cmd python3 || exit 2
  require_cmd curl || exit 2
  if [[ ! -f "$SMOKE_MANIFEST" ]]; then
    echo "System One smoke manifest not found: $SMOKE_MANIFEST" >&2
    echo "regenerate it with scripts/generate-test-model-manifests.py" >&2
    exit 2
  fi
  if [[ ! -f "$CASES_DRIVER" ]]; then
    echo "System One case driver not found: $CASES_DRIVER" >&2
    exit 2
  fi
  mkdir -p "$REPORT_DIR"

  if [[ -n "$BUILD_BACKEND" && ",$CERTIFIED_BACKENDS," == *",$BUILD_BACKEND,"* ]]; then
    READ_QUALIFIED="true"
  else
    READ_QUALIFIED="false"
  fi

  LLAMA_BUILD_DIR="$(resolve_llama_build_dir)"
  local contract_status="skipped" read_status="skipped" rc=0
  if [[ "$SKIP_CONTRACT" == "1" || "$SKIP_CONTRACT" == "true" ]]; then
    contract_status="skipped"
  else
    rc=0
    contract_part || rc=$?
    case "$rc" in
      0) contract_status="pass" ;;
      2) contract_status="error" ;;
      *) contract_status="fail" ;;
    esac
  fi
  rc=0
  read_part || rc=$?
  case "$rc" in
    0) read_status="pass" ;;
    3) read_status="unqualified" ;;
    4)
      read_status="fail"
      record_reason "declared-qualified backend ${BUILD_BACKEND:-<unset>} is missing the pinned full-model read artifact"
      ;;
    2) read_status="error" ;;
    *) read_status="fail" ;;
  esac

  local status="pass" exit_status=0
  case "$contract_status" in
    pass|skipped) ;;
    *) status="fail"; exit_status=1 ;;
  esac
  case "$read_status" in
    pass) ;;
    unqualified)
      record_reason "full-model read not certified: build_backend=${BUILD_BACKEND:-<unset>} certified_backends=${CERTIFIED_BACKENDS}"
      if [[ "$status" != "fail" ]]; then
        status="unqualified"
      fi
      if [[ "$REQUIRE_QUALIFIED" == "1" || "$REQUIRE_QUALIFIED" == "true" ]]; then
        status="fail"
        exit_status=1
        record_reason "SYSTEMONE_SMOKE_REQUIRE_QUALIFIED is set, so an unqualified full-model read is fatal"
      fi
      ;;
    *) status="fail"; exit_status=1 ;;
  esac

  SYSTEMONE_SMOKE_STATUS="$status" \
  SYSTEMONE_SMOKE_CONTRACT_STATUS="$contract_status" \
  SYSTEMONE_SMOKE_READ_STATUS="$read_status" \
  SYSTEMONE_SMOKE_BUILD_BACKEND="$BUILD_BACKEND" \
  SYSTEMONE_SMOKE_CERTIFIED_BACKENDS="$CERTIFIED_BACKENDS" \
  SYSTEMONE_SMOKE_REASONS="$REPORT_REASONS" \
  SYSTEMONE_SMOKE_REPORT_PATH="$REPORT_PATH" \
  SYSTEMONE_SMOKE_READ_ARTIFACT="$READ_ARTIFACT_ID" \
  SYSTEMONE_SMOKE_RESOLVED_ARTIFACT_PATH="$READ_RESOLVED_ARTIFACT_PATH" \
  SYSTEMONE_SMOKE_ARTIFACT_CACHE_CHECKED="$READ_ARTIFACT_CACHE_CHECKED" \
  SYSTEMONE_SMOKE_REQUIRE_QUALIFIED_FLAG="$REQUIRE_QUALIFIED" \
  SYSTEMONE_SMOKE_SKIP_CONTRACT_FLAG="$SKIP_CONTRACT" \
  SYSTEMONE_SMOKE_CTX_SIZE_VALUE="$CTX_SIZE" \
    python3 - <<'PY'
import json
import os
from pathlib import Path

report = {
    "schema_version": 1,
    "status": os.environ["SYSTEMONE_SMOKE_STATUS"],
    "contract": {"status": os.environ["SYSTEMONE_SMOKE_CONTRACT_STATUS"]},
    "full_model_read": {
        "status": os.environ["SYSTEMONE_SMOKE_READ_STATUS"],
        "artifact": os.environ["SYSTEMONE_SMOKE_READ_ARTIFACT"],
        "backend": os.environ["SYSTEMONE_SMOKE_BUILD_BACKEND"] or None,
        "artifact_path": os.environ["SYSTEMONE_SMOKE_RESOLVED_ARTIFACT_PATH"] or None,
        "artifact_cache_checked": os.environ["SYSTEMONE_SMOKE_ARTIFACT_CACHE_CHECKED"]
        in ("1", "true"),
        "certified_backends": [
            item
            for item in os.environ["SYSTEMONE_SMOKE_CERTIFIED_BACKENDS"].split(",")
            if item
        ],
        "require_qualified": os.environ["SYSTEMONE_SMOKE_REQUIRE_QUALIFIED_FLAG"]
        in ("1", "true"),
    },
    "contract_part_skipped": os.environ["SYSTEMONE_SMOKE_SKIP_CONTRACT_FLAG"]
    in ("1", "true"),
    "reasons": [item for item in os.environ["SYSTEMONE_SMOKE_REASONS"].split("; ") if item],
}
path = Path(os.environ["SYSTEMONE_SMOKE_REPORT_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

  case "$status" in
    pass) echo "system-one smoke passed" ;;
    unqualified)
      if [[ "$contract_status" == "skipped" ]]; then
        echo "system-one smoke: contract part skipped; the full-model read is NOT CERTIFIED on this runner"
      else
        echo "system-one smoke: contract part passed; the full-model read is NOT CERTIFIED on this runner"
      fi
      if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::warning title=System One smoke::full-model read NOT CERTIFIED: build_backend=${BUILD_BACKEND:-<unset>} certified_backends=${CERTIFIED_BACKENDS}; report $REPORT_PATH"
      fi
      ;;
    fail)
      echo "system-one smoke failed" >&2
      if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::error title=System One smoke::$(printf '%s' "${REPORT_REASONS:-see $REPORT_PATH}")"
      fi
      ;;
  esac
  echo "system-one report: $REPORT_PATH"
  exit "$exit_status"
}

main "$@"
