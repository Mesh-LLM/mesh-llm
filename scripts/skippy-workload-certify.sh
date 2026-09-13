#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_CLASS=""
LANE=""
MODEL_PATH=""
MODEL_ID=""
PROJECTOR_PATH=""
WORK_DIR=""
SKIP_BUILD=0
ORACLE_SERVER=""
ORACLE_COMPLETION=""
ORACLE_TTS=""
ORACLE_REQUIRED=0
STARTUP_TIMEOUT_SECS=180

usage() {
  cat >&2 <<'EOF'
usage: scripts/skippy-workload-certify.sh --class CLASS --lane LANE
  --model-path PATH --model-id ID --work-dir PATH [--projector-path PATH]
  [--oracle-server PATH] [--oracle-completion PATH] [--oracle-tts PATH]
  [--require-oracle]  # fail closed unless the class-appropriate oracle is selected
  [--startup-timeout-secs SECONDS]  # per-server readiness deadline (default: 180)
  [--skip-build]  # oracle runs require a prebuilt SKIPPY_WORKLOAD_PRODUCER_MANIFEST
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --class) MODEL_CLASS="$2"; shift ;;
    --lane) LANE="$2"; shift ;;
    --model-path) MODEL_PATH="$2"; shift ;;
    --model-id) MODEL_ID="$2"; shift ;;
    --projector-path) PROJECTOR_PATH="$2"; shift ;;
    --work-dir) WORK_DIR="$2"; shift ;;
    --oracle-server) ORACLE_SERVER="$2"; shift ;;
    --oracle-completion) ORACLE_COMPLETION="$2"; shift ;;
    --oracle-tts) ORACLE_TTS="$2"; shift ;;
    --require-oracle) ORACLE_REQUIRED=1 ;;
    --startup-timeout-secs) STARTUP_TIMEOUT_SECS="$2"; shift ;;
    --skip-build) SKIP_BUILD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if [[ ! "$STARTUP_TIMEOUT_SECS" =~ ^[1-9][0-9]{0,4}$ ]] || (( STARTUP_TIMEOUT_SECS > 86400 )); then
  echo "--startup-timeout-secs must be a positive integer between 1 and 86400 seconds" >&2
  exit 1
fi

case "$MODEL_CLASS" in
  embedding) EXPECTED_LANE="embedding-smoke" ;;
  rerank) EXPECTED_LANE="rerank-smoke" ;;
  encoder_decoder) EXPECTED_LANE="encoder-decoder-smoke" ;;
  ocr) EXPECTED_LANE="ocr-smoke" ;;
  speech_synthesis) EXPECTED_LANE="speech-synthesis-smoke" ;;
  speech_recognition) EXPECTED_LANE="speech-recognition-smoke" ;;
  *) echo "unsupported model class: $MODEL_CLASS" >&2; exit 1 ;;
esac

if [[ "$LANE" != "$EXPECTED_LANE" ]]; then
  echo "lane $LANE does not match class $MODEL_CLASS (expected $EXPECTED_LANE)" >&2
  exit 1
fi
if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
  echo "model path not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ -z "$MODEL_ID" || -z "$WORK_DIR" ]]; then
  echo "--model-id and --work-dir are required" >&2
  exit 1
fi
if [[ "$MODEL_CLASS" =~ ^(ocr|speech_synthesis|speech_recognition)$ ]] && [[ ! -f "$PROJECTOR_PATH" ]]; then
  echo "class $MODEL_CLASS requires a projector path" >&2
  exit 1
fi
if [[ ( -n "$ORACLE_SERVER" && ( -n "$ORACLE_COMPLETION" || -n "$ORACLE_TTS" ) ) ||
      ( -n "$ORACLE_COMPLETION" && -n "$ORACLE_TTS" ) ]]; then
  echo "select one local-monolithic oracle executable per class" >&2
  exit 1
fi
if (( ORACLE_REQUIRED == 1 )) && [[ -z "$ORACLE_SERVER" && -z "$ORACLE_COMPLETION" && -z "$ORACLE_TTS" ]]; then
  echo "certified workload requires a class-appropriate local-monolithic oracle" >&2
  exit 1
fi
CANDIDATE_BUILD_DIR="${SKIPPY_WORKLOAD_NATIVE_BUILD_DIR:-${LLAMA_STAGE_BUILD_DIR:-$(LLAMA_STAGE_BACKEND="${LLAMA_STAGE_BACKEND:-cpu}" LLAMA_STAGE_LINK_MODE=static "$ROOT/scripts/build-llama.sh" --print-build-dir)}}"
CANDIDATE_BIN_DIR="${SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR:-$ROOT/target/debug}"
PRODUCER_MANIFEST="${SKIPPY_WORKLOAD_PRODUCER_MANIFEST:-}"
TEST_COMMAND=(cargo test --manifest-path "$ROOT/Cargo.toml" -p skippy-server --lib)
require_pinned_cpu_oracle() {
  local executable="$1" expected_name="$2" cmake_option="$3"
  local build_dir stamp candidate_build_dir candidate_stamp patched_sha
  if [[ ! -x "$executable" ]]; then
    echo "oracle executable is not executable: $executable" >&2
    return 1
  fi
  patched_sha="$(python3 "$ROOT/scripts/llama-oracle-source.py")" || return 1
  candidate_build_dir="$CANDIDATE_BUILD_DIR"
  candidate_stamp="$candidate_build_dir/.mesh-llm-build-stamp"
  if [[ ! -f "$candidate_stamp" ]] ||
     ! grep -Fxq "patched-sha=$patched_sha" "$candidate_stamp" ||
     ! grep -Fxq 'backend=cpu' "$candidate_stamp" ||
     ! grep -Fxq 'link-mode=static' "$candidate_stamp" ||
     ! grep -Fxq 'cmake-arg=-DGGML_METAL=OFF' "$candidate_stamp"; then
    echo "candidate static ABI lacks the current pinned CPU llama.cpp build stamp" >&2
    return 1
  fi
  build_dir="$(cd "$(dirname "$executable")/.." && pwd -P)"
  stamp="$build_dir/.mesh-llm-build-stamp"
  if [[ "$(basename "$executable")" != "$expected_name" ]] ||
     [[ ! -f "$stamp" ]] ||
     ! grep -Fxq "patched-sha=$patched_sha" "$stamp" ||
     ! grep -Fxq 'backend=cpu' "$stamp" ||
     ! grep -Fxq 'cmake-arg=-DGGML_METAL=OFF' "$stamp" ||
     ! grep -Fxq "$cmake_option" "$stamp"; then
    echo "oracle executable lacks the current pinned CPU llama.cpp build stamp" >&2
    return 1
  fi
}
if [[ -n "$ORACLE_SERVER" ]]; then
  if [[ ! "$MODEL_CLASS" =~ ^(embedding|rerank|ocr|speech_recognition)$ ]]; then
    echo "class $MODEL_CLASS requires a different local-monolithic oracle executable" >&2
    exit 1
  fi
  require_pinned_cpu_oracle "$ORACLE_SERVER" llama-server 'cmake-arg=-DLLAMA_BUILD_SERVER=ON'
fi
if [[ -n "$ORACLE_COMPLETION" ]]; then
  if [[ "$MODEL_CLASS" != "encoder_decoder" ]]; then
    echo "--oracle-completion is only valid for encoder-decoder models" >&2
    exit 1
  fi
  require_pinned_cpu_oracle "$ORACLE_COMPLETION" llama-completion 'cmake-arg=-DLLAMA_BUILD_TOOLS=ON'
fi
if [[ -n "$ORACLE_TTS" ]]; then
  if [[ "$MODEL_CLASS" != "speech_synthesis" ]]; then
    echo "--oracle-tts is only valid for speech synthesis" >&2
    exit 1
  fi
  require_pinned_cpu_oracle "$ORACLE_TTS" llama-tts 'cmake-arg=-DLLAMA_BUILD_TOOLS=ON'
fi

SDK_PYTHON="${SKIPPY_WORKLOAD_SDK_PYTHON:-python3}"
if [[ "$MODEL_CLASS" == "embedding" ]] && ! "$SDK_PYTHON" -c 'import openai' >/dev/null 2>&1; then
  echo "official openai-python SDK smoke requires the openai package in $SDK_PYTHON" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"
EVIDENCE_PATH="$WORK_DIR/workload-oracle-evidence.json"
COMPARISON_LOG="$WORK_DIR/workload-oracle-comparison.txt"
rm -f "$EVIDENCE_PATH" "$COMPARISON_LOG"
DIMENSIONS="$("$ROOT/scripts/plan-family-battery.py" --inspect-gguf "$MODEL_PATH")"
LAYER_END="$(jq -r '.layer_count' <<<"$DIMENSIONS")"
MODEL_SHA256="$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')"
N_GPU_LAYERS="${SKIPPY_WORKLOAD_N_GPU_LAYERS:-0}"
BACKEND="${LLAMA_STAGE_BACKEND:-cpu}"
if [[ -n "$PRODUCER_MANIFEST" ]]; then
  BACKEND=cpu
fi
if [[ ( -n "$ORACLE_SERVER" || -n "$ORACLE_COMPLETION" || -n "$ORACLE_TTS" ) &&
      ( "$N_GPU_LAYERS" != "0" || "$BACKEND" != "cpu" ) ]]; then
  echo "local-monolithic comparison requires CPU-only candidate execution" >&2
  exit 1
fi
export LLAMA_STAGE_BACKEND="$BACKEND"
if [[ -n "$ORACLE_SERVER" || -n "$ORACLE_COMPLETION" || -n "$ORACLE_TTS" ]]; then
  export LLAMA_STAGE_LINK_MODE=static
  export LLAMA_STAGE_BUILD_DIR="$CANDIDATE_BUILD_DIR"
fi

# The canary explicitly produces a CPU candidate and test binary separately
# from its Metal lane. Consume that immutable, source-bound closure without
# rebuilding or changing the other family lanes' native/Rust outputs.
if [[ -n "$PRODUCER_MANIFEST" ]]; then
  python3 "$ROOT/scripts/check-skippy-workload-candidate.py" \
    --candidate-binary "$CANDIDATE_BIN_DIR/skippy-server" \
    --native-build-dir "$CANDIDATE_BUILD_DIR" --producer-manifest "$PRODUCER_MANIFEST"
  TEST_COMMAND=("$(jq -er '.files.test_binary.path' "$PRODUCER_MANIFEST")")
elif (( SKIP_BUILD == 0 )); then
  LLAMA_STAGE_BUILD_DIR="$CANDIDATE_BUILD_DIR" \
    cargo build -p skippy-server
elif [[ -n "$ORACLE_SERVER" || -n "$ORACLE_COMPLETION" || -n "$ORACLE_TTS" ]]; then
  echo "--skip-build oracle certification requires a source-bound workload producer manifest" >&2
  exit 1
fi
if [[ -n "$ORACLE_SERVER" || -n "$ORACLE_COMPLETION" || -n "$ORACLE_TTS" ]]; then
  python3 "$ROOT/scripts/check-skippy-workload-candidate.py" \
    --candidate-binary "$CANDIDATE_BIN_DIR/skippy-server" \
    --native-build-dir "$CANDIDATE_BUILD_DIR"
fi

MEDIA_PATH=""
case "$MODEL_CLASS" in
  ocr) MEDIA_PATH="$ROOT/ci/llama-canary/fixtures/multimodal-smoke.png" ;;
  speech_recognition) MEDIA_PATH="$ROOT/ci/llama-canary/fixtures/audio-smoke.wav" ;;
esac

TEST_FILTER=frontend::tests::non_chat::real_non_chat_class_smoke_when_fixture_is_set
if [[ -n "$PRODUCER_MANIFEST" ]]; then
  TEST_COMMAND+=("$TEST_FILTER" --nocapture --exact --test-threads=1)
else
  TEST_COMMAND+=("$TEST_FILTER" -- --nocapture --exact --test-threads=1)
fi
env \
  SKIPPY_WORKLOAD_CLASS="$MODEL_CLASS" \
  SKIPPY_WORKLOAD_MODEL="$MODEL_PATH" \
  SKIPPY_WORKLOAD_MODEL_ID="$MODEL_ID" \
  SKIPPY_WORKLOAD_PROJECTOR="$PROJECTOR_PATH" \
  SKIPPY_WORKLOAD_MEDIA="$MEDIA_PATH" \
  SKIPPY_WORKLOAD_LAYER_END="$LAYER_END" \
  SKIPPY_WORKLOAD_CTX_SIZE="${SKIPPY_WORKLOAD_CTX_SIZE:-2048}" \
  SKIPPY_WORKLOAD_MAX_TOKENS="${SKIPPY_WORKLOAD_MAX_TOKENS:-32}" \
  SKIPPY_WORKLOAD_N_GPU_LAYERS="$N_GPU_LAYERS" \
  LLAMA_STAGE_BACKEND="$BACKEND" \
  "${TEST_COMMAND[@]}"

PORT="${SKIPPY_WORKLOAD_OPENAI_PORT:-19337}"
CONFIG_PATH="$WORK_DIR/stage-openai.json"
python3 - "$CONFIG_PATH" "$MODEL_ID" "$MODEL_PATH" "$MODEL_SHA256" "$LAYER_END" "$N_GPU_LAYERS" "$PROJECTOR_PATH" <<'PY'
import json
import sys

config_path, model_id, model_path, model_sha256, layer_end, n_gpu_layers, projector_path = sys.argv[1:]
config = {
    "run_id": "workload-http-smoke",
    "topology_id": "workload-http-smoke-local",
    "model_id": model_id,
    "model_path": model_path,
    "source_model_sha256": model_sha256,
    "stage_id": "stage-0",
    "stage_index": 0,
    "layer_start": 0,
    "layer_end": int(layer_end),
    "ctx_size": 2048,
    "lane_count": 1,
    "n_batch": 2048,
    "n_ubatch": 2048,
    "n_gpu_layers": int(n_gpu_layers),
    "selected_device": ({"backend_device": "CPU"} if int(n_gpu_layers) == 0 else None),
    "kv_offload": (False if int(n_gpu_layers) == 0 else None),
    "op_offload": (False if int(n_gpu_layers) == 0 else None),
    "filter_tensors_on_load": False,
    "native_mtp_enabled": False,
    "load_mode": "runtime-slice",
    "bind_addr": "127.0.0.1:0",
}
if projector_path:
    config["projector_path"] = projector_path
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY

SERVER_LOG="$WORK_DIR/workload-openai-server.log"
LLAMA_STAGE_BACKEND="$BACKEND" \
  "$CANDIDATE_BIN_DIR/skippy-server" serve-openai \
    --config "$CONFIG_PATH" \
    --bind-addr "127.0.0.1:$PORT" \
    --default-max-tokens 128 \
    --telemetry-level off \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
ORACLE_PID=""
cleanup() {
  if [[ -n "$ORACLE_PID" ]] && kill -0 "$ORACLE_PID" >/dev/null 2>&1; then
    kill "$ORACLE_PID" >/dev/null 2>&1 || true
    wait "$ORACLE_PID" >/dev/null 2>&1 || true
  fi
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Both processes receive the same planned wall-clock startup budget, including
# time spent probing the endpoint. An early exit retains the owning log.
wait_for_workload_server() {
  local pid="$1" port="$2" log="$3" label="$4"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECS))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "$MODEL_CLASS $label exited early" >&2
      tail -80 "$log" >&2
      return 1
    fi
    if curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" 2>/dev/null \
      | jq -e --arg model "$MODEL_ID" '.data[]? | select(.id == $model)' >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "$MODEL_CLASS $label was not ready within $STARTUP_TIMEOUT_SECS seconds" >&2
  tail -80 "$log" >&2
  return 1
}
wait_for_workload_server "$SERVER_PID" "$PORT" "$SERVER_LOG" "OpenAI server"
python3 "$ROOT/scripts/ci-openai-workload-smoke.py" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL_ID" \
  --class "$MODEL_CLASS" \
  --media-path "$MEDIA_PATH"

if [[ -n "$ORACLE_SERVER" ]]; then
  ORACLE_PORT="${SKIPPY_WORKLOAD_ORACLE_PORT:-19338}"
  if [[ ! "$ORACLE_PORT" =~ ^[0-9]+$ ]] ||
     (( ORACLE_PORT < 1 || ORACLE_PORT > 65535 || ORACLE_PORT == PORT )); then
    echo "invalid or conflicting oracle port: $ORACLE_PORT" >&2
    exit 1
  fi
  ORACLE_ARGS=(
    -m "$MODEL_PATH" -a "$MODEL_ID" --host 127.0.0.1 --port "$ORACLE_PORT"
    -c 2048 -b 2048 -ub 2048 -ngl 0 --parallel 1 --no-repack
  )
  case "$MODEL_CLASS" in
    embedding) ORACLE_ARGS+=(--embedding) ;;
    rerank) ORACLE_ARGS+=(--embedding --reranking --pooling rank) ;;
    ocr|speech_recognition) ORACLE_ARGS+=(--mmproj "$PROJECTOR_PATH") ;;
  esac
  ORACLE_LOG="$WORK_DIR/workload-monolithic-oracle-server.log"
  "$ORACLE_SERVER" "${ORACLE_ARGS[@]}" >"$ORACLE_LOG" 2>&1 &
  ORACLE_PID="$!"
  wait_for_workload_server "$ORACLE_PID" "$ORACLE_PORT" "$ORACLE_LOG" "monolithic oracle server"
  if [[ "$MODEL_CLASS" =~ ^(ocr|speech_recognition)$ ]]; then
    ORACLE_MEDIA_PATH="$MEDIA_PATH"
    if [[ "$MODEL_CLASS" == "ocr" ]]; then
      ORACLE_MEDIA_PATH="$WORK_DIR/ocr-oracle-mesh-42.png"
      python3 "$ROOT/scripts/generate-ocr-oracle-fixture.py" --output "$ORACLE_MEDIA_PATH"
    fi
    python3 "$ROOT/scripts/skippy-ocr-asr-oracle.py" \
      --candidate-url "http://127.0.0.1:$PORT/v1" \
      --oracle-url "http://127.0.0.1:$ORACLE_PORT/v1" \
      --model "$MODEL_ID" \
      --class "$MODEL_CLASS" \
      --media-path "$ORACLE_MEDIA_PATH" | tee "$COMPARISON_LOG"
  else
    python3 "$ROOT/scripts/ci-workload-monolithic-oracle.py" \
      --candidate-url "http://127.0.0.1:$PORT/v1" \
      --oracle-url "http://127.0.0.1:$ORACLE_PORT/v1" \
      --model "$MODEL_ID" \
      --class "$MODEL_CLASS" | tee "$COMPARISON_LOG"
  fi
fi

if [[ -n "$ORACLE_COMPLETION" ]]; then
  python3 "$ROOT/scripts/ci-workload-monolithic-oracle.py" \
    --candidate-url "http://127.0.0.1:$PORT/v1" \
    --oracle-completion "$ORACLE_COMPLETION" \
    --model-path "$MODEL_PATH" \
    --model "$MODEL_ID" \
    --class "$MODEL_CLASS" | tee "$COMPARISON_LOG"
fi

if [[ -n "$ORACLE_TTS" ]]; then
  python3 "$ROOT/scripts/skippy-tts-oracle.py" \
    --oracle-cli "$ORACLE_TTS" \
    --model-path "$MODEL_PATH" \
    --projector-path "$PROJECTOR_PATH" \
    --model "$MODEL_ID" \
    --layer-end "$LAYER_END" \
    --work-dir "$WORK_DIR" | tee "$COMPARISON_LOG"
fi

if [[ "$MODEL_CLASS" == "embedding" ]]; then
  "$SDK_PYTHON" "$ROOT/scripts/ci-openai-embeddings-smoke.py" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL_ID"
fi

if [[ -n "$ORACLE_SERVER" || -n "$ORACLE_COMPLETION" || -n "$ORACLE_TTS" ]]; then
  ORACLE_EXECUTABLE="${ORACLE_SERVER:-${ORACLE_COMPLETION:-$ORACLE_TTS}}"
  evidence_command=(python3 "$ROOT/scripts/write-workload-oracle-evidence.py"
    --output "$EVIDENCE_PATH"
    --comparison-log "$COMPARISON_LOG"
    --class "$MODEL_CLASS"
    --smoke-lane "$EXPECTED_LANE"
    --model-id "$MODEL_ID"
    --model-sha256 "$MODEL_SHA256"
    --candidate-executable "$CANDIDATE_BIN_DIR/skippy-server"
    --oracle-executable "$ORACLE_EXECUTABLE"
    --pinned-patch-sha "$(python3 "$ROOT/scripts/llama-oracle-source.py")"
    --work-dir "$WORK_DIR")
  if [[ -n "$PROJECTOR_PATH" ]]; then
    evidence_command+=(--projector-path "$PROJECTOR_PATH")
  fi
  "${evidence_command[@]}"
fi
