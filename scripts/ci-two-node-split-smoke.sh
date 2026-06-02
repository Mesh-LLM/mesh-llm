#!/usr/bin/env bash
# ci-two-node-split-smoke.sh - verify real two-node split serving.
#
# Usage: scripts/ci-two-node-split-smoke.sh <mesh-llm-binary> <bin-dir> <model-path-or-ref>
#
# Unlike ci-two-node-client-serving-smoke.sh, both processes are serving nodes.
# The smoke requires the runtime to publish a topology with stages on at least
# two distinct nodes before it sends an OpenAI chat request through stage 0.

set -euo pipefail

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}"
MODEL="${MESH_TWO_NODE_SPLIT_MODEL:-${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEED_API_PORT="${MESH_TWO_NODE_SPLIT_SEED_API_PORT:-9367}"
SEED_CONSOLE_PORT="${MESH_TWO_NODE_SPLIT_SEED_CONSOLE_PORT:-3161}"
SEED_BIND_PORT="${MESH_TWO_NODE_SPLIT_SEED_BIND_PORT:-53647}"
WORKER_API_PORT="${MESH_TWO_NODE_SPLIT_WORKER_API_PORT:-9368}"
WORKER_CONSOLE_PORT="${MESH_TWO_NODE_SPLIT_WORKER_CONSOLE_PORT:-3162}"
WORKER_BIND_PORT="${MESH_TWO_NODE_SPLIT_WORKER_BIND_PORT:-53648}"
MAX_WAIT="${MESH_TWO_NODE_SPLIT_MAX_WAIT:-300}"
CTX_SIZE="${MESH_TWO_NODE_SPLIT_CTX_SIZE:-}"
MAX_VRAM="${MESH_TWO_NODE_SPLIT_MAX_VRAM:-1}"
DEVICE="${MESH_TWO_NODE_SPLIT_DEVICE:-CPU}"
WORK_DIR="${MESH_TWO_NODE_SPLIT_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/mesh-two-node-split.XXXXXX")}"
# Keep this under /tmp with a short prefix because plugin Unix socket paths
# must fit platform SUN_LEN limits, especially on macOS where TMPDIR is long.
PROCESS_ROOT="${MESH_TWO_NODE_SPLIT_PROCESS_ROOT:-$(mktemp -d "/tmp/m2split.XXXXXX")}"
SEED_LOG="${WORK_DIR}/seed.log"
WORKER_LOG="${WORK_DIR}/worker.log"

echo "=== CI Two-Node Split Smoke ==="
echo "  mesh-llm:       $MESH_LLM"
echo "  bin-dir:        $BIN_DIR (compatibility placeholder)"
echo "  model:          $MODEL"
echo "  seed api:       $SEED_API_PORT"
echo "  seed console:   $SEED_CONSOLE_PORT"
echo "  seed bind:      $SEED_BIND_PORT"
echo "  worker api:     $WORKER_API_PORT"
echo "  worker console: $WORKER_CONSOLE_PORT"
echo "  worker bind:    $WORKER_BIND_PORT"
echo "  ctx size:       ${CTX_SIZE:-model default}"
echo "  max vram:       ${MAX_VRAM}GB"
echo "  device:         $DEVICE"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "Missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi

RUNTIME_CACHE="$("$REPO_ROOT/scripts/ci-install-native-runtime.sh" "$MESH_LLM" "$REPO_ROOT/target/two-node-split-native-runtime" cpu)"
export MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$RUNTIME_CACHE"

descendant_pids() {
    local pid="$1"
    local children
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    for child in $children; do
        descendant_pids "$child"
        printf '%s\n' "$child"
    done
}

kill_tree() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    local children
    children="$(descendant_pids "$pid" | sort -u || true)"
    kill "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill 2>/dev/null || true
    fi
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill -9 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

SEED_PID=""
WORKER_PID=""
cleanup() {
    kill_tree "$WORKER_PID"
    kill_tree "$SEED_PID"
    echo "--- seed log tail ---"
    tail -160 "$SEED_LOG" 2>/dev/null || true
    echo "--- worker log tail ---"
    tail -160 "$WORKER_LOG" 2>/dev/null || true
    echo "--- end logs ---"
    if [[ -z "${MESH_TWO_NODE_SPLIT_WORK_DIR:-}" ]]; then
        rm -rf "$WORK_DIR"
    fi
    if [[ -z "${MESH_TWO_NODE_SPLIT_PROCESS_ROOT:-}" ]]; then
        rm -rf "$PROCESS_ROOT"
    fi
}
trap cleanup EXIT

status_json() {
    local console_port="$1"
    curl -fsS --max-time 5 "http://127.0.0.1:${console_port}/api/status" 2>/dev/null || true
}

stages_json() {
    local console_port="$1"
    curl -fsS --max-time 5 "http://127.0.0.1:${console_port}/api/runtime/stages" 2>/dev/null || true
}

query_token() {
    STATUS_JSON="$1" python3 - <<'PY'
import json
import os

try:
    status = json.loads(os.environ.get("STATUS_JSON", "") or "{}")
except Exception:
    status = {}
print(status.get("token") or "")
PY
}

query_peer_count() {
    STATUS_JSON="$1" python3 - <<'PY'
import json
import os

try:
    status = json.loads(os.environ.get("STATUS_JSON", "") or "{}")
except Exception:
    status = {}
print(len(status.get("peers") or []))
PY
}

query_split_ready() {
    STAGES_JSON="$1" MODELS_JSON="$2" python3 - <<'PY'
import json
import os

try:
    stages = json.loads(os.environ.get("STAGES_JSON", "") or "{}")
except Exception:
    stages = {}
try:
    models = json.loads(os.environ.get("MODELS_JSON", "") or "{}")
except Exception:
    models = {}

nodes = []
stage_count = 0
for topology in stages.get("topologies") or []:
    for stage in topology.get("stages") or []:
        stage_count += 1
        node = stage.get("node_id")
        if node and node not in nodes:
            nodes.append(node)

model_count = len(models.get("data") or [])
ready = stage_count >= 2 and len(nodes) >= 2 and model_count >= 1
print(
    f"ready={str(ready).lower()} stages={stage_count} "
    f"nodes={len(nodes)} models={model_count}"
)
raise SystemExit(0 if ready else 1)
PY
}

start_node() {
    local label="$1"
    local join_token="$2"
    local api_port="$3"
    local console_port="$4"
    local bind_port="$5"
    local log_file="$6"
    local home="${PROCESS_ROOT}/${label}/h"
    local runtime="${PROCESS_ROOT}/${label}/r"
    mkdir -p "$home" "$runtime"

    local -a args=(
        --log-format json
        serve
        --model "$MODEL"
        --split
        --no-draft
        --device "$DEVICE"
        --max-vram "$MAX_VRAM"
        --port "$api_port"
        --console "$console_port"
        --bind-port "$bind_port"
        --headless
    )
    if [[ -n "$join_token" ]]; then
        args+=(--join "$join_token")
    fi
    if [[ -n "$CTX_SIZE" ]]; then
        args+=(--ctx-size "$CTX_SIZE")
    fi

    HOME="$home" \
        MESH_LLM_RUNTIME_ROOT="$runtime" \
        MESH_LLM_EPHEMERAL_KEY=1 \
        "$MESH_LLM" "${args[@]}" >"$log_file" 2>&1 &
    printf '%s\n' "$!"
}

SEED_PID="$(start_node seed "" "$SEED_API_PORT" "$SEED_CONSOLE_PORT" "$SEED_BIND_PORT" "$SEED_LOG")"

TOKEN=""
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$SEED_PID" 2>/dev/null; then
        echo "seed exited unexpectedly" >&2
        tail -160 "$SEED_LOG" >&2 || true
        exit 1
    fi
    TOKEN="$(query_token "$(status_json "$SEED_CONSOLE_PORT")")"
    if [[ -n "$TOKEN" ]]; then
        echo "Seed produced invite token after ${i}s"
        break
    fi
    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "timed out waiting for seed invite token" >&2
        tail -160 "$SEED_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

WORKER_PID="$(start_node worker "$TOKEN" "$WORKER_API_PORT" "$WORKER_CONSOLE_PORT" "$WORKER_BIND_PORT" "$WORKER_LOG")"

DRIVER_LABEL=""
DRIVER_API_PORT=""
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$SEED_PID" 2>/dev/null; then
        echo "seed exited unexpectedly" >&2
        tail -160 "$SEED_LOG" >&2 || true
        exit 1
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "worker exited unexpectedly" >&2
        tail -160 "$WORKER_LOG" >&2 || true
        exit 1
    fi

    for endpoint in \
        "seed:${SEED_API_PORT}:${SEED_CONSOLE_PORT}" \
        "worker:${WORKER_API_PORT}:${WORKER_CONSOLE_PORT}"; do
        IFS=: read -r label api_port console_port <<<"$endpoint"
        PEERS="$(query_peer_count "$(status_json "$console_port")")"
        if [[ "$PEERS" -lt 1 ]]; then
            continue
        fi
        MODELS_JSON="$(curl -fsS --max-time 5 "http://127.0.0.1:${api_port}/v1/models" 2>/dev/null || true)"
        READY_SUMMARY="$(query_split_ready "$(stages_json "$console_port")" "$MODELS_JSON" 2>/dev/null || true)"
        if [[ "$READY_SUMMARY" == ready=true* ]]; then
            DRIVER_LABEL="$label"
            DRIVER_API_PORT="$api_port"
            echo "Split topology ready after ${i}s on ${label}: ${READY_SUMMARY}"
            break 2
        fi
    done

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "timed out waiting for real split topology" >&2
        echo "last checked endpoint: ${label:-unknown}" >&2
        echo "last peer count: ${PEERS:-unknown}" >&2
        echo "last split summary: ${READY_SUMMARY:-unknown}" >&2
        tail -160 "$SEED_LOG" >&2 || true
        tail -160 "$WORKER_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

WORK_PAYLOAD="${WORK_DIR}/chat-payload.json"
CHAT_RESPONSE="${WORK_DIR}/chat-response.json"
if [[ -z "$DRIVER_API_PORT" ]]; then
    echo "no split driver API port was selected" >&2
    exit 1
fi
MODEL_ID="$(
    curl -fsS --max-time 5 "http://127.0.0.1:${DRIVER_API_PORT}/v1/models" |
        python3 -c 'import json,sys; data=json.load(sys.stdin).get("data", []); print(data[0].get("id", "") if data else "")'
)"
if [[ -z "$MODEL_ID" ]]; then
    echo "${DRIVER_LABEL:-selected driver} /v1/models did not return a model id" >&2
    exit 1
fi

python3 - "$MODEL_ID" "$WORK_PAYLOAD" <<'PY'
import json
import sys

model, path = sys.argv[1:3]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with one word."}],
    "stream": False,
    "max_tokens": 8,
    "temperature": 0,
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY

curl -fsS --max-time 180 \
    "http://127.0.0.1:${DRIVER_API_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d @"$WORK_PAYLOAD" \
    -o "$CHAT_RESPONSE"

python3 - "$CHAT_RESPONSE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    body = json.load(fh)
if body.get("object") != "chat.completion":
    raise SystemExit(f"unexpected chat object: {body.get('object')!r}")
if not body.get("choices"):
    raise SystemExit("chat response had no choices")
print("Split chat response validated")
PY

echo "Two-node split smoke passed"
