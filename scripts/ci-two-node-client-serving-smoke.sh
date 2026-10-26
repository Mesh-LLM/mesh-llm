#!/usr/bin/env bash
# ci-two-node-client-serving-smoke.sh - verify client-to-serving-node routing.
#
# Usage: scripts/ci-two-node-client-serving-smoke.sh <mesh-llm-binary> <bin-dir> <model-path>
#
# The host serves the model. A passive client joins the host and exposes its own
# OpenAI-compatible API. The smoke targets the client API so requests exercise
# client -> host mesh routing and tunneling.

set -euo pipefail
automation=(cargo xtool)
if [[ -n "${MESH_LLM_AUTOMATION_BIN:-}" ]]; then
    automation=("$MESH_LLM_AUTOMATION_BIN")
fi

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
MODEL="${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"

HOST_API_PORT="${MESH_TWO_NODE_HOST_API_PORT:-9357}"
HOST_CONSOLE_PORT="${MESH_TWO_NODE_HOST_CONSOLE_PORT:-3151}"
HOST_BIND_PORT="${MESH_TWO_NODE_HOST_BIND_PORT:-53547}"
CLIENT_API_PORT="${MESH_TWO_NODE_CLIENT_API_PORT:-9358}"
CLIENT_CONSOLE_PORT="${MESH_TWO_NODE_CLIENT_CONSOLE_PORT:-3152}"
MAX_WAIT="${MESH_TWO_NODE_MAX_WAIT:-240}"
HOST_LOG="${MESH_TWO_NODE_HOST_LOG:-/tmp/mesh-llm-two-node-host.log}"
CLIENT_LOG="${MESH_TWO_NODE_CLIENT_LOG:-/tmp/mesh-llm-two-node-client.log}"
CLIENT_STABLE_PROBES="${MESH_TWO_NODE_CLIENT_STABLE_PROBES:-5}"

echo "=== CI Two-Node Client/Serving Smoke ==="
echo "  mesh-llm:       $MESH_LLM"
echo "  bin-dir:        $BIN_DIR (compatibility placeholder)"
echo "  model:          $MODEL"
echo "  host api:       $HOST_API_PORT"
echo "  host console:   $HOST_CONSOLE_PORT"
echo "  host bind:      $HOST_BIND_PORT"
echo "  client api:     $CLIENT_API_PORT"
echo "  client console: $CLIENT_CONSOLE_PORT"
echo "  stable probes:  $CLIENT_STABLE_PROBES"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "Missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "Missing model: $MODEL" >&2
    exit 1
fi

RUNTIME_BUNDLE="${MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR:-$(cd "$(dirname "$MESH_LLM")" && pwd)/native-runtimes}"
if [[ ! -d "$RUNTIME_BUNDLE" ]]; then
    echo "Missing packaged native runtime beside mesh-llm: $RUNTIME_BUNDLE" >&2
    exit 1
fi
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$RUNTIME_BUNDLE"

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

HOST_PID=""
CLIENT_PID=""
cleanup() {
    kill_tree "$CLIENT_PID"
    kill_tree "$HOST_PID"
    echo "--- host log tail ---"
    tail -120 "$HOST_LOG" 2>/dev/null || true
    echo "--- client log tail ---"
    tail -120 "$CLIENT_LOG" 2>/dev/null || true
    echo "--- end logs ---"
}
trap cleanup EXIT

"$MESH_LLM" \
    --log-format json \
    serve \
    --model "$MODEL" \
    --no-draft \
    --device CPU \
    --ctx-size "${MESH_TWO_NODE_CTX_SIZE:-1024}" \
    --port "$HOST_API_PORT" \
    --console "$HOST_CONSOLE_PORT" \
    --bind-port "$HOST_BIND_PORT" \
    --headless \
    >"$HOST_LOG" 2>&1 &
HOST_PID=$!

TOKEN=""
HOST_MODEL_ID=""
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$HOST_PID" 2>/dev/null; then
        echo "host exited unexpectedly" >&2
        tail -120 "$HOST_LOG" >&2 || true
        exit 1
    fi

    STATUS_JSON="$(curl -sf "http://127.0.0.1:${HOST_CONSOLE_PORT}/api/status" 2>/dev/null || true)"
    READY="$(
        printf '%s' "$STATUS_JSON" | "${automation[@]}" automation smoke-observation ready 2>/dev/null || echo "False"
    )"
    TOKEN="$(
        printf '%s' "$STATUS_JSON" | "${automation[@]}" automation smoke-observation token 2>/dev/null || echo ""
    )"

    if [[ "$READY" == "True" && -n "$TOKEN" ]]; then
        MODELS_JSON="$(curl -sf "http://127.0.0.1:${HOST_API_PORT}/v1/models")"
        HOST_MODEL_ID="$(
            printf '%s' "$MODELS_JSON" | "${automation[@]}" automation smoke-observation first-model
        )"
        if [[ -n "$HOST_MODEL_ID" ]]; then
            echo "Host ready after ${i}s with model: $HOST_MODEL_ID"
            break
        fi
    fi

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "timed out waiting for host readiness" >&2
        tail -120 "$HOST_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

"$MESH_LLM" \
    --log-format json \
    client \
    --join "$TOKEN" \
    --port "$CLIENT_API_PORT" \
    --console "$CLIENT_CONSOLE_PORT" \
    --headless \
    >"$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!

CLIENT_STABLE_COUNT=0
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
        echo "client exited unexpectedly" >&2
        tail -120 "$CLIENT_LOG" >&2 || true
        exit 1
    fi

    CLIENT_MODELS_JSON="$(curl -sf "http://127.0.0.1:${CLIENT_API_PORT}/v1/models" 2>/dev/null || true)"
    if printf '%s' "$CLIENT_MODELS_JSON" | "${automation[@]}" automation smoke-observation has-model "$HOST_MODEL_ID" 2>/dev/null; then
        CLIENT_STABLE_COUNT=$((CLIENT_STABLE_COUNT + 1))
        if [[ "$CLIENT_STABLE_COUNT" -ge "$CLIENT_STABLE_PROBES" ]]; then
            echo "Client routed /v1/models stably after ${i}s"
            break
        fi
    else
        CLIENT_STABLE_COUNT=0
    fi

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "timed out waiting for stable client /v1/models to include host model" >&2
        echo "$CLIENT_MODELS_JSON" >&2
        tail -120 "$CLIENT_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mesh-two-node-client-serving.XXXXXX")"
CHAT_RESPONSE="${WORK_DIR}/chat-response.json"
STREAM_RESPONSE="${WORK_DIR}/chat-stream.txt"

"${automation[@]}" automation smoke-observation chat-payload "$HOST_MODEL_ID" "${WORK_DIR}/chat-payload.json"

curl -fsS --max-time 120 \
    "http://127.0.0.1:${CLIENT_API_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d @"${WORK_DIR}/chat-payload.json" \
    -o "$CHAT_RESPONSE"

"${automation[@]}" automation smoke-observation chat <"$CHAT_RESPONSE"

"${automation[@]}" automation smoke-observation stream-payload "$HOST_MODEL_ID" "${WORK_DIR}/stream-payload.json"

curl -fsS --max-time 120 \
    "http://127.0.0.1:${CLIENT_API_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d @"${WORK_DIR}/stream-payload.json" \
    -o "$STREAM_RESPONSE"

"${automation[@]}" automation smoke-observation stream <"$STREAM_RESPONSE"

echo "Two-node client/serving smoke passed"
