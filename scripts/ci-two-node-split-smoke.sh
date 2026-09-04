#!/usr/bin/env bash
# ci-two-node-split-smoke.sh - verify real two-node split serving.
#
# Usage: scripts/ci-two-node-split-smoke.sh <mesh-llm-binary> <bin-dir> <model-path-or-ref>
#
# Unlike ci-two-node-client-serving-smoke.sh, both processes are serving nodes.
# The smoke requires the runtime to publish a topology with stages on at least
# two distinct nodes before it sends OpenAI requests through stage 0. It then
# grows one raw prompt over three requests and requires the split stages to
# restore a longer shared prefix after every request.

set -euo pipefail

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}"
MODEL="${MESH_TWO_NODE_SPLIT_MODEL:-${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}}"

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

PREFIX_PAYLOAD_DIR="${WORK_DIR}/prefix-payloads"
PREFIX_RESPONSE_DIR="${WORK_DIR}/prefix-responses"
mkdir -p "$PREFIX_PAYLOAD_DIR" "$PREFIX_RESPONSE_DIR"

python3 - "$MODEL_ID" "$PREFIX_PAYLOAD_DIR" <<'PY'
import json
from pathlib import Path
import sys

model, output_dir = sys.argv[1:3]
output = Path(output_dir)
shared = (
    "Split prefix cache smoke shared context. "
    "Every request keeps these tokens in the same order. "
) * 48
extensions = [
    "First extension block remains reusable by later prompts. " * 16,
    "Second extension block makes the reusable prefix longer. " * 16,
    "Third extension block proves reuse keeps growing. " * 16,
]
prompt = shared
for index, extension in enumerate(extensions, start=1):
    prompt += extension
    payload = {
        "model": model,
        "prompt": prompt,
        "user": "ci-split-prefix-growth",
        "stream": False,
        "max_tokens": 1,
        "temperature": 0,
    }
    with (output / f"prompt-{index}.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
PY

for index in 1 2 3; do
    curl -fsS --max-time 180 \
        "http://127.0.0.1:${DRIVER_API_PORT}/v1/completions" \
        -H 'content-type: application/json' \
        -d @"${PREFIX_PAYLOAD_DIR}/prompt-${index}.json" \
        -o "${PREFIX_RESPONSE_DIR}/response-${index}.json"
    # The host returns the OpenAI response before the stage connection has
    # released its single CI lane. Give graceful Stop enough time to finish so
    # the next request tests cache reuse rather than transient admission.
    sleep 1
done

python3 - "$PREFIX_RESPONSE_DIR" <<'PY'
import json
from pathlib import Path
import sys

response_dir = Path(sys.argv[1])
metrics = []
for index in range(1, 4):
    with (response_dir / f"response-{index}.json").open(encoding="utf-8") as fh:
        body = json.load(fh)
    if body.get("object") != "text_completion":
        raise SystemExit(
            f"prefix request {index} returned unexpected object: {body.get('object')!r}"
        )
    if not body.get("choices"):
        raise SystemExit(f"prefix request {index} returned no choices")
    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = details.get("cached_tokens", 0)
    if not isinstance(prompt_tokens, int) or not isinstance(cached_tokens, int):
        raise SystemExit(f"prefix request {index} omitted numeric cache usage: {usage!r}")
    metrics.append((prompt_tokens, cached_tokens))

prompt_counts = [prompt for prompt, _ in metrics]
cached_counts = [cached for _, cached in metrics]
if not prompt_counts[0] < prompt_counts[1] < prompt_counts[2]:
    raise SystemExit(f"prompt token counts did not increase: {prompt_counts}")
if cached_counts[0] != 0:
    raise SystemExit(f"cold prefix request unexpectedly restored tokens: {cached_counts}")
if not 0 < cached_counts[1] < cached_counts[2]:
    raise SystemExit(f"split prefix reuse did not increase: {cached_counts}")
if any(cached >= prompt for prompt, cached in metrics[1:]):
    raise SystemExit(f"growing prompts must retain an uncached suffix: {metrics}")

print(
    "Split prefix cache reuse increased: "
    + ", ".join(
        f"request {index}: prompt_tokens={prompt}, cached_tokens={cached}"
        for index, (prompt, cached) in enumerate(metrics, start=1)
    )
)
PY

echo "Two-node split smoke passed"
