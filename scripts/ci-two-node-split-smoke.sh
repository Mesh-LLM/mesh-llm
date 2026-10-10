#!/usr/bin/env bash
# ci-two-node-split-smoke.sh - verify real two-node split serving.
#
# Usage: scripts/ci-two-node-split-smoke.sh <mesh-llm-binary> <bin-dir> <model-path-or-ref>
#
# Unlike ci-two-node-client-serving-smoke.sh, both processes are serving nodes.
# The smoke requires the runtime to publish a topology with stages on at least
# two distinct nodes before it sends OpenAI requests through stage 0. For each
# model leg it sends every prompt length twice in a row (X, X, X+more, X+more,
# ...): the first sight of each length proves reuse keeps growing, and the
# identical re-send must restore more of the prompt from cache. Dense KV cache
# repeats must be near-full restores; recurrent KV cache repeats may restore
# from the latest checkpoint. With MESH_TWO_NODE_SPLIT_RECURRENT_MODEL set, the
# dense leg runs first and the recurrent leg repeats the whole flow against a
# second model in the same job.

set -euo pipefail
automation=(cargo xtool)
if [[ -n "${MESH_LLM_AUTOMATION_BIN:-}" ]]; then
    automation=("$MESH_LLM_AUTOMATION_BIN")
fi

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}"
MODEL="${MESH_TWO_NODE_SPLIT_MODEL:-${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}}"

SEED_API_PORT="${MESH_TWO_NODE_SPLIT_SEED_API_PORT:-9367}"
SEED_CONSOLE_PORT="${MESH_TWO_NODE_SPLIT_SEED_CONSOLE_PORT:-3161}"
SEED_BIND_PORT="${MESH_TWO_NODE_SPLIT_SEED_BIND_PORT:-53647}"
WORKER_API_PORT="${MESH_TWO_NODE_SPLIT_WORKER_API_PORT:-9368}"
WORKER_CONSOLE_PORT="${MESH_TWO_NODE_SPLIT_WORKER_CONSOLE_PORT:-3162}"
WORKER_BIND_PORT="${MESH_TWO_NODE_SPLIT_WORKER_BIND_PORT:-53648}"
READINESS_TIMEOUT_SECONDS="${MESH_TWO_NODE_SPLIT_READINESS_TIMEOUT_SECONDS:-${MESH_TWO_NODE_SPLIT_MAX_WAIT:-300}}"
SNAPSHOT_REQUEST_TIMEOUT_SECONDS="${MESH_TWO_NODE_SPLIT_SNAPSHOT_REQUEST_TIMEOUT_SECONDS:-2}"
if [[ ! "$READINESS_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    [[ "$READINESS_TIMEOUT_SECONDS" -gt 300 ]]; then
    echo "MESH_TWO_NODE_SPLIT_READINESS_TIMEOUT_SECONDS must be an integer from 1 through 300" >&2
    exit 2
fi
if [[ ! "$SNAPSHOT_REQUEST_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    [[ "$SNAPSHOT_REQUEST_TIMEOUT_SECONDS" -gt 10 ]]; then
    echo "MESH_TWO_NODE_SPLIT_SNAPSHOT_REQUEST_TIMEOUT_SECONDS must be an integer from 1 through 10" >&2
    exit 2
fi
# Optional second model: when set, the smoke runs the whole split flow once
# against MODEL (dense) and once against this (recurrent), restarting both
# processes between legs so a single CI job covers both cache families.
RECURRENT_MODEL="${MESH_TWO_NODE_SPLIT_RECURRENT_MODEL:-}"
RECURRENT_MODEL_FILE="${MESH_TWO_NODE_SPLIT_RECURRENT_MODEL_FILE:-}"
RECURRENT_CTX_SIZE="${MESH_TWO_NODE_SPLIT_RECURRENT_CTX_SIZE:-4096}"
RECURRENT_EXPECTED_EXACT_PAYLOAD_KIND="${MESH_TWO_NODE_SPLIT_RECURRENT_EXPECTED_EXACT_PAYLOAD_KIND:-}"
if [[ -z "$RECURRENT_MODEL" && -n "$RECURRENT_MODEL_FILE" ]]; then
    RECURRENT_MODEL="${HOME}/.models/${RECURRENT_MODEL_FILE}"
fi
REQUEST_SETTLE_SECONDS="${MESH_TWO_NODE_SPLIT_REQUEST_SETTLE_SECONDS:-1}"
PREFIX_ATTEMPTS="${MESH_TWO_NODE_SPLIT_PREFIX_ATTEMPTS:-3}"
EXPECTED_EXACT_PAYLOAD_KIND="${MESH_TWO_NODE_SPLIT_EXPECTED_EXACT_PAYLOAD_KIND:-}"
CTX_SIZE="${MESH_TWO_NODE_SPLIT_CTX_SIZE:-${MESH_LLM_SMOKE_CONTEXT_SIZE:-}}"
MAX_VRAM="${MESH_TWO_NODE_SPLIT_MAX_VRAM:-1}"
DEVICE="${MESH_TWO_NODE_SPLIT_DEVICE:-CPU}"
WORK_DIR="${MESH_TWO_NODE_SPLIT_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/mesh-two-node-split.XXXXXX")}"
mkdir -p "$WORK_DIR"
# Keep this under /tmp with a short prefix because plugin Unix socket paths
# must fit platform SUN_LEN limits, especially on macOS where TMPDIR is long.
PROCESS_ROOT="${MESH_TWO_NODE_SPLIT_PROCESS_ROOT:-$(mktemp -d "/tmp/m2split.XXXXXX")}"
CLIENT_ROUTING="${MESH_TWO_NODE_SPLIT_CLIENT_ROUTING:-0}"
DURABLE_L3="${MESH_TWO_NODE_SPLIT_DURABLE_L3:-0}"
DURABLE_L3_ROOT="${MESH_TWO_NODE_SPLIT_DURABLE_L3_ROOT:-${WORK_DIR}/durable-l3-roots}"
DURABLE_L3_EVIDENCE_PATH="${WORK_DIR}/durable-l3-evidence.json"
DURABLE_L3_RECORDS="${WORK_DIR}/.durable-l3-evidence.jsonl"
DENSE_ARTIFACT_ID="${MESH_TWO_NODE_SPLIT_DENSE_ARTIFACT_ID:-unspecified}"
DENSE_MODEL_SHA256="${MESH_TWO_NODE_SPLIT_DENSE_SHA256:-unspecified}"
RECURRENT_ARTIFACT_ID="${MESH_TWO_NODE_SPLIT_RECURRENT_ARTIFACT_ID:-unspecified}"
RECURRENT_MODEL_SHA256="${MESH_TWO_NODE_SPLIT_RECURRENT_SHA256:-unspecified}"
# PR smoke callers execute from the protected default-branch workflow, so a
# branch-local caller input cannot authorize a newly added unsafe flag until
# that workflow change lands. Infer the narrow test-only default from the
# inputs instead: this harness creates a fresh, unlisted package identity only
# when it receives a local GGUF file. Existing package-v2 inputs remain
# fail-closed unless their caller explicitly opts in.
AUTO_ALLOW_UNCERTIFIED_SPLIT=0
if [[ -f "$MODEL" ]] || [[ -n "$RECURRENT_MODEL" && -f "$RECURRENT_MODEL" ]]; then
    AUTO_ALLOW_UNCERTIFIED_SPLIT=1
fi
ALLOW_UNCERTIFIED_SPLIT="${MESH_TWO_NODE_SPLIT_ALLOW_UNCERTIFIED:-$AUTO_ALLOW_UNCERTIFIED_SPLIT}"
CLIENT_API_PORT="${MESH_TWO_NODE_SPLIT_CLIENT_API_PORT:-9369}"
CLIENT_CONSOLE_PORT="${MESH_TWO_NODE_SPLIT_CLIENT_CONSOLE_PORT:-3163}"
PRIMARY_MODEL_LABEL="${MESH_TWO_NODE_SPLIT_MODEL_LABEL:-}"
if [[ -z "$PRIMARY_MODEL_LABEL" ]]; then
    if [[ "$EXPECTED_EXACT_PAYLOAD_KIND" == "kv-recurrent" ]]; then
        PRIMARY_MODEL_LABEL="recurrent"
    else
        PRIMARY_MODEL_LABEL="dense"
    fi
fi
SEED_LOG="${WORK_DIR}/${PRIMARY_MODEL_LABEL}-seed.log"
WORKER_LOG="${WORK_DIR}/${PRIMARY_MODEL_LABEL}-worker.log"
CLIENT_LOG="${WORK_DIR}/${PRIMARY_MODEL_LABEL}-client.log"
MODEL_LABEL="$PRIMARY_MODEL_LABEL"
SPLIT_EVIDENCE_PATH=""
SPLIT_SNAPSHOT_DIR=""
SPLIT_RECONCILE_LOG=""

if [[ "$DURABLE_L3" != "0" && "$DURABLE_L3" != "1" ]]; then
    echo "MESH_TWO_NODE_SPLIT_DURABLE_L3 must be 0 or 1" >&2
    exit 2
fi
if [[ "$DURABLE_L3" == "1" ]]; then
    mkdir -p "$DURABLE_L3_ROOT"
    : >"$DURABLE_L3_RECORDS"
fi

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
echo "  readiness timeout: ${READINESS_TIMEOUT_SECONDS}s"
echo "  snapshot request timeout: ${SNAPSHOT_REQUEST_TIMEOUT_SECONDS}s"
echo "  request settle: ${REQUEST_SETTLE_SECONDS}s"
echo "  prefix attempts: ${PREFIX_ATTEMPTS}"
echo "  expected exact payload: ${EXPECTED_EXACT_PAYLOAD_KIND:-none}"
echo "  ctx size:       ${CTX_SIZE:-model default}"
echo "  max vram:       ${MAX_VRAM}GB"
echo "  device:         $DEVICE"
echo "  client routing: $CLIENT_ROUTING"
echo "  durable L3:     $DURABLE_L3"
echo "  uncertified split override: $ALLOW_UNCERTIFIED_SPLIT"

if [[ "$ALLOW_UNCERTIFIED_SPLIT" != "0" && "$ALLOW_UNCERTIFIED_SPLIT" != "1" ]]; then
    echo "MESH_TWO_NODE_SPLIT_ALLOW_UNCERTIFIED must be 0 or 1" >&2
    exit 2
fi

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

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    else
        shasum -a 256 "$path" | awk '{print $1}'
    fi
}

quant_selector_from_gguf_file() {
    local filename="$1"
    "${automation[@]}" automation split-probe quant "$filename"
}

resolve_package_tool() {
    "${automation[@]}" automation split-probe package-tool "$RUNTIME_BUNDLE"
}

prepare_split_package() {
    local label="$1"
    local source="$2"
    local package_tool="$3"

    if [[ -d "$source" && -s "$source/model-package.json" ]]; then
        printf '%s\n' "$source"
        return 0
    fi
    [[ -f "$source" ]] || {
        echo "Generation-9 split smoke input must be a package-v2 directory or local GGUF: $source" >&2
        return 1
    }

    local source_file selector digest package_dir model_id
    source_file="$(basename "$source")"
    selector="$(quant_selector_from_gguf_file "$source_file")"
    digest="$(sha256_file "$source")"
    package_dir="$WORK_DIR/prepared-packages/$label"
    model_id="ci/${label}-$(printf '%s' "$digest" | cut -c1-16):${selector}"
    rm -rf "$package_dir"
    mkdir -p "$(dirname "$package_dir")"

    local write_log="$WORK_DIR/${label}-write-package.log"
    local verify_log="$WORK_DIR/${label}-verify-package-v2.log"
    if ! "$package_tool" write-package "$source" \
        --model-id "$model_id" \
        --out-dir "$package_dir" \
        --source-repo "ci/two-node-split-smoke" \
        --source-revision "$digest" \
        --source-file "$source_file" >"$write_log" 2>&1; then
        echo "Package-v2 preparation failed for $source; see $write_log" >&2
        return 1
    fi
    if ! "$package_tool" verify-package-v2 "$package_dir" \
        --source "$source" \
        --source-file "$source_file" >"$verify_log" 2>&1; then
        echo "Package-v2 verification failed for $source; see $verify_log" >&2
        return 1
    fi
    [[ -s "$package_dir/model-package.json" ]] || {
        echo "Package-v2 preparation did not emit $package_dir/model-package.json" >&2
        return 1
    }
    printf '%s\n' "$package_dir"
}

prepare_split_inputs() {
    local package_tool
    if [[ -d "$MODEL" && -s "$MODEL/model-package.json" ]] &&
        { [[ -z "$RECURRENT_MODEL" ]] || [[ -d "$RECURRENT_MODEL" && -s "$RECURRENT_MODEL/model-package.json" ]]; }; then
        return 0
    fi
    package_tool="$(resolve_package_tool)"
    MODEL="$(prepare_split_package "$PRIMARY_MODEL_LABEL" "$MODEL" "$package_tool")"
    if [[ -n "$RECURRENT_MODEL" ]]; then
        RECURRENT_MODEL="$(prepare_split_package recurrent "$RECURRENT_MODEL" "$package_tool")"
    fi
}

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
    if command -v taskkill.exe >/dev/null 2>&1; then
        taskkill.exe //PID "$pid" //T //F >/dev/null 2>&1 || true
        wait "$pid" 2>/dev/null || true
        return 0
    fi
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
CLIENT_PID=""
cleanup() {
    kill_tree "$CLIENT_PID"
    kill_tree "$WORKER_PID"
    kill_tree "$SEED_PID"
    echo "--- seed log tail ---"
    tail -1200 "$SEED_LOG" 2>/dev/null || true
    echo "--- worker log tail ---"
    tail -1200 "$WORKER_LOG" 2>/dev/null || true
    echo "--- client log tail ---"
    tail -1200 "$CLIENT_LOG" 2>/dev/null || true
    echo "--- end logs ---"
    if [[ -z "${MESH_TWO_NODE_SPLIT_WORK_DIR:-}" ]]; then
        rm -rf "$WORK_DIR"
    fi
    if [[ -z "${MESH_TWO_NODE_SPLIT_PROCESS_ROOT:-}" ]]; then
        rm -rf "$PROCESS_ROOT"
    fi
}
trap cleanup EXIT

prepare_split_inputs

status_json() {
    local console_port="$1"
    local request_timeout="${2:-$SNAPSHOT_REQUEST_TIMEOUT_SECONDS}"
    curl -fsS --connect-timeout "$request_timeout" \
        --max-time "$request_timeout" \
        "http://127.0.0.1:${console_port}/api/status" 2>/dev/null || true
}

query_token() {
    printf '%s' "$1" | "${automation[@]}" automation smoke-observation token 2>/dev/null || true
}

wait_for_seed_token() {
    local context="$1"
    local started_at
    local deadline
    local now
    local remaining
    local request_timeout

    TOKEN=""
    started_at="$(date +%s)"
    deadline=$((started_at + READINESS_TIMEOUT_SECONDS))
    while :; do
        if ! kill -0 "$SEED_PID" 2>/dev/null; then
            echo "${context}seed exited unexpectedly" >&2
            tail -160 "$SEED_LOG" >&2 || true
            return 1
        fi
        now="$(date +%s)"
        remaining=$((deadline - now))
        if [[ "$remaining" -le 0 ]]; then
            echo "${context}timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for seed invite token" >&2
            tail -160 "$SEED_LOG" >&2 || true
            return 1
        fi
        request_timeout="$SNAPSHOT_REQUEST_TIMEOUT_SECONDS"
        if [[ "$remaining" -lt "$request_timeout" ]]; then
            request_timeout="$remaining"
        fi
        TOKEN="$(query_token "$(status_json "$SEED_CONSOLE_PORT" "$request_timeout")")"
        now="$(date +%s)"
        if [[ "$now" -ge "$deadline" ]]; then
            echo "${context}timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for seed invite token" >&2
            tail -160 "$SEED_LOG" >&2 || true
            return 1
        fi
        if [[ -n "$TOKEN" ]]; then
            echo "Seed produced invite token after $((now - started_at))s${context:+ (${context%: })}"
            return 0
        fi
        sleep 1
    done
}

configure_split_evidence_paths() {
    local label="$1"
    local prefix=""
    if [[ "$label" != "$PRIMARY_MODEL_LABEL" ]]; then
        prefix="${label}-"
    fi
    SPLIT_EVIDENCE_PATH="${WORK_DIR}/${prefix}split-evidence.json"
    SPLIT_SNAPSHOT_DIR="${WORK_DIR}/${prefix}split-evidence-snapshots"
    SPLIT_RECONCILE_LOG="${WORK_DIR}/${prefix}split-evidence-reconcile.log"
    mkdir -p "$SPLIT_SNAPSHOT_DIR"
}

capture_json_snapshot() {
    local kind="$1"
    local url="$2"
    local output="$3"
    local request_timeout="$4"
    local raw="${output}.curl.$$.tmp"

    if ! curl -fsS --connect-timeout "$request_timeout" \
        --max-time "$request_timeout" "$url" >"$raw" 2>/dev/null; then
        : >"$raw"
    fi
    "${automation[@]}" automation split-probe snapshot "$kind" "$raw" "$output"
    rm -f "$raw"
}

capture_split_snapshots() {
    local request_timeout="$1"
    local -a capture_pids=()
    local capture_pid

    capture_json_snapshot status \
        "http://127.0.0.1:${SEED_CONSOLE_PORT}/api/status" \
        "$SPLIT_SNAPSHOT_DIR/seed-status.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot stages \
        "http://127.0.0.1:${SEED_CONSOLE_PORT}/api/runtime/stages" \
        "$SPLIT_SNAPSHOT_DIR/seed-stages.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot models \
        "http://127.0.0.1:${SEED_API_PORT}/v1/models" \
        "$SPLIT_SNAPSHOT_DIR/seed-models.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot status \
        "http://127.0.0.1:${WORKER_CONSOLE_PORT}/api/status" \
        "$SPLIT_SNAPSHOT_DIR/worker-status.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot stages \
        "http://127.0.0.1:${WORKER_CONSOLE_PORT}/api/runtime/stages" \
        "$SPLIT_SNAPSHOT_DIR/worker-stages.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot models \
        "http://127.0.0.1:${WORKER_API_PORT}/v1/models" \
        "$SPLIT_SNAPSHOT_DIR/worker-models.json" "$request_timeout" &
    capture_pids+=("$!")

    for capture_pid in "${capture_pids[@]}"; do
        wait "$capture_pid"
    done
}

reconcile_split_snapshots() {
    "${automation[@]}" automation split-evidence \
        --seed-status "$SPLIT_SNAPSHOT_DIR/seed-status.json" \
        --seed-stages "$SPLIT_SNAPSHOT_DIR/seed-stages.json" \
        --seed-models "$SPLIT_SNAPSHOT_DIR/seed-models.json" \
        --worker-status "$SPLIT_SNAPSHOT_DIR/worker-status.json" \
        --worker-stages "$SPLIT_SNAPSHOT_DIR/worker-stages.json" \
        --worker-models "$SPLIT_SNAPSHOT_DIR/worker-models.json" \
        --model-label "$MODEL_LABEL" \
        --output "$SPLIT_EVIDENCE_PATH"
}

report_split_readiness_failure() {
    local context="$1"
    local reason="$2"
    echo "${context}${reason}" >&2
    for snapshot in \
        "$SPLIT_SNAPSHOT_DIR/seed-status.json" \
        "$SPLIT_SNAPSHOT_DIR/seed-stages.json" \
        "$SPLIT_SNAPSHOT_DIR/seed-models.json" \
        "$SPLIT_SNAPSHOT_DIR/worker-status.json" \
        "$SPLIT_SNAPSHOT_DIR/worker-stages.json" \
        "$SPLIT_SNAPSHOT_DIR/worker-models.json" \
        "$SPLIT_EVIDENCE_PATH"; do
        echo "--- ${snapshot} at timeout ---" >&2
        cat "$snapshot" >&2 2>/dev/null || true
    done
    echo "--- split evidence reconciler error at timeout ---" >&2
    cat "$SPLIT_RECONCILE_LOG" >&2 2>/dev/null || true
    echo "--- seed log tail at timeout ---" >&2
    tail -1200 "$SEED_LOG" >&2 || true
    echo "--- worker log tail at timeout ---" >&2
    tail -1200 "$WORKER_LOG" >&2 || true
}

wait_for_split_topology() {
    local context="$1"
    local started_at
    local deadline
    local now
    local remaining
    local request_timeout
    local reconciliation_ready
    configure_split_evidence_paths "$MODEL_LABEL"
    started_at="$(date +%s)"
    deadline=$((started_at + READINESS_TIMEOUT_SECONDS))
    while :; do
        if ! kill -0 "$SEED_PID" 2>/dev/null; then
            capture_split_snapshots 1
            reconcile_split_snapshots 2>"$SPLIT_RECONCILE_LOG" || true
            report_split_readiness_failure "$context" \
                "seed exited unexpectedly before split readiness"
            return 1
        fi
        if ! kill -0 "$WORKER_PID" 2>/dev/null; then
            capture_split_snapshots 1
            reconcile_split_snapshots 2>"$SPLIT_RECONCILE_LOG" || true
            report_split_readiness_failure "$context" \
                "worker exited unexpectedly before split readiness"
            return 1
        fi

        now="$(date +%s)"
        remaining=$((deadline - now))
        if [[ "$remaining" -le 0 ]]; then
            report_split_readiness_failure "$context" \
                "timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for reconciled real split topology"
            return 1
        fi
        request_timeout="$SNAPSHOT_REQUEST_TIMEOUT_SECONDS"
        if [[ "$remaining" -lt "$request_timeout" ]]; then
            request_timeout="$remaining"
        fi
        capture_split_snapshots "$request_timeout"
        reconciliation_ready=0
        if READY_SUMMARY="$(reconcile_split_snapshots 2>"$SPLIT_RECONCILE_LOG")"; then
            reconciliation_ready=1
        fi
        now="$(date +%s)"
        if [[ "$now" -ge "$deadline" ]]; then
            report_split_readiness_failure "$context" \
                "timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for reconciled real split topology"
            return 1
        fi
        if [[ "$reconciliation_ready" -eq 1 ]]; then
            DRIVER_LABEL="$("${automation[@]}" automation split-probe driver "$SPLIT_EVIDENCE_PATH")"
            case "$DRIVER_LABEL" in
                seed) DRIVER_API_PORT="$SEED_API_PORT" ;;
                worker) DRIVER_API_PORT="$WORKER_API_PORT" ;;
                *)
                    echo "split evidence selected unknown stage-0 driver: $DRIVER_LABEL" >&2
                    return 1
                    ;;
            esac
            echo "Split topology ready after $((now - started_at))s (${MODEL_LABEL}): ${READY_SUMMARY}"
            echo "Selected ${DRIVER_LABEL} as stage-0 OpenAI driver"
            return 0
        fi
        sleep 1
    done
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
    if [[ "$ALLOW_UNCERTIFIED_SPLIT" == "1" ]]; then
        args+=(--allow-uncertified-split)
    fi
    if [[ -n "$CTX_SIZE" ]]; then
        args+=(--ctx-size "$CTX_SIZE")
    fi
    if [[ "$DURABLE_L3" == "1" ]]; then
        args+=(
            --kv-cache-disk 2GiB
            --kv-cache-disk-dir "${DURABLE_L3_ROOT}/${label}"
            --kv-cache-min-free 1GiB
        )
    fi

    local -a node_env=(
        env
        "HOME=$home"
        "MESH_LLM_RUNTIME_ROOT=$runtime"
        "SKIPPY_TELEMETRY_STDERR=1"
    )
    if [[ "$DURABLE_L3" != "1" ]]; then
        node_env+=("MESH_LLM_EPHEMERAL_KEY=1")
    fi
    "${node_env[@]}" "$MESH_LLM" "${args[@]}" >"$log_file" 2>&1 &
    printf '%s\n' "$!"
}

run_client_routing_probe() {
    [[ "$CLIENT_ROUTING" == "1" ]] || return 0
    echo "Starting passive client against dense split topology"
    local client_home="${PROCESS_ROOT}/client/h"
    local client_runtime="${PROCESS_ROOT}/client/r"
    mkdir -p "$client_home" "$client_runtime"
    HOME="$client_home" \
        MESH_LLM_RUNTIME_ROOT="$client_runtime" \
        MESH_LLM_EPHEMERAL_KEY=1 \
        "$MESH_LLM" --log-format json client --join "$TOKEN" \
        --port "$CLIENT_API_PORT" --console "$CLIENT_CONSOLE_PORT" --headless \
        >"$CLIENT_LOG" 2>&1 &
    CLIENT_PID=$!

    local started_at
    local deadline
    local now
    local remaining
    local request_timeout
    started_at="$(date +%s)"
    deadline=$((started_at + READINESS_TIMEOUT_SECONDS))
    while :; do
        if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
            echo "passive client exited unexpectedly" >&2
            tail -160 "$CLIENT_LOG" >&2 || true
            exit 1
        fi
        now="$(date +%s)"
        remaining=$((deadline - now))
        if [[ "$remaining" -le 0 ]]; then
            echo "timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for passive client model routing" >&2
            tail -160 "$CLIENT_LOG" >&2 || true
            exit 1
        fi
        request_timeout="$SNAPSHOT_REQUEST_TIMEOUT_SECONDS"
        if [[ "$remaining" -lt "$request_timeout" ]]; then
            request_timeout="$remaining"
        fi
        client_models="$(curl -fsS --connect-timeout "$request_timeout" \
            --max-time "$request_timeout" \
            "http://127.0.0.1:${CLIENT_API_PORT}/v1/models" 2>/dev/null || true)"
        now="$(date +%s)"
        if [[ "$now" -ge "$deadline" ]]; then
            echo "timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for passive client model routing" >&2
            tail -160 "$CLIENT_LOG" >&2 || true
            exit 1
        fi
        if printf '%s' "$client_models" | "${automation[@]}" automation smoke-observation has-model "$MODEL_ID" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    local probe_root="${WORK_DIR}/client-routing"
    mkdir -p "$probe_root"
    "${automation[@]}" automation split-probe client-payloads "$MODEL_ID" "$probe_root/request.json" "$probe_root/stream.json"
    curl -fsS --max-time 120 "http://127.0.0.1:${CLIENT_API_PORT}/v1/chat/completions" \
        -H 'content-type: application/json' -d @"$probe_root/request.json" \
        -o "$probe_root/response.json"
    "${automation[@]}" automation smoke-observation chat <"$probe_root/response.json"
    curl -fsS --max-time 120 -N "http://127.0.0.1:${CLIENT_API_PORT}/v1/chat/completions" \
        -H 'content-type: application/json' -d @"$probe_root/stream.json" \
        -o "$probe_root/stream.txt"
    grep -q 'data: \[DONE\]' "$probe_root/stream.txt"
    echo "Passive client routing and streaming validated against dense split topology"
}

SEED_PID="$(start_node seed "" "$SEED_API_PORT" "$SEED_CONSOLE_PORT" "$SEED_BIND_PORT" "$SEED_LOG")"

wait_for_seed_token ""

WORKER_PID="$(start_node worker "$TOKEN" "$WORKER_API_PORT" "$WORKER_CONSOLE_PORT" "$WORKER_BIND_PORT" "$WORKER_LOG")"

DRIVER_LABEL=""
DRIVER_API_PORT=""
wait_for_split_topology ""

if [[ -z "$DRIVER_API_PORT" ]]; then
    echo "no split driver API port was selected" >&2
    exit 1
fi
MODEL_ID="$(
    "${automation[@]}" automation split-probe model "$SPLIT_EVIDENCE_PATH"
)"
if [[ -z "$MODEL_ID" ]]; then
    echo "${DRIVER_LABEL:-selected driver} split evidence did not return a model id" >&2
    exit 1
fi
export MODEL_ID MODEL_LABEL

# Stage readiness is published before the serving target finishes registering
# on every observer. Give routing the same bounded settle interval used between
# inference requests so the first probe cannot race that final handoff.
sleep "$REQUEST_SETTLE_SECONDS"

run_client_routing_probe

PREFIX_PAYLOAD_ROOT="${WORK_DIR}/prefix-payloads"
PREFIX_RESPONSE_ROOT="${WORK_DIR}/prefix-responses"

# Exit code the prefix validator uses for the one failure this smoke cannot
# distinguish from a regression on a single pass: every request restored
# nothing, which is what an unreleased stage lane looks like from the outside.
# Anything else is a real assertion failure and is never retried.
PREFIX_TRANSIENT_STATUS=75

write_prefix_payloads() {
    "${automation[@]}" automation split-probe prefix-payloads "$MODEL_ID" "$1" "$2"
}

# Requests 1..6: odd indexes are first sights of each prompt length (growth
# arms), even indexes are identical re-sends of the previous prompt (repeat
# arms). Requests 1-2 share prompt X, 3-4 share X+E1, 5-6 share X+E1+E2.
PREFIX_REQUEST_COUNT=6

validate_prefix_responses() {
    "${automation[@]}" automation split-probe prefix-verify "$1" "$PREFIX_REQUEST_COUNT" "$EXPECTED_EXACT_PAYLOAD_KIND"
}

assert_expected_stage_payload() {
    [[ -n "$EXPECTED_EXACT_PAYLOAD_KIND" ]] || return 0
    "${automation[@]}" automation split-probe payload-kind "$EXPECTED_EXACT_PAYLOAD_KIND" "$SEED_LOG" "$WORKER_LOG"
}

capture_kv_cache_statuses() {
    local prefix="$1"
    "$MESH_LLM" kv-cache status --port "$SEED_CONSOLE_PORT" --json \
        >"${prefix}-seed.json"
    "$MESH_LLM" kv-cache status --port "$WORKER_CONSOLE_PORT" --json \
        >"${prefix}-worker.json"
}

durable_population_ready() {
    "${automation[@]}" automation split-probe durable-ready "$1"
}

wait_for_durable_population() {
    local prefix="$1"
    local deadline=$(( $(date +%s) + READINESS_TIMEOUT_SECONDS ))
    while [[ "$(date +%s)" -lt "$deadline" ]]; do
        if capture_kv_cache_statuses "$prefix" 2>/dev/null && \
            durable_population_ready "$prefix" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    echo "durable L3 did not publish restorable state before restart" >&2
    capture_kv_cache_statuses "$prefix" 2>/dev/null || true
    return 1
}

record_durable_restart() {
    local evidence_dir="$1"
    local artifact_id model_sha256
    case "$MODEL_LABEL" in
        dense)
            artifact_id="$DENSE_ARTIFACT_ID"
            model_sha256="$DENSE_MODEL_SHA256"
            ;;
        recurrent)
            artifact_id="$RECURRENT_ARTIFACT_ID"
            model_sha256="$RECURRENT_MODEL_SHA256"
            ;;
        *)
            echo "unsupported durable L3 model label: $MODEL_LABEL" >&2
            return 1
            ;;
    esac
    "${automation[@]}" automation split-probe durable-record "$DURABLE_L3_RECORDS" "$MODEL_LABEL" "$MODEL" "$artifact_id" \
        "$model_sha256" "$EXPECTED_EXACT_PAYLOAD_KIND" \
        "${DURABLE_L3_ROOT}/seed" "${DURABLE_L3_ROOT}/worker" \
        "$evidence_dir"
}

run_durable_restart_probe() {
    [[ "$DURABLE_L3" == "1" ]] || return 0
    local request_path="$1"
    local warm_response_path="$2"
    local evidence_dir="${WORK_DIR}/durable-${MODEL_LABEL}"
    mkdir -p "$evidence_dir"
    cp "$request_path" "$evidence_dir/request.json"
    cp "$warm_response_path" "$evidence_dir/warm-response.json"
    wait_for_durable_population "$evidence_dir/before"

    kill_tree "$CLIENT_PID"
    CLIENT_PID=""
    kill_tree "$WORKER_PID"
    WORKER_PID=""
    kill_tree "$SEED_PID"
    SEED_PID=""

    SEED_LOG="${WORK_DIR}/${MODEL_LABEL}-restart-seed.log"
    WORKER_LOG="${WORK_DIR}/${MODEL_LABEL}-restart-worker.log"
    SEED_PID="$(start_node seed "" "$SEED_API_PORT" "$SEED_CONSOLE_PORT" "$SEED_BIND_PORT" "$SEED_LOG")"
    wait_for_seed_token "${MODEL_LABEL} durable restart: "
    WORKER_PID="$(start_node worker "$TOKEN" "$WORKER_API_PORT" "$WORKER_CONSOLE_PORT" "$WORKER_BIND_PORT" "$WORKER_LOG")"
    DRIVER_LABEL=""
    DRIVER_API_PORT=""
    wait_for_split_topology "${MODEL_LABEL} durable restart: "
    MODEL_ID="$("${automation[@]}" automation split-probe model "$SPLIT_EVIDENCE_PATH")"
    [[ -n "$MODEL_ID" ]] || {
        echo "durable restart split evidence did not return a model id" >&2
        return 1
    }
    sleep "$REQUEST_SETTLE_SECONDS"
    capture_kv_cache_statuses "$evidence_dir/restart-before"
    "${automation[@]}" automation split-probe rewrite-model "$MODEL_ID" "$evidence_dir/request.json"
    curl -fsS --max-time 180 \
        "http://127.0.0.1:${DRIVER_API_PORT}/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d @"$evidence_dir/request.json" \
        -o "$evidence_dir/restored-response.json"
    sleep "$REQUEST_SETTLE_SECONDS"
    capture_kv_cache_statuses "$evidence_dir/after"
    "$MESH_LLM" kv-cache clear --port "$SEED_CONSOLE_PORT" --yes --json \
        >"$evidence_dir/clear-seed.json"
    "$MESH_LLM" kv-cache clear --port "$WORKER_CONSOLE_PORT" --yes --json \
        >"$evidence_dir/clear-worker.json"
    capture_kv_cache_statuses "$evidence_dir/cleared"
    record_durable_restart "$evidence_dir"
}

write_durable_l3_evidence() {
    [[ "$DURABLE_L3" == "1" ]] || return 0
    "${automation[@]}" automation split-probe durable-evidence "$DURABLE_L3_RECORDS" "$DURABLE_L3_EVIDENCE_PATH" \
        "$([[ -n "$RECURRENT_MODEL" ]] && printf 'dense,recurrent' || printf '%s' "$PRIMARY_MODEL_LABEL")"
}

prefix_validated=0
for attempt in $(seq 1 "$PREFIX_ATTEMPTS"); do
    payload_dir="${PREFIX_PAYLOAD_ROOT}/attempt-${attempt}"
    response_dir="${PREFIX_RESPONSE_ROOT}/attempt-${attempt}"
    mkdir -p "$payload_dir" "$response_dir"
    write_prefix_payloads "$payload_dir" "attempt-${attempt}"

    for index in $(seq 1 "$PREFIX_REQUEST_COUNT"); do
        response_path="${response_dir}/response-${index}.json"
        if ! response_status="$(curl -sS --max-time 180 \
            "http://127.0.0.1:${DRIVER_API_PORT}/v1/chat/completions" \
            -H 'content-type: application/json' \
            -d @"${payload_dir}/prompt-${index}.json" \
            -o "$response_path" \
            -w '%{http_code}')"; then
            echo "split inference request ${index} failed through ${DRIVER_LABEL} stage-0 driver" >&2
            cat "$response_path" >&2 2>/dev/null || true
            exit 1
        fi
        if [[ ! "$response_status" =~ ^2[0-9][0-9]$ ]]; then
            echo "split inference request ${index} failed through ${DRIVER_LABEL} stage-0 driver" >&2
            cat "$response_path" >&2 2>/dev/null || true
            exit 1
        fi
        # The host returns the OpenAI response before the stage connection has
        # released its single CI lane. Give graceful Stop enough time to finish
        # so the next request tests cache reuse rather than transient admission.
        sleep "$REQUEST_SETTLE_SECONDS"
    done

    set +e
    validate_prefix_responses "$response_dir"
    prefix_status=$?
    set -e
    if [[ "$prefix_status" -eq 0 ]]; then
        prefix_validated=1
        break
    fi
    if [[ "$prefix_status" -ne "$PREFIX_TRANSIENT_STATUS" ]]; then
        exit "$prefix_status"
    fi
    echo "prefix attempt ${attempt} of ${PREFIX_ATTEMPTS} saw no reuse; retrying from a cold prefix" >&2
done

if [[ "$prefix_validated" -ne 1 ]]; then
    echo "split prefix reuse never materialized across ${PREFIX_ATTEMPTS} attempts" >&2
    exit 1
fi

assert_expected_stage_payload

run_durable_restart_probe "${payload_dir}/prompt-${PREFIX_REQUEST_COUNT}.json" \
    "${response_dir}/response-${PREFIX_REQUEST_COUNT}.json"

echo "Two-node split smoke passed for model leg: ${MODEL_LABEL:-default}"

# Optional recurrent leg: rerun the identical flow against a second model in
# the same process pair so one CI job proves both cache families. The phase
# function restores MODEL/CTX_SIZE, restarts both nodes from scratch (fresh
# token, fresh stage split), and reruns the prefix cache assertions.
run_recurrent_leg() {
    echo "=== Two-node split smoke: recurrent leg ==="
    kill_tree "$CLIENT_PID"
    CLIENT_PID=""
    kill_tree "$WORKER_PID"
    kill_tree "$SEED_PID"
    SEED_LOG="${WORK_DIR}/recurrent-seed.log"
    WORKER_LOG="${WORK_DIR}/recurrent-worker.log"

    MODEL="$RECURRENT_MODEL"
    CTX_SIZE="$RECURRENT_CTX_SIZE"
    MODEL_LABEL="recurrent"
    EXPECTED_EXACT_PAYLOAD_KIND="$RECURRENT_EXPECTED_EXACT_PAYLOAD_KIND"
    export MODEL CTX_SIZE

    SEED_PID="$(start_node seed "" "$SEED_API_PORT" "$SEED_CONSOLE_PORT" "$SEED_BIND_PORT" "$SEED_LOG")"

    wait_for_seed_token "recurrent leg: "

    WORKER_PID="$(start_node worker "$TOKEN" "$WORKER_API_PORT" "$WORKER_CONSOLE_PORT" "$WORKER_BIND_PORT" "$WORKER_LOG")"

    DRIVER_LABEL=""
    DRIVER_API_PORT=""
    wait_for_split_topology "recurrent leg: "

    if [[ -z "$DRIVER_API_PORT" ]]; then
        echo "recurrent leg: no split driver API port was selected" >&2
        exit 1
    fi
    MODEL_ID="$(
        "${automation[@]}" automation split-probe model "$SPLIT_EVIDENCE_PATH"
    )"
    if [[ -z "$MODEL_ID" ]]; then
        echo "${DRIVER_LABEL:-selected driver} split evidence did not return a model id (recurrent leg)" >&2
        exit 1
    fi
}

if [[ -n "$RECURRENT_MODEL" ]]; then
    run_recurrent_leg

    PREFIX_PAYLOAD_ROOT="${WORK_DIR}/prefix-payloads-recurrent"
    PREFIX_RESPONSE_ROOT="${WORK_DIR}/prefix-responses-recurrent"

    prefix_validated=0
    for attempt in $(seq 1 "$PREFIX_ATTEMPTS"); do
        payload_dir="${PREFIX_PAYLOAD_ROOT}/attempt-${attempt}"
        response_dir="${PREFIX_RESPONSE_ROOT}/attempt-${attempt}"
        mkdir -p "$payload_dir" "$response_dir"
        write_prefix_payloads "$payload_dir" "recurrent-attempt-${attempt}"

        for index in $(seq 1 "$PREFIX_REQUEST_COUNT"); do
            curl -fsS --max-time 180 \
                "http://127.0.0.1:${DRIVER_API_PORT}/v1/chat/completions" \
                -H 'content-type: application/json' \
                -d @"${payload_dir}/prompt-${index}.json" \
                -o "${response_dir}/response-${index}.json"
            sleep "$REQUEST_SETTLE_SECONDS"
        done

        set +e
        validate_prefix_responses "$response_dir"
        prefix_status=$?
        set -e
        if [[ "$prefix_status" -eq 0 ]]; then
            prefix_validated=1
            break
        fi
        if [[ "$prefix_status" -ne "$PREFIX_TRANSIENT_STATUS" ]]; then
            exit "$prefix_status"
        fi
        echo "recurrent leg: prefix attempt ${attempt} of ${PREFIX_ATTEMPTS} saw no reuse; retrying from a cold prefix" >&2
    done

    if [[ "$prefix_validated" -ne 1 ]]; then
        echo "recurrent leg: split prefix reuse never materialized across ${PREFIX_ATTEMPTS} attempts" >&2
        exit 1
    fi

    assert_expected_stage_payload

    run_durable_restart_probe "${payload_dir}/prompt-${PREFIX_REQUEST_COUNT}.json" \
        "${response_dir}/response-${PREFIX_REQUEST_COUNT}.json"

    echo "Two-node split smoke passed for model leg: recurrent"
fi

write_durable_l3_evidence
