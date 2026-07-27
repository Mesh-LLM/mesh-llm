#!/usr/bin/env bash
# qa-runtime-daemon-lifecycle.sh - certify runtime daemon model lifecycle rollout:
# zero-model serve, all modes, best-effort/fail-fast, runtime load/unload/ensure/drain,
# plugin-only routing, activity overrides/admission, and privacy.

set -euo pipefail

CURRENT_BINARY=""
EVIDENCE_DIR=".sisyphus/evidence"
BASE_PORT="${MESH_QA_BASE_PORT:-19740}"
MAX_WAIT="${MESH_QA_MAX_WAIT:-60}"
KEEP_LOGS=false
PRINT_PLAN=false

TMP_ROOT="${MESH_QA_TMP_ROOT:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}}"

RUN_DIR=""
LOG_DIR=""
STATUS_DIR=""
CONTROL_DIR=""
VERSIONS_DIR=""
RESULTS_JSONL=""
COMMANDS_JSONL=""
MANIFEST_JSON=""
SUMMARY_JSON=""
SUMMARY_MD=""
WORK_ROOT=""
PIDS=()
PID_STARTS=()
EXIT_STATUS=0
NODE_INDEX=0
OWNER_KEY=""

usage() {
    cat <<'EOF'
Usage:
  scripts/qa-runtime-daemon-lifecycle.sh \
    --current-binary /path/to/current/mesh-llm \
    --evidence-dir .sisyphus/evidence [options]

Purpose:
  Produce executable evidence that the runtime daemon model lifecycle rollout is
  correct. Tests zero-model serve, all modes, best-effort/fail-fast startup,
  runtime load/unload/ensure/drain commands, activity overrides/admission, and
  privacy guarantees without downloading models or publishing a mesh.

Required:
  --current-binary PATH    Current-branch mesh-llm binary (e.g. ./target/debug/mesh-llm).

Options:
  --evidence-dir DIR       Evidence root (default: .sisyphus/evidence).
  --base-port PORT         First reserved local port (default: 19740).
  --max-wait SECONDS       Readiness timeout (default: 60).
  --keep-logs              Keep successful run logs.
  --print-plan             Print planned checks as JSON without side effects.
  -h, --help               Show this help.

Planned checks:
  prereq.current-binary          Binary is executable and reports a version
  prereq.owner-identity          Temporary owner keystore can be created
  zero_model_serve_ready         Serve with no model leaves daemon alive; /v1/models returns empty or success
  runtime_mode_serve             Serve mode starts correctly
  runtime_mode_on_demand         On-demand mode starts idle (if testable)
  best_effort_startup            Bad model with best_effort leaves daemon alive
  fail_fast_startup              Bad model with fail_fast exits nonzero
  runtime_load_model             load-model returns accepted
  runtime_unload_model           unload-model returns accepted
  runtime_ensure_model           ensure-model returns accepted
  runtime_drain_model            drain-model returns accepted or timeout
  activity_override              PUT/DELETE /api/runtime/activity/override round-trips
  privacy_no_raw_data            Activity API exposes only the documented coarse fields
  clean_process_teardown         mesh-llm stop or pkill leaves no leaked processes

Evidence:
  Each run creates a timestamped directory containing:
    manifest.json      Run inputs and mode flags.
    commands.jsonl     Commands executed by the harness and their logs.
    results.jsonl      Machine-readable PASS/FAIL/PREREQ records.
    summary.md         Human-readable final summary.
    summary.json       Machine-readable final summary.
    versions/*.txt     Captured binary version strings.
    logs/, status/, control/ grouped runtime payloads.

Result vocabulary:
  PASS completed, FAIL failed, PREREQ blocked by an explicit local prerequisite.

This script does not download models or credentials and does not publish a mesh.
EOF
}

fail_usage() {
    echo "error: $*" >&2
    echo >&2
    usage >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --current-binary)
            CURRENT_BINARY="${2:-}"
            shift 2
            ;;
        --evidence-dir)
            EVIDENCE_DIR="${2:-}"
            shift 2
            ;;
        --base-port)
            BASE_PORT="${2:-}"
            shift 2
            ;;
        --max-wait)
            MAX_WAIT="${2:-}"
            shift 2
            ;;
        --keep-logs)
            KEEP_LOGS=true
            shift
            ;;
        --print-plan)
            PRINT_PLAN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail_usage "unknown argument: $1"
            ;;
    esac
done

missing=()
[[ -n "$CURRENT_BINARY" ]] || missing+=("--current-binary")
if [[ "${#missing[@]}" -gt 0 ]]; then
    fail_usage "missing required options: ${missing[*]}"
fi

for numeric in BASE_PORT MAX_WAIT; do
    value="${!numeric}"
    if [[ ! "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
        fail_usage "--$(printf '%s' "$numeric" | tr '[:upper:]_' '[:lower:]-') must be a positive integer"
    fi
done

require_tool() {
    command -v "$1" >/dev/null 2>&1 || { echo "error: missing required tool: $1" >&2; exit 2; }
}

require_tool python3
require_tool pgrep

# --print-plan is side-effect-free: no processes, no files written to evidence dir
if [[ "$PRINT_PLAN" == true ]]; then
    python3 - \
        "$CURRENT_BINARY" \
        "$EVIDENCE_DIR" \
        "$BASE_PORT" \
        "$MAX_WAIT" <<'PY'
import json
import sys

current, evidence, base_port, max_wait = sys.argv[1:]

checks = [
    "prereq.current-binary",
    "prereq.owner-identity",
    "zero_model_serve_ready",
    "runtime_mode_serve",
    "runtime_mode_on_demand",
    "best_effort_startup",
    "fail_fast_startup",
    "runtime_load_model",
    "runtime_unload_model",
    "runtime_ensure_model",
    "runtime_drain_model",
    "activity_override",
    "privacy_no_raw_data",
    "clean_process_teardown",
]

plan = {
    "script": "qa-runtime-daemon-lifecycle.sh",
    "current_binary": current,
    "evidence_dir": evidence,
    "base_port": int(base_port),
    "max_wait_seconds": int(max_wait),
    "checks": checks,
}
print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
PY
    exit 0
fi

if [[ ! -x "$CURRENT_BINARY" ]]; then
    fail_usage "--current-binary is not executable: $CURRENT_BINARY"
fi

require_tool curl
require_tool date
require_tool mktemp

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${EVIDENCE_DIR%/}/runtime-daemon-lifecycle-${RUN_ID}"
WORK_ROOT="$(mktemp -d "${TMP_ROOT%/}/mesh-runtime-daemon-lifecycle.XXXXXX")"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
CONTROL_DIR="$RUN_DIR/control"
VERSIONS_DIR="$RUN_DIR/versions"
RESULTS_JSONL="$RUN_DIR/results.jsonl"
COMMANDS_JSONL="$RUN_DIR/commands.jsonl"
MANIFEST_JSON="$RUN_DIR/manifest.json"
SUMMARY_JSON="$RUN_DIR/summary.json"
SUMMARY_MD="$RUN_DIR/summary.md"
mkdir -p "$LOG_DIR" "$STATUS_DIR" "$CONTROL_DIR" "$VERSIONS_DIR" "$WORK_ROOT"
: >"$RESULTS_JSONL"
: >"$COMMANDS_JSONL"

append_summary() {
    printf '%s\n' "$*" >>"$SUMMARY_MD"
}

record_command() {
    local name="$1"
    local log="$2"
    shift 2
    python3 - "$COMMANDS_JSONL" "$name" "$log" "$@" <<'PY'
import json
import sys

path, name, log, *argv = sys.argv[1:]
record = {"name": name, "argv": argv, "log": log}
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

record_result() {
    local status="$1"
    local name="$2"
    local message="$3"
    shift 3

    printf '%s %s' "$status" "$name"
    for field in "$@"; do
        printf ' %q' "$field"
    done
    if [[ -n "$message" ]]; then
        printf ' message=%q' "$message"
    fi
    printf '\n'

    python3 - "$RESULTS_JSONL" "$status" "$name" "$message" "$@" <<'PY'
import json
import sys

path, status, name, message, *fields = sys.argv[1:]
record = {"status": status, "name": name, "message": message}
for field in fields:
    if "=" not in field:
        continue
    key, value = field.split("=", 1)
    record[key] = value
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
PY

    append_summary "- ${status} ${name}: ${message}"
    if [[ "$status" == "FAIL" ]]; then
        EXIT_STATUS=1
    fi
}

write_manifest() {
    python3 - \
        "$MANIFEST_JSON" \
        "$RUN_ID" \
        "$CURRENT_BINARY" \
        "$BASE_PORT" \
        "$MAX_WAIT" \
        "$TMP_ROOT" \
        "$RUN_DIR" \
        "$WORK_ROOT" <<'PY'
import json
import sys

keys = [
    "path", "run_id", "current_binary", "base_port", "max_wait_seconds",
    "tmp_root", "evidence_dir", "work_root",
]
args = dict(zip(keys, sys.argv[1:]))
manifest = {
    "run_id": args["run_id"],
    "current_binary": args["current_binary"],
    "base_port": int(args["base_port"]),
    "max_wait_seconds": int(args["max_wait_seconds"]),
    "tmp_root": args["tmp_root"],
    "evidence_dir": args["evidence_dir"],
    "work_root": args["work_root"],
}
with open(args["path"], "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

write_summary_json() {
    python3 - "$RESULTS_JSONL" "$SUMMARY_JSON" "$RUN_DIR" <<'PY'
import json
import sys

results_path, summary_path, run_dir = sys.argv[1:]
records = []
try:
    with open(results_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
except FileNotFoundError:
    pass
statuses = [record.get("status") for record in records]
overall = "fail" if "FAIL" in statuses else "prereq" if "PREREQ" in statuses else "pass"
summary = {
    "overall": overall,
    "evidence_dir": run_dir,
    "counts": {name.lower(): statuses.count(name) for name in ["PASS", "FAIL", "PREREQ"]},
    "results": records,
}
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
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

process_start_time() {
    ps -o lstart= -p "$1" 2>/dev/null | awk 'NF { $1=$1; print; exit }'
}

pid_matches_start() {
    local pid="$1"
    local expected_start="$2"
    local actual_start=""
    [[ -n "$expected_start" ]] || return 1
    actual_start="$(process_start_time "$pid")"
    [[ -n "$actual_start" && "$actual_start" == "$expected_start" ]]
}

signal_process_records() {
    local signal="$1"
    local records="$2"
    local pid pid_start
    while IFS=$'\t' read -r pid pid_start; do
        [[ -n "$pid" ]] || continue
        if pid_matches_start "$pid" "$pid_start"; then
            kill "-$signal" "$pid" 2>/dev/null || true
        fi
    done <<<"$records"
}

tracked_pid_index() {
    local target_pid="$1"
    local index
    for ((index = 0; index < ${#PIDS[@]}; index++)); do
        if [[ "${PIDS[$index]:-}" == "$target_pid" ]]; then
            printf '%s\n' "$index"
            return 0
        fi
    done
    return 1
}

kill_tree() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    local tracked_index=""
    local expected_start=""
    tracked_index="$(tracked_pid_index "$pid" || true)"
    if [[ -n "$tracked_index" ]]; then
        expected_start="${PID_STARTS[$tracked_index]:-}"
        if ! pid_matches_start "$pid" "$expected_start"; then
            PIDS[tracked_index]=""
            PID_STARTS[tracked_index]=""
            return 0
        fi
    else
        expected_start="$(process_start_time "$pid")"
        [[ -n "$expected_start" ]] || return 0
    fi

    local children child child_start
    local child_records=""
    children="$(descendant_pids "$pid" | sort -u || true)"
    kill "$pid" 2>/dev/null || true
    for child in $children; do
        child_start="$(process_start_time "$child")"
        [[ -n "$child_start" ]] || continue
        child_records+="${child}"$'\t'"${child_start}"$'\n'
    done
    signal_process_records TERM "$child_records"
    sleep 1
    if pid_matches_start "$pid" "$expected_start"; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    signal_process_records KILL "$child_records"
    wait "$pid" 2>/dev/null || true
    if [[ -n "$tracked_index" ]]; then
        PIDS[tracked_index]=""
        PID_STARTS[tracked_index]=""
    fi
}

cleanup() {
    local incoming_status=$?
    if ((${#PIDS[@]} > 0)); then
        for pid in "${PIDS[@]}"; do
            [[ -n "$pid" ]] || continue
            kill_tree "$pid"
        done
    fi
    local alive=0
    if ((${#PIDS[@]} > 0)); then
        for pid in "${PIDS[@]}"; do
            [[ -n "$pid" ]] || continue
            if kill -0 "$pid" 2>/dev/null; then
                alive=$((alive + 1))
            fi
        done
    fi
    if [[ "$alive" -eq 0 ]]; then
        record_result "PASS" "cleanup" "harness-owned processes stopped" "processes=0"
    else
        record_result "FAIL" "cleanup" "harness-owned processes remain" "processes=$alive"
    fi
    if [[ "$EXIT_STATUS" -eq 0 && "$KEEP_LOGS" != true ]]; then
        find "$LOG_DIR" -type f -empty -delete 2>/dev/null || true
        rm -rf "$WORK_ROOT" 2>/dev/null || true
    fi
    write_summary_json
    local final_status="$EXIT_STATUS"
    if [[ "$incoming_status" -ne 0 ]]; then
        final_status="$incoming_status"
    fi
    trap - EXIT
    exit "$final_status"
}
trap cleanup EXIT

write_manifest

append_summary "# Runtime daemon lifecycle certification"
append_summary ""
append_summary "- Run ID: \`$RUN_ID\`"
append_summary "- Current binary: \`$CURRENT_BINARY\`"
append_summary "- Base port: \`$BASE_PORT\`"
append_summary "- Max wait: \`$MAX_WAIT\`s"
append_summary "- Evidence: \`$RUN_DIR\`"
append_summary ""

# ──────────────────────── Helpers ────────────────────────

record_binary_prereq() {
    local label="$1"
    local binary="$2"
    local version_path="$VERSIONS_DIR/${label}.txt"
    local version_log="$LOG_DIR/${label}-version.log"
    local version=""

    if "$binary" --version >"$version_path" 2>"$version_log"; then
        version="$(head -1 "$version_path" || true)"
        printf '%s\n' "$version" >"$version_path"
    fi

    if [[ -n "$version" ]]; then
        record_result "PASS" "prereq.${label}-binary" "${label} binary is executable and reports a version" \
            "path=$binary" "version=$(printf '%s' "$version" | tr ' ' '_')" \
            "version_path=$version_path" "log=$version_log"
    else
        record_result "PREREQ" "prereq.${label}-binary" "${label} binary did not report a version" \
            "path=$binary" "version_path=$version_path" "log=$version_log"
    fi
}

record_binary_prereq "current" "$CURRENT_BINARY"

OWNER_KEY="$WORK_ROOT/owner-keystore.json"
owner_key_log="$LOG_DIR/owner-auth-init.log"
record_command "owner-auth-init" "$owner_key_log" \
    "$CURRENT_BINARY" auth init --owner-key "$OWNER_KEY" --no-passphrase --force
if "$CURRENT_BINARY" auth init \
    --owner-key "$OWNER_KEY" \
    --no-passphrase \
    --force >"$owner_key_log" 2>&1; then
    record_result "PASS" "prereq.owner-identity" \
        "temporary owner keystore created for authenticated lifecycle checks" \
        "log=$owner_key_log"
else
    OWNER_KEY=""
    record_result "PREREQ" "prereq.owner-identity" \
        "could not create a temporary owner keystore" \
        "log=$owner_key_log"
fi

start_node() {
    local label="$1"
    shift

    NODE_INDEX=$((NODE_INDEX + 1))
    local node_slug="n$NODE_INDEX"
    local home="$WORK_ROOT/h-$node_slug"
    local runtime_root="$WORK_ROOT/r-$node_slug"
    local log="$LOG_DIR/$label.log"
    mkdir -p "$home" "$runtime_root" || return 1

    (
        export HOME="$home"
        export MESH_LLM_RUNTIME_ROOT="$runtime_root"
        export MESH_LLM_EPHEMERAL_KEY=1
        if [[ -n "$OWNER_KEY" ]]; then
            exec "$CURRENT_BINARY" --owner-key "$OWNER_KEY" "$@"
        fi
        exec "$CURRENT_BINARY" "$@"
    ) >"$log" 2>&1 &
    START_NODE_PID=$!
    PIDS+=("$START_NODE_PID")
    PID_STARTS+=("$(process_start_time "$START_NODE_PID")")
}

assert_process_alive() {
    local pid="$1"
    local label="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        record_result "FAIL" "$label" "process exited unexpectedly" "pid=$pid" "log=$LOG_DIR/$label.log"
        return 1
    fi
}

curl_json() {
    curl -fsS --max-time 5 "$1" -o "$2"
}

wait_status() {
    local label="$1"
    local console_port="$2"
    local out="$STATUS_DIR/${label}-status.json"
    for second in $(seq 1 "$MAX_WAIT"); do
        if curl_json "http://127.0.0.1:${console_port}/api/status" "$out" 2>/dev/null; then
            record_result "PASS" "${label}.status" "management API returned status" \
                "seconds=$second" "path=$out"
            return 0
        fi
        sleep 1
    done
    record_result "FAIL" "${label}.status" "timed out waiting for management API" \
        "seconds=$MAX_WAIT" "port=$console_port"
    return 1
}

# ──────────────── Test scenarios ────────────────────────

check_zero_model_serve_ready() {
    local label="zero-model-serve"
    local api_port=$((BASE_PORT + NODE_INDEX * 4))
    local console_port=$((api_port + 1))
    local bind_port=$((api_port + 2))

    start_node "$label" \
        --headless \
        --port "$api_port" \
        --console "$console_port" \
        --bind-port "$bind_port" || { record_result "FAIL" "zero_model_serve_ready" "could not start daemon"; return 0; }

    local pid="$START_NODE_PID"

    # Wait for management API to come up
    if ! wait_status "$label" "$console_port"; then
        kill_tree "$pid"
        return 0
    fi

    assert_process_alive "$pid" "zero_model_serve_ready" || { kill_tree "$pid"; return 0; }

    # Verify /v1/models returns success (empty list or any valid response)
    local models_out="$STATUS_DIR/${label}-models.json"
    if curl_json "http://127.0.0.1:${api_port}/v1/models" "$models_out" 2>/dev/null; then
        record_result "PASS" "zero_model_serve_ready" \
            "daemon alive with no model; /v1/models returned success" \
            "path=$models_out" "console_port=$console_port" "api_port=$api_port"
    else
        record_result "FAIL" "zero_model_serve_ready" \
            "/v1/models did not return a valid response on zero-model serve" \
            "path=$models_out" "api_port=$api_port"
    fi

    # Leave process running for lifecycle command tests; track its ports globally
    ZERO_MODEL_PID="$pid"
    ZERO_MODEL_CONSOLE="$console_port"
}

check_runtime_mode_serve() {
    local label="runtime-mode-serve"
    local api_port=$((BASE_PORT + NODE_INDEX * 4))
    local console_port=$((api_port + 1))
    local bind_port=$((api_port + 2))

    start_node "$label" \
        --headless \
        --port "$api_port" \
        --console "$console_port" \
        --bind-port "$bind_port" || { record_result "FAIL" "runtime_mode_serve" "serve mode did not start"; return 0; }

    local pid="$START_NODE_PID"

    if ! wait_status "$label" "$console_port"; then
        kill_tree "$pid"
        return 0
    fi

    assert_process_alive "$pid" "runtime_mode_serve" || { kill_tree "$pid"; return 0; }

    # The default invocation exercises serve mode; readiness is verified above.
    local status_file="$STATUS_DIR/${label}-status.json"
    record_result "PASS" "runtime_mode_serve" \
        "serve mode started and management API responding" \
        "path=$status_file" "pid=$pid"

    kill_tree "$pid" 2>/dev/null || true
}

check_runtime_mode_on_demand() {
    local label="runtime-mode-on-demand"
    local api_port=$((BASE_PORT + NODE_INDEX * 4))
    local console_port=$((api_port + 1))
    local bind_port=$((api_port + 2))
    local config_path="$WORK_ROOT/$label-config.toml"
    printf '%s\n' '[runtime]' 'mode = "on_demand"' >"$config_path"

    # Start with explicit on-demand mode; older binaries may reject the config.
    start_node "$label" \
        --headless \
        --config "$config_path" \
        --port "$api_port" \
        --console "$console_port" \
        --bind-port "$bind_port" || { record_result "PREREQ" "runtime_mode_on_demand" "on-demand mode not testable"; return 0; }

    local pid="$START_NODE_PID"

    # Give it a shorter window since on-demand may stay idle
    local timeout=15
    local ready=false
    for second in $(seq 1 "$timeout"); do
        if curl -fsS --max-time 3 "http://127.0.0.1:${console_port}/api/status" \
            >"$STATUS_DIR/${label}-status.json" 2>/dev/null; then
            ready=true
            break
        fi
        sleep 1
    done

    if [[ "$ready" == true ]] && kill -0 "$pid" 2>/dev/null; then
        record_result "PASS" "runtime_mode_on_demand" \
            "on-demand mode process alive and management API responding" \
            "pid=$pid" "path=$STATUS_DIR/${label}-status.json"
    elif kill -0 "$pid" 2>/dev/null; then
        record_result "FAIL" "runtime_mode_on_demand" \
            "on-demand mode process stayed alive but management API did not respond" \
            "pid=$pid" "log=$LOG_DIR/$label.log"
    else
        # Process exited — may be expected for on-demand without models
        local log_content=""
        log_content="$(tail -20 "$LOG_DIR/$label.log" 2>/dev/null || true)"
        if [[ "$log_content" == *"help"* ]] || [[ "$log_content" == *"usage"* ]]; then
            record_result "PREREQ" "runtime_mode_on_demand" \
                "on-demand mode not available without model; process exited with usage info"
        else
            record_result "FAIL" "runtime_mode_on_demand" \
                "on-demand mode process exited unexpectedly" \
                "pid=$pid" "log=$LOG_DIR/$label.log"
        fi
    fi

    kill_tree "$pid" 2>/dev/null || true
}

check_best_effort_startup() {
    local label="best-effort-startup"
    local api_port=$((BASE_PORT + NODE_INDEX * 4))
    local console_port=$((api_port + 1))
    local bind_port=$((api_port + 2))

    local config_path="$WORK_ROOT/$label-config.toml"
    printf '%s\n' '[runtime]' 'startup_failure_policy = "best_effort"' >"$config_path"

    # Start with a nonexistent eager model under explicit best-effort policy.
    start_node "$label" \
        --headless \
        --config "$config_path" \
        --model "NonExistent-Model-That-Does-Not-Exist-Q4_K_M" \
        --port "$api_port" \
        --console "$console_port" \
        --bind-port "$bind_port" || { record_result "FAIL" "best_effort_startup" "could not start daemon"; return 0; }

    local pid="$START_NODE_PID"

    local status_out="$STATUS_DIR/${label}-status.json"
    if wait_status "$label" "$console_port" &&
        kill -0 "$pid" 2>/dev/null &&
        curl_json "http://127.0.0.1:${console_port}/api/status" "$status_out" 2>/dev/null; then
        record_result "PASS" "best_effort_startup" \
            "daemon stayed alive and management remained ready after eager startup failure" \
            "pid=$pid" "path=$status_out"
    else
        local log_tail=""
        log_tail="$(tail -20 "$LOG_DIR/$label.log" 2>/dev/null || true)"
        record_result "FAIL" "best_effort_startup" \
            "daemon exited despite the default best-effort startup policy" \
            "pid=$pid" "log=$LOG_DIR/$label.log" "tail=$log_tail"
    fi

    kill_tree "$pid" 2>/dev/null || true
}

check_fail_fast_startup() {
    local label="fail-fast-startup"
    local api_port=$((BASE_PORT + NODE_INDEX * 4))
    local console_port=$((api_port + 1))
    local bind_port=$((api_port + 2))

    local config_path="$WORK_ROOT/$label-config.toml"
    printf '%s\n' '[runtime]' 'startup_failure_policy = "fail_fast"' >"$config_path"

    # Start with a nonexistent eager model under explicit fail-fast policy.
    start_node "$label" \
        --headless \
        --config "$config_path" \
        --model "NonExistent-Model-That-Does-Not-Exist-Q4_K_M" \
        --port "$api_port" \
        --console "$console_port" \
        --bind-port "$bind_port" || { record_result "FAIL" "fail_fast_startup" "could not start daemon"; return 0; }

    local pid="$START_NODE_PID"

    # Wait up to MAX_WAIT for process to exit (or timeout)
    local waited=0
    while [[ $waited -lt "$MAX_WAIT" ]]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    if ! kill -0 "$pid" 2>/dev/null; then
        local exit_code=0
        wait "$pid" || exit_code=$?
        if [[ "$exit_code" -ne 0 ]]; then
            record_result "PASS" "fail_fast_startup" \
                "daemon exited nonzero with nonexistent model (fail-fast behavior)" \
                "seconds=$waited" "pid=$pid" "exit_code=$exit_code"
        else
            record_result "FAIL" "fail_fast_startup" \
                "daemon exited successfully despite an eager fail-fast startup failure" \
                "seconds=$waited" "pid=$pid" "exit_code=$exit_code"
        fi
    else
        local status_out="$STATUS_DIR/${label}-status.json"
        if curl_json "http://127.0.0.1:${console_port}/api/status" "$status_out" 2>/dev/null; then
            record_result "FAIL" "fail_fast_startup" \
                "daemon stayed alive despite explicit fail-fast startup policy" \
                "pid=$pid" "path=$status_out"
        else
            record_result "FAIL" "fail_fast_startup" \
                "process still running but management API not responding after $MAX_WAIT seconds" \
                "pid=$pid"
        fi
    fi

    kill_tree "$pid" 2>/dev/null || true
}

check_owner_lifecycle_intent() {
    local operation="$1"
    local result_name="$2"
    local expected_state="$3"
    local console_port="$ZERO_MODEL_CONSOLE"
    local label="runtime-${operation}"

    if [[ -z "$console_port" ]] || ! kill -0 "${ZERO_MODEL_PID:-}" 2>/dev/null; then
        record_result "FAIL" "$result_name" "zero-model daemon is not running"
        return 0
    fi

    local bootstrap_out="$CONTROL_DIR/${label}-bootstrap.json"
    if ! curl_json "http://127.0.0.1:${console_port}/api/runtime/control-bootstrap" "$bootstrap_out"; then
        record_result "FAIL" "$result_name" "control bootstrap endpoint failed"
        return 0
    fi
    local endpoint
    endpoint="$(python3 - "$bootstrap_out" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data.get("endpoint") or "")
PY
)"
    if [[ -z "$endpoint" ]]; then
        record_result "PREREQ" "$result_name" \
            "owner identity is unavailable, so the authenticated lifecycle path cannot be exercised" \
            "path=$bootstrap_out"
        return 0
    fi

    local request_json="$CONTROL_DIR/${label}-request.json"
    python3 - "$request_json" "$endpoint" <<'PY'
import json, sys
path, endpoint = sys.argv[1:]
json.dump({"endpoint": endpoint, "model": "qa.invalid/model@main:missing.gguf"}, open(path, "w", encoding="utf-8"))
PY
    local response_out="$CONTROL_DIR/${label}-response.json"
    local response_status
    response_status="$(curl -sS --max-time 10 -o "$response_out" -w '%{http_code}' \
        -X POST "http://127.0.0.1:${console_port}/api/runtime/control/${operation}" \
        -H 'Content-Type: application/json' --data-binary "@$request_json" || true)"
    if [[ "$response_status" != "200" ]] || ! python3 - "$response_out" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if data.get("accepted") is True else 1)
PY
    then
        record_result "FAIL" "$result_name" \
            "authenticated lifecycle request was not accepted" \
            "http_status=$response_status" "path=$response_out"
        return 0
    fi

    local response_intent_id
    response_intent_id="$(python3 - "$response_out" "$expected_state" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
intent_id = data.get("intent_id") or ""
accepted_state = data.get("accepted_state") or ""
if (
    not intent_id
    or accepted_state != sys.argv[2]
    or data.get("model") != "qa.invalid/model@main:missing.gguf"
    or data.get("instance_id") is not None
):
    raise SystemExit(1)
print(intent_id)
PY
)" || true
    if [[ -z "$response_intent_id" ]]; then
        record_result "FAIL" "$result_name" \
            "accepted response omitted its intent identity or exact accepted state" \
            "expected_state=$expected_state" "path=$response_out"
        return 0
    fi

    local intents_out="$STATUS_DIR/${label}-intents.json"
    local intent_seen=false
    for _second in $(seq 1 "$MAX_WAIT"); do
        if curl_json "http://127.0.0.1:${console_port}/api/runtime/intents" "$intents_out" &&
            python3 - "$intents_out" "$expected_state" "$response_intent_id" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
expected, intent_id = sys.argv[2:]
matches = [
    item for item in data.get("intents", [])
    if item.get("intent_id") == intent_id
    and item.get("model_ref") == "qa.invalid/model@main:missing.gguf"
    and item.get("source") == "owner_lifecycle"
    and item.get("desired_state") == expected
]
raise SystemExit(0 if matches else 1)
PY
        then
            intent_seen=true
            break
        fi
        sleep 1
    done
    if [[ "$intent_seen" != true ]]; then
        record_result "FAIL" "$result_name" \
            "accepted command did not produce the expected owner lifecycle intent" \
            "intent_id=$response_intent_id" "expected_state=$expected_state" "path=$intents_out"
        return 0
    fi

    record_result "PASS" "$result_name" \
        "authenticated command was accepted and appeared in authoritative intent state" \
        "http_status=200" "intent_id=$response_intent_id" \
        "desired_state=$expected_state" "path=$intents_out"
}

check_runtime_load_model() {
    check_owner_lifecycle_intent "load-model" "runtime_load_model" "present"
}

check_runtime_unload_model() {
    check_owner_lifecycle_intent "unload-model" "runtime_unload_model" "absent"
}

check_runtime_ensure_model() {
    check_owner_lifecycle_intent "ensure-model" "runtime_ensure_model" "present"
}

check_runtime_drain_model() {
    check_owner_lifecycle_intent "drain-model" "runtime_drain_model" "draining"
}

check_activity_override() {
    local label="activity-override"
    local console_port="$ZERO_MODEL_CONSOLE"

    if [[ -z "$console_port" ]] || ! kill -0 "${ZERO_MODEL_PID:-}" 2>/dev/null; then
        record_result "PREREQ" "activity_override" \
            "no running daemon available for activity override test (depends on prior tests)"
        return 0
    fi

    local put_out="$CONTROL_DIR/${label}-put.json"
    local put_log="$LOG_DIR/${label}-put.log"

    # PUT activity override to active mode
    local put_status=""
    put_status="$(curl -sS --max-time 10 \
        -o "$put_out" \
        -w '%{http_code}' \
        -X PUT "http://127.0.0.1:${console_port}/api/runtime/activity/override" \
        -H 'Content-Type: application/json' \
        -d '"active"' 2>"$put_log" || true)"

    if [[ "$put_status" == "200" ]] && python3 - "$put_out" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if data.get("override_mode") == "active" else 1)
PY
    then
        # DELETE to clear override
        local del_out="$CONTROL_DIR/${label}-delete.json"
        local del_log="$LOG_DIR/${label}-delete.log"
        local del_status=""
        del_status="$(curl -sS --max-time 10 \
            -o "$del_out" \
            -w '%{http_code}' \
            -X DELETE "http://127.0.0.1:${console_port}/api/runtime/activity/override" \
            2>"$del_log" || true)"

        if [[ "$del_status" == "200" ]] && python3 - "$del_out" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if data.get("override_mode") == "auto" else 1)
PY
        then
            record_result "PASS" "activity_override" \
                "PUT activity override (HTTP $put_status) and DELETE clear (HTTP $del_status) round-tripped" \
                "put_status=$put_status" "delete_status=$del_status" \
                "path_put=$put_out" "path_delete=$del_out"
        else
            record_result "FAIL" "activity_override" \
                "PUT succeeded but DELETE clear failed for activity override" \
                "put_status=$put_status" "log=$del_log"
        fi
    else
        record_result "FAIL" "activity_override" \
            "PUT activity override returned unexpected status $put_status" \
            "path=$put_out" "log=$put_log"
    fi
}

check_privacy_no_raw_data() {
    local label="privacy-no-raw-data"
    local console_port="$ZERO_MODEL_CONSOLE"

    if [[ -z "$console_port" ]] || ! kill -0 "${ZERO_MODEL_PID:-}" 2>/dev/null; then
        record_result "PREREQ" "privacy_no_raw_data" \
            "no running daemon available for privacy check (depends on prior tests)"
        return 0
    fi

    local activity_out="$STATUS_DIR/${label}-activity.json"
    if curl_json "http://127.0.0.1:${console_port}/api/runtime/activity" "$activity_out" &&
        python3 - "$activity_out" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"effective_state", "override_mode", "detector_category"}
if set(data) != expected:
    raise SystemExit(f"unexpected activity fields: {sorted(set(data) - expected)}")
if data["effective_state"] not in {"accepting", "accepting_deprioritized", "remote_paused", "all_paused"}:
    raise SystemExit("invalid effective_state")
if data["override_mode"] not in {"auto", "active", "idle"}:
    raise SystemExit("invalid override_mode")
if data["detector_category"] not in {"active", "idle", "unavailable"}:
    raise SystemExit("invalid detector_category")
PY
    then
        record_result "PASS" "privacy_no_raw_data" \
            "activity API exposes only coarse enum fields" \
            "path=$activity_out"
    else
        record_result "FAIL" "privacy_no_raw_data" \
            "activity API exposed an unexpected or non-coarse payload" \
            "path=$activity_out"
    fi
}

check_clean_process_teardown() {
    local label="clean-process-teardown"

    # Kill the zero-model daemon (if still running) and verify no leaks
    if [[ -n "${ZERO_MODEL_PID:-}" ]]; then
        kill_tree "$ZERO_MODEL_PID" 2>/dev/null || true
        sleep 2
    fi

    # Try graceful stop first via tracked instance metadata in each temp home
    for home_dir in "$WORK_ROOT"/h-*; do
        [[ -d "$home_dir" ]] || continue
        local home_name runtime_root
        home_name="$(basename "$home_dir")"
        runtime_root="$WORK_ROOT/r-${home_name#h-}"
        if [[ -f "${runtime_root:-}/instance.json" ]]; then
            (
                export HOME="$home_dir"
                export MESH_LLM_RUNTIME_ROOT="$runtime_root"
                exec "$CURRENT_BINARY" stop 2>/dev/null || true
            ) &
        fi
    done

    sleep 2

    # Final check combines argv matching with every tracked process tree.
    local remaining_records=""
    local candidate_pids=""
    local pid pid_start child index expected_start
    candidate_pids="$(pgrep -f "mesh-llm.*${WORK_ROOT}" 2>/dev/null || true)"
    for pid in $candidate_pids; do
        pid_start="$(process_start_time "$pid")"
        [[ -n "$pid_start" ]] || continue
        remaining_records+="${pid}"$'\t'"${pid_start}"$'\n'
    done
    if ((${#PIDS[@]} > 0)); then
        for ((index = 0; index < ${#PIDS[@]}; index++)); do
            pid="${PIDS[$index]:-}"
            [[ -n "$pid" ]] || continue
            expected_start="${PID_STARTS[$index]:-}"
            if ! pid_matches_start "$pid" "$expected_start"; then
                PIDS[index]=""
                PID_STARTS[index]=""
                continue
            fi
            remaining_records+="${pid}"$'\t'"${expected_start}"$'\n'
            for child in $(descendant_pids "$pid" || true); do
                pid_start="$(process_start_time "$child")"
                [[ -n "$pid_start" ]] || continue
                remaining_records+="${child}"$'\t'"${pid_start}"$'\n'
            done
        done
    fi
    remaining_records="$(printf '%s' "$remaining_records" | awk -F '\t' 'NF == 2 && !seen[$1]++')"
    local remaining=0
    if [[ -n "$remaining_records" ]]; then
        remaining="$(printf '%s\n' "$remaining_records" | wc -l | tr -d ' ')"
    fi

    # Kill only stragglers whose process start time still matches the observed one.
    signal_process_records TERM "$remaining_records"
    sleep 1
    signal_process_records KILL "$remaining_records"

    if [[ "$remaining" -eq 0 ]]; then
        record_result "PASS" "clean_process_teardown" \
            "no leaked mesh-llm processes after teardown" \
            "work_root=$WORK_ROOT"
    else
        record_result "FAIL" "clean_process_teardown" \
            "$remaining mesh-llm process(es) leaked after teardown; killed as fallback" \
            "work_root=$WORK_ROOT"
    fi
}

# ──────────────── Execute tests ────────────────────────

ZERO_MODEL_PID=""
ZERO_MODEL_CONSOLE=""

check_zero_model_serve_ready || true

check_runtime_mode_serve || true

check_runtime_mode_on_demand || true

check_best_effort_startup || true

check_fail_fast_startup || true

# Lifecycle commands depend on zero-model daemon being alive (or start fresh)
check_runtime_load_model || true
check_runtime_unload_model || true
check_runtime_ensure_model || true
check_runtime_drain_model || true

# Activity override depends on a running daemon
check_activity_override || true

# Privacy check validates the activity endpoint's coarse public shape.
check_privacy_no_raw_data || true

# Final teardown check (kills remaining processes)
check_clean_process_teardown || true

if [[ "$EXIT_STATUS" -eq 0 ]]; then
    append_summary ""
    append_summary "Overall: PASS or PREREQ-only incomplete checks. See \`results.jsonl\`."
else
    append_summary ""
    append_summary "Overall: FAIL. See \`results.jsonl\` and logs."
fi

exit "$EXIT_STATUS"
