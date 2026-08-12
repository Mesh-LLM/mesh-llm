#!/usr/bin/env bash
# Certify logging restart persistence, delete cascade, and fail-open behavior.

set -euo pipefail

CURRENT_BINARY=""
EVIDENCE_DIR=".sisyphus/evidence"
BASE_PORT="${MESH_QA_BASE_PORT:-19860}"
MAX_WAIT="${MESH_QA_MAX_WAIT:-45}"
DETERMINISTIC_OPENAI_ENDPOINT=""
DETERMINISTIC_OPENAI_MODEL="qa-deterministic"
KEEP_LOGS=false
PRINT_PLAN=false
TMP_ROOT="${MESH_QA_TMP_ROOT:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}}"

RUN_ID=""
RUN_DIR=""
WORK_ROOT=""
LOG_DIR=""
STATUS_DIR=""
REQUESTS_DIR=""
VERSIONS_DIR=""
RESULTS_JSONL=""
COMMANDS_JSONL=""
MANIFEST_JSON=""
SUMMARY_JSON=""
SUMMARY_MD=""
PIDS=()
PID_STARTS=()
EXIT_STATUS=0
NODE_INDEX=0

usage() {
    cat <<'EOF'
Usage:
  scripts/qa-logging-recovery.sh --current-binary PATH [options]

Certifies process-level logging restart/privacy, delete-cascade retention, and
fail-open behavior. It starts only harness-owned local processes and writes
machine-readable evidence.

Required:
  --current-binary PATH                 Current mesh-llm binary.

Options:
  --evidence-dir DIR                    Evidence root (default: .sisyphus/evidence).
  --base-port PORT                      First local port (default: 19860).
  --max-wait SECONDS                    Per-node readiness cap (default: 45).
  --deterministic-openai-endpoint URL   Explicit local deterministic OpenAI endpoint plugin URL.
  --deterministic-openai-model MODEL    Model served by the deterministic endpoint.
  --keep-logs                           Preserve successful temporary process logs.
  --print-plan                          Print JSON plan without files or processes.
  -h, --help                            Show this help.

Checks:
  logging_restart_privacy      Persist a rejected OpenAI lifecycle over restart and keep its private marker out of summaries.
  logging_retention_cascade    Delete that terminal request with POST /api/logs/requests/{id}/delete and a caller operationId.
  logging_trusted_local_rejection  Hostile Host/Origin callers receive the typed 403 forbidden response before log dispatch.
  logging_sse_recovery          Reconnect with Last-Event-ID/cursor and prove an evicted replay_gap frame is privacy-safe.
  logging_fail_open            A deliberately unusable logging state root leaves the management process available.
  logging_fail_open_inference  With an explicitly supplied deterministic OpenAI endpoint plugin, inference succeeds while logging is unavailable; otherwise PREREQ.

Evidence per run:
  manifest.json, commands.jsonl, results.jsonl, summary.json, summary.md,
  logs/, status/, requests/, and versions/.

Result vocabulary: PASS, FAIL, PREREQ.
EOF
}

fail_usage() {
    echo "error: $*" >&2
    usage >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --current-binary) CURRENT_BINARY="${2:-}"; shift 2 ;;
        --evidence-dir) EVIDENCE_DIR="${2:-}"; shift 2 ;;
        --base-port) BASE_PORT="${2:-}"; shift 2 ;;
        --max-wait) MAX_WAIT="${2:-}"; shift 2 ;;
        --deterministic-openai-endpoint) DETERMINISTIC_OPENAI_ENDPOINT="${2:-}"; shift 2 ;;
        --deterministic-openai-model) DETERMINISTIC_OPENAI_MODEL="${2:-}"; shift 2 ;;
        --keep-logs) KEEP_LOGS=true; shift ;;
        --print-plan) PRINT_PLAN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail_usage "unknown argument: $1" ;;
    esac
done

[[ -n "$CURRENT_BINARY" ]] || fail_usage "missing required option: --current-binary"
for numeric in BASE_PORT MAX_WAIT; do
    value="${!numeric}"
    [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || \
        fail_usage "--$(printf '%s' "$numeric" | tr '[:upper:]_' '[:lower:]-') must be a positive integer"
done

require_tool() {
    command -v "$1" >/dev/null 2>&1 || { echo "error: missing required tool: $1" >&2; exit 2; }
}

require_tool python3

if [[ "$PRINT_PLAN" == true ]]; then
    python3 - "$CURRENT_BINARY" "$EVIDENCE_DIR" "$BASE_PORT" "$MAX_WAIT" \
        "$DETERMINISTIC_OPENAI_ENDPOINT" "$DETERMINISTIC_OPENAI_MODEL" <<'PY'
import json
import sys

binary, evidence, base_port, max_wait, endpoint, model = sys.argv[1:]
print(json.dumps({
    "script": "qa-logging-recovery.sh",
    "current_binary": binary,
    "evidence_dir": evidence,
    "base_port": int(base_port),
    "max_wait_seconds": int(max_wait),
    "deterministic_openai_endpoint_supplied": bool(endpoint),
    "deterministic_openai_model": model if endpoint else None,
    "optional_plugin_behavior": {
        "without_endpoint": {"logging_fail_open_inference": "PREREQ"},
        "with_endpoint": {"logging_fail_open_inference": "execute"},
    },
    "checks": [
        "logging_restart_privacy",
        "logging_retention_cascade",
        "logging_trusted_local_rejection",
        "logging_sse_recovery",
        "logging_fail_open",
        "logging_fail_open_inference",
        "cleanup",
    ],
    "evidence_files": [
        "manifest.json", "commands.jsonl", "results.jsonl", "summary.json", "summary.md",
        "logs/", "status/", "requests/", "versions/",
    ],
}, sort_keys=True, separators=(",", ":")))
PY
    exit 0
fi

[[ -x "$CURRENT_BINARY" ]] || fail_usage "--current-binary is not executable: $CURRENT_BINARY"
require_tool curl
require_tool date
require_tool mktemp
require_tool pgrep
require_tool ps

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${EVIDENCE_DIR%/}/logging-recovery-${RUN_ID}"
WORK_ROOT="$(mktemp -d "${TMP_ROOT%/}/mesh-logging-recovery.XXXXXX")"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
REQUESTS_DIR="$RUN_DIR/requests"
VERSIONS_DIR="$RUN_DIR/versions"
RESULTS_JSONL="$RUN_DIR/results.jsonl"
COMMANDS_JSONL="$RUN_DIR/commands.jsonl"
MANIFEST_JSON="$RUN_DIR/manifest.json"
SUMMARY_JSON="$RUN_DIR/summary.json"
SUMMARY_MD="$RUN_DIR/summary.md"
mkdir -p "$LOG_DIR" "$STATUS_DIR" "$REQUESTS_DIR" "$VERSIONS_DIR"
: >"$RESULTS_JSONL"
: >"$COMMANDS_JSONL"

append_summary() { printf '%s\n' "$*" >>"$SUMMARY_MD"; }

record_command() {
    local name="$1" log="$2"
    python3 - "$COMMANDS_JSONL" "$name" "$log" <<'PY'
import json
import sys
path, name, log = sys.argv[1:]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"name": name, "log": log}, sort_keys=True) + "\n")
PY
}

record_result() {
    local status="$1" name="$2" message="$3"
    shift 3
    case "$status" in PASS|FAIL|PREREQ) ;; *) echo "invalid result status: $status" >&2; exit 2 ;; esac
    python3 - "$RESULTS_JSONL" "$status" "$name" "$message" "$@" <<'PY'
import json
import sys
path, status, name, message, *fields = sys.argv[1:]
record = {"status": status, "name": name, "message": message}
for field in fields:
    if "=" in field:
        key, value = field.split("=", 1)
        record[key] = value
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
PY
    append_summary "- $status $name: $message"
    [[ "$status" != FAIL ]] || EXIT_STATUS=1
}

write_manifest() {
    python3 - "$MANIFEST_JSON" "$RUN_ID" "$CURRENT_BINARY" "$RUN_DIR" "$WORK_ROOT" \
        "$BASE_PORT" "$MAX_WAIT" "$DETERMINISTIC_OPENAI_ENDPOINT" "$DETERMINISTIC_OPENAI_MODEL" <<'PY'
import json
import sys
path, run_id, binary, run_dir, work_root, base_port, max_wait, endpoint, model = sys.argv[1:]
json.dump({
    "run_id": run_id,
    "current_binary": binary,
    "evidence_dir": run_dir,
    "work_root": work_root,
    "base_port": int(base_port),
    "max_wait_seconds": int(max_wait),
    "deterministic_openai_endpoint_supplied": bool(endpoint),
    "deterministic_openai_model": model if endpoint else None,
}, open(path, "w", encoding="utf-8"), indent=2, sort_keys=True)
open(path, "a", encoding="utf-8").write("\n")
PY
}

write_summary_json() {
    python3 - "$RESULTS_JSONL" "$SUMMARY_JSON" "$RUN_DIR" <<'PY'
import json
import sys
results_path, summary_path, run_dir = sys.argv[1:]
with open(results_path, encoding="utf-8") as handle:
    results = [json.loads(line) for line in handle if line.strip()]
statuses = [row["status"] for row in results]
overall = "fail" if "FAIL" in statuses else "prereq" if "PREREQ" in statuses else "pass"
summary = {"overall": overall, "evidence_dir": run_dir, "counts": {
    status.lower(): statuses.count(status) for status in ("PASS", "FAIL", "PREREQ")
}, "results": results}
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

process_start_time() { ps -o lstart= -p "$1" 2>/dev/null | awk 'NF { $1=$1; print; exit }'; }

pid_matches_start() {
    local pid="$1" expected="$2" actual
    [[ -n "$expected" ]] || return 1
    actual="$(process_start_time "$pid")"
    [[ -n "$actual" && "$actual" == "$expected" ]]
}

descendant_pids() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        descendant_pids "$child"
        printf '%s\n' "$child"
    done
}

kill_tree() {
    local pid="$1" expected="$2" children child
    pid_matches_start "$pid" "$expected" || return 0
    children="$(descendant_pids "$pid" | sort -u || true)"
    kill -TERM "$pid" 2>/dev/null || true
    for child in $children; do kill -TERM "$child" 2>/dev/null || true; done
    local deadline=$((SECONDS + 3))
    while pid_matches_start "$pid" "$expected" && [[ $SECONDS -lt $deadline ]]; do sleep 1; done
    if pid_matches_start "$pid" "$expected"; then kill -KILL "$pid" 2>/dev/null || true; fi
    for child in $children; do kill -KILL "$child" 2>/dev/null || true; done
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    local incoming_status=$? index alive=0
    for ((index=0; index<${#PIDS[@]}; index++)); do
        [[ -n "${PIDS[$index]:-}" ]] || continue
        kill_tree "${PIDS[$index]}" "${PID_STARTS[$index]}"
    done
    for ((index=0; index<${#PIDS[@]}; index++)); do
        [[ -n "${PIDS[$index]:-}" ]] || continue
        if pid_matches_start "${PIDS[$index]}" "${PID_STARTS[$index]}"; then alive=$((alive + 1)); fi
    done
    if [[ $alive -eq 0 ]]; then
        record_result PASS cleanup "harness-owned processes stopped" "processes=0"
    else
        record_result FAIL cleanup "harness-owned processes remain" "processes=$alive"
    fi
    write_summary_json
    if [[ "$EXIT_STATUS" -eq 0 && "$KEEP_LOGS" != true ]]; then
        rm -rf "$WORK_ROOT"
    fi
    trap - EXIT
    [[ $incoming_status -eq 0 ]] || exit "$incoming_status"
    exit "$EXIT_STATUS"
}
trap cleanup EXIT

write_manifest
append_summary "# Logging recovery certification"
append_summary ""
append_summary "- Run ID: \`$RUN_ID\`"
append_summary "- Evidence: \`$RUN_DIR\`"
append_summary ""

version_log="$LOG_DIR/current-version.log"
record_command current-version "$version_log"
if "$CURRENT_BINARY" --version >"$VERSIONS_DIR/current.txt" 2>"$version_log"; then
    record_result PASS prereq.current_binary "current binary reported a version" "path=$VERSIONS_DIR/current.txt"
else
    record_result PREREQ prereq.current_binary "current binary did not report a version" "log=$version_log"
fi

write_config() {
    local config="$1" state_root="$2" endpoint="$3"
    python3 - "$config" "$state_root" "$endpoint" <<'PY'
import json
import sys
config, state_root, endpoint = sys.argv[1:]
lines = [
    "[logging]",
    "enabled = true",
    f"application_state_root = {json.dumps(state_root)}",
    "retention_max_rows = 64",
    "replay_capacity = 1",
    "cleanup_cadence_secs = 300",
    "[logging.artifact]",
    'capture_mode = "metadata_only"',
]
if endpoint:
    lines.extend(["", "[[plugin]]", 'name = "openai-endpoint"', f"url = {json.dumps(endpoint)}"])
with open(config, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
PY
}

start_node() {
    local label="$1" config="$2"
    NODE_INDEX=$((NODE_INDEX + 1))
    local api_port=$((BASE_PORT + NODE_INDEX * 10))
    local console_port=$((api_port + 1))
    local bind_port=$((api_port + 2))
    local home="$WORK_ROOT/home-$NODE_INDEX" runtime_root="$WORK_ROOT/runtime-$NODE_INDEX"
    local log="$LOG_DIR/$label.log"
    mkdir -p "$home" "$runtime_root"
    record_command "start-$label" "$log"
    (
        export HOME="$home" MESH_LLM_RUNTIME_ROOT="$runtime_root" MESH_LLM_EPHEMERAL_KEY=1
        exec "$CURRENT_BINARY" serve --headless --config "$config" --port "$api_port" --console "$console_port" --bind-port "$bind_port"
    ) >"$log" 2>&1 &
    START_NODE_PID=$!
    START_NODE_START="$(process_start_time "$START_NODE_PID")"
    START_API_PORT="$api_port"
    START_CONSOLE_PORT="$console_port"
    PIDS+=("$START_NODE_PID")
    PID_STARTS+=("$START_NODE_START")
}

wait_status() {
    local label="$1" console_port="$2" second
    local output="$STATUS_DIR/$label.json"
    for ((second=1; second<=MAX_WAIT; second++)); do
        if curl -fsS --max-time 3 "http://127.0.0.1:$console_port/api/status" -o "$output"; then return 0; fi
        sleep 1
    done
    return 1
}

request_id_from_page() {
    python3 - "$1" <<'PY'
import json
import sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
    items = payload.get("items", [])
    value = items[0].get("requestId") if items else None
    if isinstance(value, str): print(value)
except (OSError, ValueError, AttributeError):
    pass
PY
}

list_requests_until_found() {
    local console_port="$1" output="$2" second
    for ((second=1; second<=MAX_WAIT; second++)); do
        if curl -fsS --max-time 3 "http://127.0.0.1:$console_port/api/logs/requests?limit=10" -o "$output"; then
            REQUEST_ID="$(request_id_from_page "$output")"
            [[ -n "$REQUEST_ID" ]] && return 0
        fi
        sleep 1
    done
    return 1
}

private_marker="QA_PRIVATE_MARKER_NOT_FOR_PERSISTENCE"
submit_rejected_request() {
    local api_port="$1" output="$2"
    record_command rejected-openai-request "$output"
    curl -sS --max-time 8 -o "$output" -w '%{http_code}' \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"qa-no-model\",\"messages\":[{\"role\":\"user\",\"content\":\"$private_marker\"}]}" \
        "http://127.0.0.1:$api_port/v1/chat/completions" >"$output.status" || true
}

assert_no_private_marker() {
    local check="$1"; shift
    if python3 - "$private_marker" "$@" <<'PY'
import sys
marker, *paths = sys.argv[1:]
for path in paths:
    try:
        if marker in open(path, encoding="utf-8").read():
            raise SystemExit(1)
    except OSError:
        raise SystemExit(1)
PY
    then
        return 0
    fi
    record_result FAIL "$check" "private request marker appeared in a logging API response"
    return 1
}

typed_forbidden_response() {
    python3 - "$1" <<'PY'
import json
import sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
    error = payload.get("error")
    code = error.get("code") if isinstance(error, dict) else payload.get("code")
    raise SystemExit(0 if code == "forbidden" else 1)
except (OSError, ValueError, AttributeError):
    raise SystemExit(1)
PY
}

check_trusted_local_rejection() {
    local console_port="$1"
    local hostile_host="$REQUESTS_DIR/trusted-host-response.json"
    local hostile_origin="$REQUESTS_DIR/trusted-origin-response.json"
    local host_status origin_status
    record_command trusted-host-rejection "$hostile_host"
    host_status="$(curl -sS --max-time 5 -o "$hostile_host" -w '%{http_code}' \
        -H 'Host: attacker.example' -H 'Accept: text/event-stream' \
        "http://127.0.0.1:$console_port/api/logs/events?channel=requests" || true)"
    record_command trusted-origin-rejection "$hostile_origin"
    origin_status="$(curl -sS --max-time 5 -o "$hostile_origin" -w '%{http_code}' \
        -H "Host: localhost:$console_port" -H 'Origin: https://attacker.example' \
        -H 'Accept: text/event-stream' \
        "http://127.0.0.1:$console_port/api/logs/events?channel=requests" || true)"
    if [[ "$host_status" == 403 && "$origin_status" == 403 ]] \
        && typed_forbidden_response "$hostile_host" \
        && typed_forbidden_response "$hostile_origin"; then
        record_result PASS logging_trusted_local_rejection "hostile Host and Origin callers received typed 403 forbidden responses" \
            "host_status=$host_status" "origin_status=$origin_status"
    else
        record_result FAIL logging_trusted_local_rejection "trusted-local rejection did not return typed 403 responses" \
            "host_status=$host_status" "origin_status=$origin_status"
    fi
}

check_sse_recovery() {
    local api_port="$1" console_port="$2"
    local first="$REQUESTS_DIR/sse-seed-one.json" second="$REQUESTS_DIR/sse-seed-two.json"
    local page="$REQUESTS_DIR/sse-seed-list.json"
    local headers="$REQUESTS_DIR/sse-reconnect-headers.txt" body="$REQUESTS_DIR/sse-reconnect-body.txt"
    submit_rejected_request "$api_port" "$first"
    submit_rejected_request "$api_port" "$second"
    if ! list_requests_until_found "$console_port" "$page"; then
        record_result PREREQ logging_sse_recovery "could not create deterministic request events for replay eviction" "log=$LOG_DIR/restart-after.log"
        return 0
    fi
    record_command sse-reconnect-replay-gap "$body"
    curl -sS -N --max-time 5 -D "$headers" -o "$body" \
        -H 'Accept: text/event-stream' \
        -H 'Last-Event-ID: v1:0.0.0' \
        "http://127.0.0.1:$console_port/api/logs/events?channel=requests&cursor=v1%3A0.0.0" || true
    if python3 - "$headers" "$body" "$private_marker" <<'PY'
import sys
headers, body, marker = sys.argv[1:]
try:
    header_text = open(headers, encoding="utf-8").read()
    body_text = open(body, encoding="utf-8").read()
except OSError:
    raise SystemExit(1)
required = ("HTTP/1.1 200", "text/event-stream", "event: replay_gap", "/api/logs/requests")
if not all(value in header_text + body_text for value in required):
    raise SystemExit(1)
if marker in body_text or "private/operator" in body_text or "token=" in body_text:
    raise SystemExit(1)
raise SystemExit(0)
PY
    then
        record_result PASS logging_sse_recovery "Last-Event-ID/cursor reconnect produced a bounded privacy-safe replay gap" \
            "headers=$headers" "body=$body"
    elif ! python3 - "$headers" <<'PY'
import sys
try:
    raise SystemExit(0 if "HTTP/1.1 200" in open(sys.argv[1], encoding="utf-8").read() else 1)
except OSError:
    raise SystemExit(1)
PY
    then
        record_result FAIL logging_sse_recovery "SSE reconnect did not produce the expected privacy-safe replay gap" \
            "headers=$headers" "body=$body"
    else
        record_result PREREQ logging_sse_recovery "SSE endpoint was not available for deterministic replay-gap setup" \
            "headers=$headers" "body=$body"
    fi
}

restart_config="$WORK_ROOT/restart.toml"
restart_state="$WORK_ROOT/restart-state"
write_config "$restart_config" "$restart_state" ""
start_node restart-before "$restart_config"
restart_first_pid="$START_NODE_PID"
restart_first_start="$START_NODE_START"
restart_api="$START_API_PORT"
restart_console="$START_CONSOLE_PORT"
if ! wait_status restart-before "$restart_console"; then
    record_result FAIL logging_restart_privacy "initial logging process did not become ready" "log=$LOG_DIR/restart-before.log"
elif ! submit_rejected_request "$restart_api" "$REQUESTS_DIR/rejected-before-restart.json"; then
    record_result FAIL logging_restart_privacy "could not submit rejected OpenAI lifecycle request"
elif ! list_requests_until_found "$restart_console" "$REQUESTS_DIR/list-before-restart.json"; then
    record_result PREREQ logging_restart_privacy "no durable request summary became available from the zero-model process" "log=$LOG_DIR/restart-before.log"
else
    restart_request_id="$REQUEST_ID"
    kill_tree "$restart_first_pid" "$restart_first_start"
    start_node restart-after "$restart_config"
    restart_after_pid="$START_NODE_PID"
    restart_after_start="$START_NODE_START"
    restart_after_api="$START_API_PORT"
    restart_after_console="$START_CONSOLE_PORT"
    detail_after="$REQUESTS_DIR/detail-after-restart.json"
    if ! wait_status restart-after "$restart_after_console"; then
        record_result FAIL logging_restart_privacy "restarted process did not become ready" "log=$LOG_DIR/restart-after.log"
    elif ! curl -fsS --max-time 5 "http://127.0.0.1:$restart_after_console/api/logs/requests/$restart_request_id" -o "$detail_after"; then
        record_result FAIL logging_restart_privacy "durable request was unavailable after restart" "request_id=$restart_request_id"
    elif assert_no_private_marker logging_restart_privacy "$REQUESTS_DIR/list-before-restart.json" "$detail_after"; then
        record_result PASS logging_restart_privacy "durable request survived restart and summaries omitted the private marker" "request_id=$restart_request_id"
    fi
    if pid_matches_start "$restart_after_pid" "$restart_after_start"; then
        check_trusted_local_rejection "$restart_after_console"
    else
        record_result PREREQ logging_trusted_local_rejection "restarted process was not available for a trusted-local route probe"
    fi
fi
if [[ -z "${restart_request_id:-}" ]]; then
    record_result PREREQ logging_trusted_local_rejection "restart process did not produce a terminal request context"
fi

if [[ -n "${restart_request_id:-}" && -n "${restart_after_console:-}" ]]; then
    delete_output="$REQUESTS_DIR/delete-receipt.json"
    delete_operation_id="$(python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
    record_command logging-retention-delete "$delete_output"
    if curl -fsS --max-time 5 -X POST \
        -H 'Content-Type: application/json' \
        -d "{\"operationId\":\"$delete_operation_id\",\"reason\":\"qa retention cascade\"}" \
        "http://127.0.0.1:$restart_after_console/api/logs/requests/$restart_request_id/delete" \
        -o "$delete_output" \
        && python3 - "$delete_output" "$delete_operation_id" "$restart_request_id" <<'PY'
import json
import sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if body.get("operationId") == sys.argv[2] and body.get("requestId") == sys.argv[3] else 1)
PY
    then
        deleted_detail="$REQUESTS_DIR/detail-after-delete.json"
        http_code="$(curl -sS --max-time 5 -o "$deleted_detail" -w '%{http_code}' "http://127.0.0.1:$restart_after_console/api/logs/requests/$restart_request_id" || true)"
        if [[ "$http_code" == 404 ]]; then
            record_result PASS logging_retention_cascade "delete receipt used the caller operationId and removed the durable request" "request_id=$restart_request_id" "operation_id=$delete_operation_id"
        else
            record_result FAIL logging_retention_cascade "request remained queryable after delete cascade" "http_status=$http_code"
        fi
    else
        record_result FAIL logging_retention_cascade "v6 delete request did not return a matching cascade receipt" "request_id=$restart_request_id"
    fi
else
    record_result PREREQ logging_retention_cascade "restart check did not produce a terminal request eligible for delete cascade"
fi

if [[ -n "${restart_after_console:-}" ]] \
    && pid_matches_start "${restart_after_pid:-}" "${restart_after_start:-}"; then
    check_sse_recovery "$restart_after_api" "$restart_after_console"
else
    record_result PREREQ logging_sse_recovery "restart process was not available for deterministic SSE replay-gap setup"
fi

failure_state="$WORK_ROOT/logging-state-file"
: >"$failure_state"
failure_config="$WORK_ROOT/fail-open.toml"
write_config "$failure_config" "$failure_state" "$DETERMINISTIC_OPENAI_ENDPOINT"
start_node fail-open "$failure_config"
fail_open_pid="$START_NODE_PID"
fail_open_start="$START_NODE_START"
fail_open_api="$START_API_PORT"
fail_open_console="$START_CONSOLE_PORT"
if ! wait_status fail-open "$fail_open_console"; then
    record_result FAIL logging_fail_open "process with an unusable logging root did not become ready" "log=$LOG_DIR/fail-open.log"
elif ! pid_matches_start "$fail_open_pid" "$fail_open_start"; then
    record_result FAIL logging_fail_open "process exited after logging initialization failed" "log=$LOG_DIR/fail-open.log"
else
    log_status="$(curl -sS --max-time 5 -o "$REQUESTS_DIR/fail-open-logs.json" -w '%{http_code}' "http://127.0.0.1:$fail_open_console/api/logs/requests" || true)"
    if [[ "$log_status" == 503 ]]; then
        record_result PASS logging_fail_open "management process stayed ready while unavailable logging returned a typed response" "logs_http_status=$log_status"
    else
        record_result FAIL logging_fail_open "unusable logging root did not yield the expected unavailable logging response" "logs_http_status=$log_status"
    fi
fi

if [[ -z "$DETERMINISTIC_OPENAI_ENDPOINT" ]]; then
    record_result PREREQ logging_fail_open_inference "no deterministic OpenAI endpoint plugin was supplied"
elif ! pid_matches_start "$fail_open_pid" "$fail_open_start"; then
    record_result PREREQ logging_fail_open_inference "fail-open process is not available for the supplied deterministic endpoint"
else
    inference_output="$REQUESTS_DIR/fail-open-inference.json"
    record_command fail-open-inference "$inference_output"
    inference_status="$(curl -sS --max-time 20 -o "$inference_output" -w '%{http_code}' \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$DETERMINISTIC_OPENAI_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"return deterministic QA output\"}]}" \
        "http://127.0.0.1:$fail_open_api/v1/chat/completions" || true)"
    if [[ "$inference_status" =~ ^2[0-9][0-9]$ ]]; then
        record_result PASS logging_fail_open_inference "deterministic endpoint inference succeeded while logging was unavailable" "http_status=$inference_status"
    else
        record_result FAIL logging_fail_open_inference "deterministic endpoint inference failed while logging was unavailable" "http_status=$inference_status" "response=$inference_output"
    fi
fi
