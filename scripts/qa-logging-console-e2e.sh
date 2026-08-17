#!/usr/bin/env bash
# Exercise the checked-in console bundle against a real, isolated mesh-llm
# process. This intentionally does not intercept /api/logs/** requests.

set -euo pipefail

CURRENT_BINARY=""
EVIDENCE_ROOT=".sisyphus/evidence"
BASE_PORT="${MESH_QA_BASE_PORT:-20960}"
MAX_WAIT="${MESH_QA_MAX_WAIT:-45}"
KEEP_STATE=false
PRINT_PLAN=false

usage() {
    cat <<'EOF'
Usage: scripts/qa-logging-console-e2e.sh --current-binary PATH [options]

Build-facing real-console certification for operator logging. The harness owns
an isolated HOME, runtime root, application-state root, ports, and process
tree. It records machine-readable evidence and never mocks /api/logs/**.

Options:
  --current-binary PATH  Built mesh-llm binary to run (required).
  --evidence-dir DIR     Evidence root (default: .sisyphus/evidence).
  --base-port PORT       Loopback API port (default: 20960).
  --max-wait SECONDS     Startup and durable-record wait cap (default: 45).
  --keep-state           Keep isolated HOME/runtime/application state on success.
  --print-plan           Print the side-effect-free machine-readable test plan.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --current-binary) CURRENT_BINARY="${2:-}"; shift 2 ;;
        --evidence-dir) EVIDENCE_ROOT="${2:-}"; shift 2 ;;
        --base-port) BASE_PORT="${2:-}"; shift 2 ;;
        --max-wait) MAX_WAIT="${2:-}"; shift 2 ;;
        --keep-state) KEEP_STATE=true; shift ;;
        --print-plan) PRINT_PLAN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$CURRENT_BINARY" ]] || { echo "error: --current-binary is required" >&2; exit 2; }
[[ "$BASE_PORT" =~ ^[0-9]+$ && "$BASE_PORT" -gt 1024 && "$BASE_PORT" -le 65533 ]] || {
    echo "error: invalid --base-port" >&2
    exit 2
}
[[ "$MAX_WAIT" =~ ^[0-9]+$ && "$MAX_WAIT" -gt 0 ]] || { echo "error: invalid --max-wait" >&2; exit 2; }

if [[ "$PRINT_PLAN" == true ]]; then
    python3 - "$CURRENT_BINARY" "$EVIDENCE_ROOT" "$BASE_PORT" "$MAX_WAIT" <<'PY'
import json
import sys
binary, evidence, port, wait = sys.argv[1:]
print(json.dumps({
    "script": "qa-logging-console-e2e.sh",
    "binary": binary,
    "evidence_root": evidence,
    "base_port": int(port),
    "max_wait_seconds": int(wait),
    "checks": [
        "real_embedded_console_bundle",
        "real_openai_lifecycle_and_detail",
        "restart_persistence",
        "trusted_local_rejection",
        "dedicated_sse_replay_gap_and_authoritative_hydration",
        "real_console_accessibility_and_responsive_modes",
        "cleanup",
    ],
    "no_mocked_logs_routes": True,
}, sort_keys=True))
PY
    exit 0
fi

for tool in curl pnpm pgrep ps python3 rg; do
    command -v "$tool" >/dev/null 2>&1 || { echo "error: missing required tool: $tool" >&2; exit 2; }
done
[[ -x "$CURRENT_BINARY" ]] || { echo "error: binary is not executable: $CURRENT_BINARY" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_ROOT="$REPO_ROOT/crates/mesh-llm-ui"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${EVIDENCE_ROOT%/}/logging-console-e2e-${RUN_ID}"
WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mesh-logging-console.XXXXXX")"
LOG_DIR="$RUN_DIR/logs"
REQUESTS_DIR="$RUN_DIR/requests"
PLAYWRIGHT_DIR="$RUN_DIR/playwright"
RESULTS_JSONL="$RUN_DIR/results.jsonl"
COMMANDS_JSONL="$RUN_DIR/commands.jsonl"
MANIFEST_JSON="$RUN_DIR/manifest.json"
SUMMARY_JSON="$RUN_DIR/summary.json"
mkdir -p "$LOG_DIR" "$REQUESTS_DIR" "$PLAYWRIGHT_DIR"
: >"$RESULTS_JSONL"
: >"$COMMANDS_JSONL"

API_PORT="$BASE_PORT"
CONSOLE_PORT=$((BASE_PORT + 1))
BIND_PORT=$((BASE_PORT + 2))
HOME_ROOT="$WORK_ROOT/home"
RUNTIME_ROOT="$WORK_ROOT/runtime"
STATE_ROOT="$WORK_ROOT/application-state"
CONFIG_PATH="$WORK_ROOT/logging.toml"
PROCESS_PID=""
PROCESS_START=""
EXIT_STATUS=0

record() {
    local status="$1" name="$2" message="$3"
    shift 3
    python3 - "$RESULTS_JSONL" "$status" "$name" "$message" "$@" <<'PY'
import json
import sys
path, status, name, message, *fields = sys.argv[1:]
value = {"status": status, "name": name, "message": message}
for field in fields:
    key, separator, field_value = field.partition("=")
    if separator:
        value[key] = field_value
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
PY
    [[ "$status" != "FAIL" ]] || EXIT_STATUS=1
}

record_command() {
    local name="$1" path="$2"
    python3 - "$COMMANDS_JSONL" "$name" "$path" <<'PY'
import json
import sys
with open(sys.argv[1], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"name": sys.argv[2], "path": sys.argv[3]}, sort_keys=True) + "\n")
PY
}

process_start_time() { ps -o lstart= -p "$1" 2>/dev/null | awk 'NF { $1=$1; print; exit }'; }

pid_matches_start() {
    [[ -n "${1:-}" && -n "${2:-}" && "$(process_start_time "$1")" == "$2" ]]
}

descendants() {
    local child
    for child in $(pgrep -P "$1" 2>/dev/null || true); do
        descendants "$child"
        printf '%s\n' "$child"
    done
}

stop_process() {
    local pid="$PROCESS_PID" start="$PROCESS_START" child deadline
    [[ -n "$pid" ]] || return 0
    pid_matches_start "$pid" "$start" || { PROCESS_PID=""; PROCESS_START=""; return 0; }
    for child in $(descendants "$pid" | sort -u); do kill -TERM "$child" 2>/dev/null || true; done
    kill -TERM "$pid" 2>/dev/null || true
    deadline=$((SECONDS + 5))
    while pid_matches_start "$pid" "$start" && [[ $SECONDS -lt $deadline ]]; do sleep 1; done
    if pid_matches_start "$pid" "$start"; then
        for child in $(descendants "$pid" | sort -u); do kill -KILL "$child" 2>/dev/null || true; done
        kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    PROCESS_PID=""
    PROCESS_START=""
}

write_summary() {
    python3 - "$RESULTS_JSONL" "$SUMMARY_JSON" "$RUN_DIR" <<'PY'
import json
import sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
statuses = [row["status"] for row in rows]
summary = {
    "overall": "fail" if "FAIL" in statuses else "pass",
    "evidence_dir": sys.argv[3],
    "counts": {status.lower(): statuses.count(status) for status in sorted(set(statuses))},
    "results": rows,
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

cleanup() {
    local incoming=$?
    stop_process
    if [[ -z "$PROCESS_PID" ]]; then
        record PASS cleanup "harness-owned mesh-llm process stopped" "processes=0"
    else
        record FAIL cleanup "harness-owned mesh-llm process remained"
    fi
    write_summary
    if [[ "$EXIT_STATUS" -eq 0 && "$KEEP_STATE" != true ]]; then rm -rf "$WORK_ROOT"; fi
    trap - EXIT
    [[ $incoming -eq 0 ]] || exit "$incoming"
    exit "$EXIT_STATUS"
}
trap cleanup EXIT

python3 - "$MANIFEST_JSON" "$CURRENT_BINARY" "$RUN_DIR" "$WORK_ROOT" "$API_PORT" "$CONSOLE_PORT" <<'PY'
import json
import sys
path, binary, evidence, work, api, console = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"binary": binary, "evidence_dir": evidence, "isolated_work_root": work,
               "api_port": int(api), "console_port": int(console), "logs_api_routes_mocked": False}, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

python3 - "$CONFIG_PATH" "$STATE_ROOT" <<'PY'
import json
import sys
path, state = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    handle.write("\n".join([
        "[logging]", "enabled = true", f"application_state_root = {json.dumps(state)}",
        "retention_max_rows = 64", "replay_capacity = 1", "cleanup_cadence_secs = 300",
        "[logging.artifact]", 'capture_mode = "redacted_artifacts"',
    ]) + "\n")
PY

start_node() {
    local log="$1"
    mkdir -p "$HOME_ROOT" "$RUNTIME_ROOT"
    record_command start-node "$log"
    (
        export HOME="$HOME_ROOT" MESH_LLM_RUNTIME_ROOT="$RUNTIME_ROOT" MESH_LLM_EPHEMERAL_KEY=1
        exec "$CURRENT_BINARY" serve --log-format json --config "$CONFIG_PATH" --port "$API_PORT" --console "$CONSOLE_PORT" --bind-port "$BIND_PORT"
    ) >"$log" 2>&1 &
    PROCESS_PID=$!
    PROCESS_START="$(process_start_time "$PROCESS_PID")"
}

wait_status() {
    local output="$1" second
    for ((second=1; second<=MAX_WAIT; second++)); do
        if curl -fsS --max-time 3 "http://127.0.0.1:$CONSOLE_PORT/api/status" -o "$output"; then return 0; fi
        sleep 1
    done
    return 1
}

submit_rejected() {
    local name="$1"
    local output="$REQUESTS_DIR/$name.json"
    record_command "$name" "$output"
    curl -sS --max-time 10 -o "$output" -w '%{http_code}' \
        -H 'Content-Type: application/json' \
        -d '{"model":"qa-no-model","messages":[{"role":"user","content":"real console QA request"}]}' \
        "http://127.0.0.1:$API_PORT/v1/chat/completions" >"$output.status" || true
}

wait_request_id() {
    local output="$1" second request_id
    for ((second=1; second<=MAX_WAIT; second++)); do
        if curl -fsS --max-time 3 "http://127.0.0.1:$CONSOLE_PORT/api/logs/requests?limit=10" -o "$output"; then
            request_id="$(python3 - "$output" <<'PY'
import json
import sys
try:
    rows = json.load(open(sys.argv[1], encoding="utf-8")).get("items", [])
    value = rows[0].get("requestId") if rows else None
    if isinstance(value, str): print(value)
except (OSError, ValueError):
    pass
PY
            )"
            if [[ -n "$request_id" ]]; then
                printf '%s\n' "$request_id"
                return 0
            fi
        fi
        sleep 1
    done
    return 1
}

start_node "$LOG_DIR/initial.log"
if ! wait_status "$REQUESTS_DIR/status-before-restart.json"; then
    record FAIL real_embedded_console_bundle "real mesh-llm console did not become ready" "log=$LOG_DIR/initial.log"
    exit 1
fi
submit_rejected persisted-request
PERSISTED_REQUEST_ID="$(wait_request_id "$REQUESTS_DIR/list-before-restart.json" || true)"
if [[ -z "$PERSISTED_REQUEST_ID" ]]; then
    record FAIL real_openai_lifecycle_and_detail "zero-model OpenAI request did not become a durable logging record"
    exit 1
fi
record PASS real_openai_lifecycle_and_detail "real OpenAI rejection produced a durable logging DTO" "request_id=$PERSISTED_REQUEST_ID"

stop_process
start_node "$LOG_DIR/restarted.log"
if ! wait_status "$REQUESTS_DIR/status-after-restart.json"; then
    record FAIL restart_persistence "restarted mesh-llm console did not become ready" "log=$LOG_DIR/restarted.log"
    exit 1
fi
if curl -fsS --max-time 5 "http://127.0.0.1:$CONSOLE_PORT/api/logs/requests/$PERSISTED_REQUEST_ID" -o "$REQUESTS_DIR/persisted-detail.json"; then
    record PASS restart_persistence "durable log detail survived restart" "request_id=$PERSISTED_REQUEST_ID"
else
    record FAIL restart_persistence "durable log detail was unavailable after restart" "request_id=$PERSISTED_REQUEST_ID"
fi

host_status="$(curl -sS --max-time 5 -o "$REQUESTS_DIR/hostile-host.json" -w '%{http_code}' -H 'Host: attacker.example' "http://127.0.0.1:$CONSOLE_PORT/api/logs/requests" || true)"
origin_status="$(curl -sS --max-time 5 -o "$REQUESTS_DIR/hostile-origin.json" -w '%{http_code}' -H "Host: localhost:$CONSOLE_PORT" -H 'Origin: https://attacker.example' "http://127.0.0.1:$CONSOLE_PORT/api/logs/requests" || true)"
if [[ "$host_status" == 403 && "$origin_status" == 403 ]]; then
    record PASS trusted_local_rejection "hostile Host and Origin requests were rejected by the real server" "host_status=$host_status" "origin_status=$origin_status"
else
    record FAIL trusted_local_rejection "trusted-local access policy did not reject hostile requests" "host_status=$host_status" "origin_status=$origin_status"
fi

submit_rejected replay-seed-one
submit_rejected replay-seed-two
record NOT_APPLICABLE older_host_unsupported "current binary must expose the logging API; an older host is not started by this isolated real-host lane"

playwright_log="$LOG_DIR/playwright.log"
record_command playwright-real-console "$playwright_log"
if (
    cd "$UI_ROOT"
    MESH_LOGS_E2E=1 \
    MESH_LOGS_E2E_BASE_URL="http://127.0.0.1:$CONSOLE_PORT" \
    MESH_LOGS_E2E_OPENAI_URL="http://127.0.0.1:$API_PORT/v1/chat/completions" \
    MESH_LOGS_E2E_PERSISTED_REQUEST_ID="$PERSISTED_REQUEST_ID" \
    PLAYWRIGHT_OUTPUT_DIR="$PLAYWRIGHT_DIR" \
    PLAYWRIGHT_JSON_REPORT="$PLAYWRIGHT_DIR/report.json" \
    pnpm exec playwright test e2e/logs/real-console.spec.ts --workers=1
) >"$playwright_log" 2>&1; then
    record PASS real_console_playwright "embedded-console Playwright lane completed without /api/logs route interception" "output=$PLAYWRIGHT_DIR"
else
    record FAIL real_console_playwright "embedded-console Playwright lane failed" "log=$playwright_log" "output=$PLAYWRIGHT_DIR"
fi

if curl -sS -N --max-time 5 -D "$REQUESTS_DIR/sse-replay.headers" -o "$REQUESTS_DIR/sse-replay.body" \
    -H 'Accept: text/event-stream' -H 'Last-Event-ID: v1:0.0.0' \
    "http://127.0.0.1:$CONSOLE_PORT/api/logs/events?channel=requests&cursor=v1%3A0.0.0" || true; then :; fi
if rg -q 'event: replay_gap' "$REQUESTS_DIR/sse-replay.body" && rg -q '/api/logs/requests' "$REQUESTS_DIR/sse-replay.body"; then
    record PASS dedicated_sse_replay_gap_and_authoritative_hydration "real dedicated SSE endpoint emitted a replay gap; Playwright asserted the subsequent authoritative ledger hydration" "body=$REQUESTS_DIR/sse-replay.body"
else
    record FAIL dedicated_sse_replay_gap_and_authoritative_hydration "real dedicated SSE endpoint did not emit the expected replay gap" "body=$REQUESTS_DIR/sse-replay.body"
fi
