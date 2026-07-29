#!/usr/bin/env bash
# Start a composed MeshLLM product in noninteractive client mode and require a
# JSON readiness event followed by a bounded SIGINT shutdown.

set -euo pipefail

MESH_LLM="${1:?usage: $0 <mesh-llm-binary> <native-runtime-root>}"
RUNTIME_ROOT="${2:?usage: $0 <mesh-llm-binary> <native-runtime-root>}"
MAX_WAIT="${MESH_LLM_CLIENT_READY_MAX_WAIT:-60}"
LOG="$(mktemp "${MESH_LLM_CLIENT_STATE_PARENT:-/tmp}/mlc-ready.XXXXXX.log")"
STATE_DIR="$(mktemp -d "${MESH_LLM_CLIENT_STATE_PARENT:-/tmp}/mlc-state.XXXXXX")"

[[ -x "$MESH_LLM" ]] || { echo "missing executable: $MESH_LLM" >&2; exit 2; }
[[ -d "$RUNTIME_ROOT" ]] || { echo "missing native runtime root: $RUNTIME_ROOT" >&2; exit 2; }
[[ "$MAX_WAIT" =~ ^[1-9][0-9]*$ ]] || { echo "MESH_LLM_CLIENT_READY_MAX_WAIT must be a positive integer" >&2; exit 2; }
mkdir -p "$STATE_DIR/home" "$STATE_DIR/cache" "$STATE_DIR/config" "$STATE_DIR/xdg-runtime" "$STATE_DIR/runtime-cache" "$STATE_DIR/runtime"
chmod 700 "$STATE_DIR" "$STATE_DIR/home" "$STATE_DIR/cache" "$STATE_DIR/config" "$STATE_DIR/xdg-runtime" "$STATE_DIR/runtime-cache" "$STATE_DIR/runtime"

port="$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)"

pid=""
# shellcheck disable=SC2329 # Invoked by the EXIT trap.
cleanup() {
    local cleanup_status=0
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "client did not stop cleanly after SIGINT" >&2
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            cleanup_status=1
        fi
        set +e
        wait "$pid"
        status=$?
        set -e
        if [[ "$status" -ne 0 ]]; then
            echo "client exited non-cleanly after SIGINT: $status" >&2
            cleanup_status=1
        fi
    fi
    rm -rf "$STATE_DIR"
    rm -f "$LOG"
    return "$cleanup_status"
}
trap cleanup EXIT

MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$RUNTIME_ROOT" \
MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$STATE_DIR/runtime-cache" \
MESH_LLM_CONFIG="$STATE_DIR/config.toml" \
MESH_LLM_RUNTIME_ROOT="$STATE_DIR/runtime" \
HOME="$STATE_DIR/home" \
XDG_CACHE_HOME="$STATE_DIR/cache" \
XDG_CONFIG_HOME="$STATE_DIR/config" \
XDG_RUNTIME_DIR="$STATE_DIR/xdg-runtime" \
    "$MESH_LLM" --log-format json --port "$port" --no-console client --auto >"$LOG" 2>&1 &
pid=$!

for _ in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$pid" 2>/dev/null; then
        cat "$LOG" >&2
        echo "client exited before readiness" >&2
        exit 1
    fi
    if python3 - "$LOG" <<'PY'
import json
import sys

for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    message = str(event.get("message", "")).lower()
    structured_ready = (
        event.get("event") == "passive_mode"
        and event.get("status") == "ready"
        and event.get("role") == "client"
    )
    if "client ready" in message or structured_ready:
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "client readiness observed on port $port"
        kill -INT "$pid"
        for _ in $(seq 1 15); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            cat "$LOG" >&2
            echo "client did not stop cleanly after SIGINT" >&2
            exit 1
        fi
        set +e
        wait "$pid"
        status=$?
        set -e
        if [[ "$status" -ne 0 ]]; then
            cat "$LOG" >&2
            echo "client exited non-cleanly after SIGINT: $status" >&2
            exit 1
        fi
        pid=""
        exit 0
    fi
    sleep 1
done

cat "$LOG" >&2
echo "timed out waiting for structured client readiness" >&2
exit 1
