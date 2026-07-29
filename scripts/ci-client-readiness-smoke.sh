#!/usr/bin/env bash
# Start a composed MeshLLM product in hermetic noninteractive client mode and
# require a JSON readiness event followed by a bounded platform-appropriate
# graceful shutdown. Public auto-discovery is intentionally tested elsewhere;
# product composition must not depend on a mutable external mesh.

set -euo pipefail

MESH_LLM="${1:?usage: $0 <mesh-llm-binary> <native-runtime-root>}"
RUNTIME_ROOT="${2:?usage: $0 <mesh-llm-binary> <native-runtime-root>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WINDOWS_PROCESS_HELPER="$SCRIPT_DIR/ci-client-readiness-process.py"
MAX_WAIT="${MESH_LLM_CLIENT_READY_MAX_WAIT:-60}"
SHUTDOWN_MAX_WAIT="${MESH_LLM_CLIENT_SHUTDOWN_MAX_WAIT:-15}"
LOG="$(mktemp "${MESH_LLM_CLIENT_STATE_PARENT:-/tmp}/mlc-ready.XXXXXX.log")"
STATE_DIR="$(mktemp -d "${MESH_LLM_CLIENT_STATE_PARENT:-/tmp}/mlc-state.XXXXXX")"

[[ -x "$MESH_LLM" ]] || { echo "missing executable: $MESH_LLM" >&2; exit 2; }
[[ -d "$RUNTIME_ROOT" ]] || { echo "missing native runtime root: $RUNTIME_ROOT" >&2; exit 2; }
[[ "$MAX_WAIT" =~ ^[1-9][0-9]*$ ]] || { echo "MESH_LLM_CLIENT_READY_MAX_WAIT must be a positive integer" >&2; exit 2; }
[[ "$SHUTDOWN_MAX_WAIT" =~ ^[1-9][0-9]*$ ]] || { echo "MESH_LLM_CLIENT_SHUTDOWN_MAX_WAIT must be a positive integer" >&2; exit 2; }
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
native_pid_file="$STATE_DIR/native-client.pid"
is_windows=0
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) is_windows=1 ;;
esac

# shellcheck disable=SC2329 # Invoked by the EXIT cleanup trap.
shutdown_client_unix() {
    local child_status=0
    local deadline_pid
    local shutdown_done="$STATE_DIR/shutdown-done"
    local shutdown_timed_out="$STATE_DIR/shutdown-timed-out"

    rm -f "$shutdown_done" "$shutdown_timed_out"
    kill -INT "$pid" 2>/dev/null || true
    (
        for ((attempt = 0; attempt < SHUTDOWN_MAX_WAIT; attempt++)); do
            sleep 1
            [[ ! -e "$shutdown_done" ]] || exit 0
        done
        if [[ ! -e "$shutdown_done" ]] && kill -0 "$pid" 2>/dev/null; then
            : >"$shutdown_timed_out"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            [[ -e "$shutdown_done" ]] || kill -KILL "$pid" 2>/dev/null || true
        fi
    ) </dev/null >/dev/null 2>&1 &
    deadline_pid=$!

    if wait "$pid"; then
        child_status=0
    else
        child_status=$?
    fi
    pid=""
    : >"$shutdown_done"
    wait "$deadline_pid" 2>/dev/null || true

    if [[ -e "$shutdown_timed_out" ]]; then
        echo "client did not stop cleanly after SIGINT within ${SHUTDOWN_MAX_WAIT}s" >&2
        return 1
    fi
    if [[ "$child_status" -ne 0 ]]; then
        echo "client exited non-cleanly after SIGINT: $child_status" >&2
        return 1
    fi
    return 0
}

# shellcheck disable=SC2329 # Invoked by the EXIT cleanup trap.
shutdown_client_windows() {
    local child_status=0
    local native_pid=""
    local shutdown_timed_out="$STATE_DIR/shutdown-timed-out"

    rm -f "$shutdown_timed_out"
    if [[ -f "$native_pid_file" ]]; then
        native_pid="$(<"$native_pid_file")"
        python3 "$WINDOWS_PROCESS_HELPER" ctrl-break --pid "$native_pid" 2>/dev/null || true
    fi

    for ((attempt = 0; attempt < SHUTDOWN_MAX_WAIT; attempt++)); do
        if [[ -n "$native_pid" ]] && ! python3 "$WINDOWS_PROCESS_HELPER" is-running --pid "$native_pid"; then
            break
        fi
        sleep 1
    done

    if [[ -n "$native_pid" ]] && python3 "$WINDOWS_PROCESS_HELPER" is-running --pid "$native_pid"; then
        : >"$shutdown_timed_out"
        python3 "$WINDOWS_PROCESS_HELPER" force-stop --pid "$native_pid" || true
    fi

    if wait "$pid"; then
        child_status=0
    else
        child_status=$?
    fi
    pid=""

    if [[ -e "$shutdown_timed_out" ]]; then
        echo "client did not stop cleanly after CTRL_BREAK_EVENT within ${SHUTDOWN_MAX_WAIT}s" >&2
        return 1
    fi
    if [[ "$child_status" -ne 0 ]]; then
        echo "client exited non-cleanly after CTRL_BREAK_EVENT: $child_status" >&2
        return 1
    fi
    return 0
}

# shellcheck disable=SC2329 # Invoked by the EXIT cleanup trap.
shutdown_client() {
    if [[ "$is_windows" == "1" ]]; then
        shutdown_client_windows
    else
        shutdown_client_unix
    fi
}

# shellcheck disable=SC2329 # Invoked by the EXIT trap.
cleanup() {
    local original_status=$?
    local cleanup_status=0
    trap - EXIT
    set +e

    if [[ -n "$pid" ]] && ! shutdown_client; then
        cleanup_status=1
        cat "$LOG" >&2
    fi
    rm -rf "$STATE_DIR" || cleanup_status=1
    rm -f "$LOG" || cleanup_status=1

    if [[ "$cleanup_status" -ne 0 ]]; then
        exit "$cleanup_status"
    fi
    exit "$original_status"
}
trap cleanup EXIT

if [[ "$is_windows" == "1" ]]; then
    MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$RUNTIME_ROOT" \
    MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$STATE_DIR/runtime-cache" \
    MESH_LLM_CONFIG="$STATE_DIR/config.toml" \
    MESH_LLM_RUNTIME_ROOT="$STATE_DIR/runtime" \
    HOME="$STATE_DIR/home" \
    XDG_CACHE_HOME="$STATE_DIR/cache" \
    XDG_CONFIG_HOME="$STATE_DIR/config" \
    XDG_RUNTIME_DIR="$STATE_DIR/xdg-runtime" \
        python3 "$WINDOWS_PROCESS_HELPER" run --pid-file "$native_pid_file" --log "$LOG" -- \
        "$MESH_LLM" --log-format json --port "$port" --no-console \
        client --mesh-discovery-mode mdns &
else
    MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$RUNTIME_ROOT" \
    MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$STATE_DIR/runtime-cache" \
    MESH_LLM_CONFIG="$STATE_DIR/config.toml" \
    MESH_LLM_RUNTIME_ROOT="$STATE_DIR/runtime" \
    HOME="$STATE_DIR/home" \
    XDG_CACHE_HOME="$STATE_DIR/cache" \
    XDG_CONFIG_HOME="$STATE_DIR/config" \
    XDG_RUNTIME_DIR="$STATE_DIR/xdg-runtime" \
        "$MESH_LLM" --log-format json --port "$port" --no-console \
        client --mesh-discovery-mode mdns >"$LOG" 2>&1 &
fi
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
        exit 0
    fi
    sleep 1
done

cat "$LOG" >&2
echo "timed out waiting for hermetic structured client readiness" >&2
exit 1
