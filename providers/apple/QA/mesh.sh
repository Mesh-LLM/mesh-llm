#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"
HOST_BINARY="$REPO_ROOT/target/debug/mesh-llm"
OUTPUT_DIR="$REPO_ROOT/target/apple-runtime/mesh"
HOST_LOG="$OUTPUT_DIR/host.jsonl"
HOST_ERR="$OUTPUT_DIR/host.stderr"
HOST_PID=""
PROVIDER_PID=""

[[ -x "$HOST_BINARY" ]] || {
    echo "missing MeshLLM debug product; run just build" >&2
    exit 2
}
[[ -f "$PACKAGE_ROOT/provider-runtime.json" ]] || {
    echo "missing packaged Apple provider; run just apple::package" >&2
    exit 2
}

cleanup() {
    if [[ "$HOST_PID" =~ ^[0-9]+$ ]] && kill -0 "$HOST_PID" 2>/dev/null; then
        kill -TERM "$HOST_PID" 2>/dev/null || true
        wait "$HOST_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

read -r API_PORT CONSOLE_PORT < <(python3 <<'PY'
import socket

ports = []
for _ in range(2):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    ports.append(sock.getsockname()[1])
    sock.close()
print(*ports)
PY
)

MESH_LLM_CONFIG="$OUTPUT_DIR/empty-config.toml" \
MESH_LLM_PROVIDER_RUNTIME_BUNDLE_DIR="$PACKAGE_ROOT" \
MESH_LLM_PROVIDER_RUNTIME_CACHE_DIR="$OUTPUT_DIR/provider-cache" \
MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC=1 \
    "$HOST_BINARY" --log-format json serve \
        --port "$API_PORT" \
        --console "$CONSOLE_PORT" \
        --headless \
        >"$HOST_LOG" 2>"$HOST_ERR" &
HOST_PID=$!

for _ in $(seq 1 300); do
    if curl --silent --show-error "http://127.0.0.1:$API_PORT/v1/models" \
        >"$OUTPUT_DIR/models.json" 2>/dev/null \
        && grep -q 'apple/system' "$OUTPUT_DIR/models.json"; then
        break
    fi
    kill -0 "$HOST_PID" 2>/dev/null || {
        echo "MeshLLM exited before the Apple provider became ready" >&2
        cat "$HOST_ERR" >&2
        exit 1
    }
    sleep 0.1
done

grep -q 'apple/system' "$OUTPUT_DIR/models.json" || {
    echo "apple/system did not become available through MeshLLM" >&2
    cat "$HOST_LOG" >&2
    cat "$HOST_ERR" >&2
    exit 1
}

curl --fail --silent --show-error \
    "http://127.0.0.1:$CONSOLE_PORT/api/runtime/processes" \
    >"$OUTPUT_DIR/processes.json"

PROVIDER_PID="$(python3 - "$OUTPUT_DIR/processes.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
matches = [
    process for process in payload["processes"]
    if process.get("name") == "apple/system"
    and process.get("backend") == "apple"
    and process.get("status") == "ready"
]
assert len(matches) == 1, payload
print(matches[0]["pid"])
PY
)"
[[ "$PROVIDER_PID" =~ ^[0-9]+$ ]] || {
    echo "management API did not report the Apple provider pid" >&2
    exit 1
}
INITIAL_PROVIDER_PID="$PROVIDER_PID"

kill -TERM "$INITIAL_PROVIDER_PID"
RESTARTED_PROVIDER_PID=""
for _ in $(seq 1 300); do
    if curl --silent --show-error \
        "http://127.0.0.1:$CONSOLE_PORT/api/runtime/processes" \
        >"$OUTPUT_DIR/processes-after-restart.json" 2>/dev/null; then
        RESTARTED_PROVIDER_PID="$(python3 - "$OUTPUT_DIR/processes-after-restart.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
matches = [
    process for process in payload.get("processes", [])
    if process.get("name") == "apple/system"
    and process.get("backend") == "apple"
    and process.get("status") == "ready"
]
print(matches[0]["pid"] if len(matches) == 1 else "")
PY
)"
        if [[ "$RESTARTED_PROVIDER_PID" =~ ^[0-9]+$ ]] \
            && [[ "$RESTARTED_PROVIDER_PID" != "$INITIAL_PROVIDER_PID" ]]; then
            PROVIDER_PID="$RESTARTED_PROVIDER_PID"
            break
        fi
    fi
    kill -0 "$HOST_PID" 2>/dev/null || {
        echo "MeshLLM exited while restarting the Apple provider" >&2
        exit 1
    }
    sleep 0.1
done
[[ "$PROVIDER_PID" != "$INITIAL_PROVIDER_PID" ]] || {
    echo "Apple provider did not restart with a new pid" >&2
    cat "$HOST_LOG" >&2
    exit 1
}

lsof -nP -a -p "$HOST_PID" -iTCP -sTCP:LISTEN >"$OUTPUT_DIR/host-listeners.txt"
grep -q "127.0.0.1:$API_PORT" "$OUTPUT_DIR/host-listeners.txt" || {
    echo "MeshLLM OpenAI listener is not restricted to loopback" >&2
    exit 1
}

MESH_APPLE_RUNTIME_BASE_URL="http://127.0.0.1:$API_PORT" \
MESH_APPLE_RUNTIME_REST_OUTPUT_DIR="$OUTPUT_DIR/rest" \
    "$APPLE_ROOT/QA/rest.sh"

kill -TERM "$HOST_PID"
wait "$HOST_PID"
HOST_PID=""

for _ in $(seq 1 100); do
    kill -0 "$PROVIDER_PID" 2>/dev/null || break
    sleep 0.05
done
if kill -0 "$PROVIDER_PID" 2>/dev/null; then
    echo "Apple provider remained alive after MeshLLM shutdown" >&2
    exit 1
fi

python3 - "$OUTPUT_DIR" "$API_PORT" "$CONSOLE_PORT" "$PROVIDER_PID" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rest = json.loads((root / "rest" / "summary.json").read_text())
summary = {
    "status": "pass",
    "model": "apple/system",
    "versioned_model": rest["versioned_model"],
    "mesh_api_port": int(sys.argv[2]),
    "management_port": int(sys.argv[3]),
    "provider_pid": int(sys.argv[4]),
    "provider_reported_in_management_api": True,
    "provider_restarted_after_crash": True,
    "provider_exited_with_meshllm": True,
    "completion": rest["completion"],
    "tool": rest["tool"],
    "stream_done": rest["stream_done"],
    "client_disconnect_cancelled": rest["client_disconnect_cancelled"],
    "slot_released_after_cancel": rest["slot_released_after_cancel"],
    "typed_model_error": rest["typed_model_error"],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "status": summary["status"],
    "model": summary["model"],
    "versioned_model": summary["versioned_model"],
    "completion_content": summary["completion"]["choices"][0]["message"]["content"],
    "tool_executions": summary["tool"]["mesh_tool_executions"],
    "stream_done": summary["stream_done"],
    "client_disconnect_cancelled": summary["client_disconnect_cancelled"],
    "provider_reported_in_management_api": summary["provider_reported_in_management_api"],
    "provider_restarted_after_crash": summary["provider_restarted_after_crash"],
    "provider_exited_with_meshllm": summary["provider_exited_with_meshllm"],
}, indent=2, sort_keys=True))
PY
