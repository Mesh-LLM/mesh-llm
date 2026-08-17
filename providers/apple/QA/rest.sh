#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"
BINARY="$PACKAGE_ROOT/bin/mesh-apple-runtime"
OUTPUT_DIR="$REPO_ROOT/target/apple-runtime/rest"
SERVER_LOG="$OUTPUT_DIR/server.jsonl"
SERVER_ERR="$OUTPUT_DIR/server.stderr"
SERVER_PID=""

[[ -x "$BINARY" ]] || {
    echo "missing packaged Apple runtime; run just apple::package" >&2
    exit 2
}

cleanup() {
    if [[ "$SERVER_PID" =~ ^[0-9]+$ ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

"$BINARY" serve --port 0 --parent-pid "$$" >"$SERVER_LOG" 2>"$SERVER_ERR" &
SERVER_PID=$!

READY=0
for _ in $(seq 1 200); do
    if grep -q '"type":"ready"' "$SERVER_LOG" 2>/dev/null; then
        READY=1
        break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || {
        echo "Apple runtime exited before REST became ready" >&2
        cat "$SERVER_ERR" >&2
        exit 1
    }
    sleep 0.05
done
[[ "$READY" == "1" ]] || {
    echo "Apple runtime REST did not become ready within 10s" >&2
    cat "$SERVER_ERR" >&2
    exit 1
}

PORT="$(python3 - "$SERVER_LOG" <<'PY'
import json
import pathlib
import sys

for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    event = json.loads(line)
    if event.get("type") == "ready":
        print(event["port"])
        break
PY
)"
[[ "$PORT" =~ ^[0-9]+$ ]] || {
    echo "invalid REST port: $PORT" >&2
    exit 1
}
BASE_URL="http://127.0.0.1:$PORT"
lsof -nP -a -p "$SERVER_PID" -iTCP -sTCP:LISTEN >"$OUTPUT_DIR/listeners.txt"
grep -q "127.0.0.1:$PORT" "$OUTPUT_DIR/listeners.txt" || {
    echo "Apple runtime REST listener is not restricted to IPv4 loopback" >&2
    cat "$OUTPUT_DIR/listeners.txt" >&2
    exit 1
}
if awk 'NR > 1 {print}' "$OUTPUT_DIR/listeners.txt" | grep -v "127\.0\.0\.1:$PORT" | grep -q 'LISTEN'; then
    echo "Apple runtime REST exposes a non-loopback listener" >&2
    cat "$OUTPUT_DIR/listeners.txt" >&2
    exit 1
fi

curl --fail --silent --show-error "$BASE_URL/v1/models" >"$OUTPUT_DIR/models.json"

curl --fail --silent --show-error \
    -H 'content-type: application/json' \
    --data-binary '{"model":"apple/system","messages":[{"role":"user","content":"Reply with exactly: apple runtime REST ready"}],"temperature":0,"max_tokens":32}' \
    "$BASE_URL/v1/chat/completions" >"$OUTPUT_DIR/completion.json"

curl --fail --silent --show-error --no-buffer \
    -H 'content-type: application/json' \
    --data-binary '{"model":"apple/system","messages":[{"role":"user","content":"Reply with exactly: streaming REST ready"}],"temperature":0,"max_tokens":32,"stream":true}' \
    "$BASE_URL/v1/chat/completions" >"$OUTPUT_DIR/stream.txt"

curl --fail --silent --show-error \
    -H 'content-type: application/json' \
    --data-binary '{"model":"apple/system","messages":[{"role":"user","content":"Use the tool with key: rest-demo"}],"tools":[{"type":"function","function":{"name":"mesh_fixture_lookup","description":"Look up a fixture","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}}],"temperature":0}' \
    "$BASE_URL/v1/chat/completions" >"$OUTPUT_DIR/tool.json"

set +e
curl --silent --show-error --max-time 0.1 \
    -H 'content-type: application/json' \
    --data-binary '{"model":"apple/system","messages":[{"role":"user","content":"Write a very long history of distributed computing."}],"max_tokens":512,"stream":true}' \
    "$BASE_URL/v1/chat/completions" >"$OUTPUT_DIR/cancelled-stream.txt" 2>"$OUTPUT_DIR/cancelled-stream.stderr"
CANCEL_STATUS=$?
set -e
[[ "$CANCEL_STATUS" -ne 0 ]] || {
    echo "REST cancellation probe completed before the client disconnected" >&2
    exit 1
}
grep -q 'chat.completion.chunk' "$OUTPUT_DIR/cancelled-stream.txt" || {
    echo "REST cancellation probe disconnected before streaming began" >&2
    cat "$OUTPUT_DIR/cancelled-stream.stderr" >&2 || true
    exit 1
}

curl --fail --silent --show-error \
    -H 'content-type: application/json' \
    --data-binary '{"model":"apple/system","messages":[{"role":"user","content":"Reply with exactly: slot released"}],"temperature":0,"max_tokens":16}' \
    "$BASE_URL/v1/chat/completions" >"$OUTPUT_DIR/after-cancel.json"

ERROR_STATUS="$(curl --silent --show-error \
    --output "$OUTPUT_DIR/model-not-found.json" \
    --write-out '%{http_code}' \
    -H 'content-type: application/json' \
    --data-binary '{"model":"apple/not-present","messages":[{"role":"user","content":"hello"}]}' \
    "$BASE_URL/v1/chat/completions")"
[[ "$ERROR_STATUS" == "404" ]] || {
    echo "expected model-not-found HTTP 404, got $ERROR_STATUS" >&2
    exit 1
}

python3 - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
models = json.loads((root / "models.json").read_text())
completion = json.loads((root / "completion.json").read_text())
tool = json.loads((root / "tool.json").read_text())
after_cancel = json.loads((root / "after-cancel.json").read_text())
model_not_found = json.loads((root / "model-not-found.json").read_text())
stream = (root / "stream.txt").read_text()

assert any(model["id"] == "apple/system" for model in models["data"]), models
assert completion["model"] == "apple/system", completion
assert completion["choices"][0]["message"]["content"], completion
assert completion["usage"]["completion_tokens"] > 0, completion
assert "data: [DONE]" in stream, stream
assert "chat.completion.chunk" in stream, stream
assert tool["mesh_tool_executions"] == [{
    "name": "mesh_fixture_lookup",
    "arguments": {"key": "rest-demo"},
    "result": "mesh-fixture-value-for-rest-demo",
}], tool
assert "mesh-fixture-value-for-rest-demo" in tool["choices"][0]["message"]["content"], tool
assert after_cancel["choices"][0]["message"]["content"], after_cancel
assert model_not_found["error"]["code"] == "model_not_found", model_not_found

summary = {
    "status": "pass",
    "model": "apple/system",
    "completion": completion,
    "tool": tool,
    "stream_done": True,
    "loopback_only": True,
    "client_disconnect_cancelled": True,
    "slot_released_after_cancel": True,
    "typed_model_error": True,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "status": summary["status"],
    "model": summary["model"],
    "completion_content": completion["choices"][0]["message"]["content"],
    "tool_executions": tool["mesh_tool_executions"],
    "stream_done": summary["stream_done"],
    "loopback_only": summary["loopback_only"],
    "client_disconnect_cancelled": summary["client_disconnect_cancelled"],
    "slot_released_after_cancel": summary["slot_released_after_cancel"],
    "typed_model_error": summary["typed_model_error"],
}, indent=2, sort_keys=True))
PY
