#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"
HOST_BINARY="${MESH_APPLE_RUNTIME_MESH_HOST_BINARY:-$REPO_ROOT/target/debug/mesh-llm}"
OUTPUT_DIR="${MESH_APPLE_RUNTIME_PRIVATE_MESH_OUTPUT_DIR:-$REPO_ROOT/target/apple-runtime/private-mesh}"
APPLE_MODEL_ID="$(python3 - "$PACKAGE_ROOT/provider-runtime.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    models = json.load(handle)["runtime"]["models"]
if len(models) != 1:
    raise SystemExit("Apple private-mesh QA requires exactly one packaged provider model")
print(models[0]["id"])
PY
)"
export APPLE_MODEL_ID
NODE_A_PID=""
NODE_B_PID=""
LOAD_A_PID=""
LOAD_B_PID=""

cleanup() {
    for pid in "$LOAD_B_PID" "$LOAD_A_PID"; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$NODE_B_PID" "$NODE_A_PID"; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

[[ -x "$HOST_BINARY" ]] || {
    echo "missing MeshLLM debug product; run just build" >&2
    exit 2
}
[[ -f "$PACKAGE_ROOT/provider-runtime.json" ]] || {
    echo "missing packaged Apple provider; run just apple::package" >&2
    exit 2
}

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/node-a-home" "$OUTPUT_DIR/node-b-home"

read -r API_A CONSOLE_A API_B CONSOLE_B < <(python3 <<'PY'
import socket

ports = []
for _ in range(4):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    ports.append(sock.getsockname()[1])
    sock.close()
print(*ports)
PY
)

env \
    -u MESH_LLM_PROVIDER_RUNTIME_INDEX \
    HOME="$OUTPUT_DIR/node-a-home" \
    MESH_LLM_CONFIG="$OUTPUT_DIR/node-a-config.toml" \
    MESH_LLM_PROVIDER_RUNTIME_BUNDLE_DIR="$PACKAGE_ROOT" \
    MESH_LLM_PROVIDER_RUNTIME_CACHE_DIR="$OUTPUT_DIR/node-a-provider-cache" \
    MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC=1 \
    "$HOST_BINARY" --log-format json serve --headless \
        --port "$API_A" --console "$CONSOLE_A" \
        >"$OUTPUT_DIR/node-a.jsonl" 2>"$OUTPUT_DIR/node-a.stderr" &
NODE_A_PID=$!

INVITE_TOKEN=""
for _ in $(seq 1 300); do
    INVITE_TOKEN="$(python3 - "$OUTPUT_DIR/node-a.jsonl" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
for line in path.read_text().splitlines() if path.exists() else []:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if event.get("event") == "invite_token":
        print(event.get("token", ""))
        break
PY
)"
    if [[ -n "$INVITE_TOKEN" ]] \
        && curl --silent --show-error "http://127.0.0.1:$API_A/v1/models" \
            >"$OUTPUT_DIR/node-a-models.json" 2>/dev/null \
        && grep -q "$APPLE_MODEL_ID" "$OUTPUT_DIR/node-a-models.json"; then
        break
    fi
    kill -0 "$NODE_A_PID" 2>/dev/null || {
        cat "$OUTPUT_DIR/node-a.stderr" >&2
        exit 1
    }
    sleep 0.1
done
[[ -n "$INVITE_TOKEN" ]] || {
    echo "node A did not emit an invite token" >&2
    exit 1
}

env \
    -u MESH_LLM_PROVIDER_RUNTIME_INDEX \
    HOME="$OUTPUT_DIR/node-b-home" \
    MESH_LLM_CONFIG="$OUTPUT_DIR/node-b-config.toml" \
    MESH_LLM_PROVIDER_RUNTIME_BUNDLE_DIR="$PACKAGE_ROOT" \
    MESH_LLM_PROVIDER_RUNTIME_CACHE_DIR="$OUTPUT_DIR/node-b-provider-cache" \
    MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC=1 \
    "$HOST_BINARY" --log-format json serve --headless \
        --port "$API_B" --console "$CONSOLE_B" --join "$INVITE_TOKEN" \
        >"$OUTPUT_DIR/node-b.jsonl" 2>"$OUTPUT_DIR/node-b.stderr" &
NODE_B_PID=$!

for _ in $(seq 1 400); do
    if curl --silent --show-error "http://127.0.0.1:$CONSOLE_A/api/status" \
        >"$OUTPUT_DIR/node-a-status.json" 2>/dev/null \
        && curl --silent --show-error "http://127.0.0.1:$CONSOLE_B/api/status" \
            >"$OUTPUT_DIR/node-b-status.json" 2>/dev/null \
        && curl --silent --show-error "http://127.0.0.1:$API_A/v1/models" \
            >"$OUTPUT_DIR/mesh-models.json" 2>/dev/null \
        && python3 - "$OUTPUT_DIR" 2>/dev/null <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
status_a = json.loads((root / "node-a-status.json").read_text())
status_b = json.loads((root / "node-b-status.json").read_text())
models = json.loads((root / "mesh-models.json").read_text())

def peer_has_apple(status):
    model_id = __import__("os").environ["APPLE_MODEL_ID"]
    return any(
        runtime.get("model_name") == model_id
        and runtime.get("provider_kind") == "apple"
        and runtime.get("max_concurrent_requests") == 1
        for peer in status.get("peers", [])
        for runtime in peer.get("provider_runtimes", [])
    )

model_id = __import__("os").environ["APPLE_MODEL_ID"]
apple = next((model for model in models.get("data", []) if model.get("id") == model_id), None)
metadata = apple.get("metadata", {}) if apple else {}
assert peer_has_apple(status_a)
assert peer_has_apple(status_b)
assert metadata.get("provider") == "apple"
assert metadata.get("replicas") == 2
assert metadata.get("max_concurrent_requests") == 2
PY
    then
        break
    fi
    kill -0 "$NODE_A_PID" 2>/dev/null || exit 1
    kill -0 "$NODE_B_PID" 2>/dev/null || {
        cat "$OUTPUT_DIR/node-b.stderr" >&2
        exit 1
    }
    sleep 0.1
done

python3 - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
models = json.loads((root / "mesh-models.json").read_text())
model_id = __import__("os").environ["APPLE_MODEL_ID"]
apple = next(model for model in models["data"] if model["id"] == model_id)
assert apple["metadata"]["replicas"] == 2, apple
PY

curl --fail --silent --show-error "http://127.0.0.1:$API_A/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$APPLE_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: private mesh ready\"}],\"max_tokens\":32}" \
    >"$OUTPUT_DIR/completion.json"

curl --fail --silent --show-error "http://127.0.0.1:$API_A/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$APPLE_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Write exactly 100 numbered lines, with four words on each line.\"}],\"max_tokens\":256}" \
    >"$OUTPUT_DIR/load-a-completion.json" &
LOAD_A_PID=$!

LOAD_A_OBSERVED=0
for _ in $(seq 1 200); do
    if curl --silent --show-error "http://127.0.0.1:$CONSOLE_A/api/status" \
        >"$OUTPUT_DIR/node-a-active.json" 2>/dev/null \
        && python3 - "$OUTPUT_DIR/node-a-active.json" 2>/dev/null <<'PY'
import json
import pathlib
import sys

status = json.loads(pathlib.Path(sys.argv[1]).read_text())
model_id = __import__("os").environ["APPLE_MODEL_ID"]
assert any(
    model.get("name") == model_id and model.get("active_requests") == 1
    for model in status.get("runtime", {}).get("models", [])
)
PY
    then
        LOAD_A_OBSERVED=1
        break
    fi
    kill -0 "$LOAD_A_PID" 2>/dev/null || {
        echo "first load probe completed before its active slot was observable" >&2
        exit 1
    }
    sleep 0.05
done
[[ "$LOAD_A_OBSERVED" == "1" ]] || {
    echo "first provider active slot was not observable" >&2
    exit 1
}

curl --fail --silent --show-error "http://127.0.0.1:$API_A/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$APPLE_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Write exactly 100 numbered lines, with five words on each line.\"}],\"max_tokens\":256}" \
    >"$OUTPUT_DIR/load-b-completion.json" &
LOAD_B_PID=$!

LOAD_B_OBSERVED=0
for _ in $(seq 1 200); do
    if curl --silent --show-error "http://127.0.0.1:$CONSOLE_B/api/status" \
        >"$OUTPUT_DIR/node-b-active.json" 2>/dev/null \
        && python3 - "$OUTPUT_DIR/node-b-active.json" 2>/dev/null <<'PY'
import json
import pathlib
import sys

status = json.loads(pathlib.Path(sys.argv[1]).read_text())
model_id = __import__("os").environ["APPLE_MODEL_ID"]
assert any(
    model.get("name") == model_id and model.get("active_requests") == 1
    for model in status.get("runtime", {}).get("models", [])
)
PY
    then
        LOAD_B_OBSERVED=1
        break
    fi
    kill -0 "$LOAD_B_PID" 2>/dev/null || {
        echo "second load probe did not route to the idle peer" >&2
        exit 1
    }
    sleep 0.05
done
[[ "$LOAD_B_OBSERVED" == "1" ]] || {
    echo "second request did not use the idle peer" >&2
    exit 1
}

wait "$LOAD_A_PID"
LOAD_A_PID=""
wait "$LOAD_B_PID"
LOAD_B_PID=""

PROVIDER_PORT_A="$(python3 - "$OUTPUT_DIR/node-a-status.json" <<'PY'
import json
import os
import sys

status = json.load(open(sys.argv[1], encoding="utf-8"))
model_id = os.environ["APPLE_MODEL_ID"]
for model in status.get("runtime", {}).get("models", []):
    if model.get("name") == model_id and model.get("port"):
        print(model["port"])
        break
else:
    raise SystemExit("provider port was not present in node A status")
PY
)"
PROVIDER_PID_A="$(lsof -ti "tcp:$PROVIDER_PORT_A" | head -n 1)"
[[ "$PROVIDER_PID_A" =~ ^[0-9]+$ ]] || {
    echo "could not identify node A Apple provider process" >&2
    exit 1
}
kill -KILL "$PROVIDER_PID_A"

WITHDRAW_OBSERVED=0
for _ in $(seq 1 200); do
    if curl --silent --show-error "http://127.0.0.1:$CONSOLE_A/api/runtime/processes" \
        >"$OUTPUT_DIR/node-a-processes-after-withdraw.json" 2>/dev/null \
        && curl --silent --show-error "http://127.0.0.1:$CONSOLE_B/api/status" \
        >"$OUTPUT_DIR/node-b-withdrawn-status.json" 2>/dev/null \
        && curl --silent --show-error "http://127.0.0.1:$API_A/v1/models" \
        >"$OUTPUT_DIR/mesh-models-after-withdraw.json" 2>/dev/null \
        && python3 - "$OUTPUT_DIR" 2>/dev/null <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model_id = os.environ["APPLE_MODEL_ID"]
processes = json.loads((root / "node-a-processes-after-withdraw.json").read_text())
assert not any(
    process.get("name") == model_id and process.get("backend") == "apple"
    for process in processes.get("processes", [])
), processes
PY
    then
        WITHDRAW_OBSERVED=1
        break
    fi
    sleep 0.1
done

[[ "$WITHDRAW_OBSERVED" == "1" ]] || {
    echo "node A provider withdrawal was not observed on the private mesh" >&2
    exit 1
}

curl --fail --silent --show-error "http://127.0.0.1:$API_A/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$APPLE_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: failover ready\"}],\"max_tokens\":32}" \
    >"$OUTPUT_DIR/failover-completion.json"

python3 - "$OUTPUT_DIR" <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
models = json.loads((root / "mesh-models.json").read_text())
completion = json.loads((root / "completion.json").read_text())
failover = json.loads((root / "failover-completion.json").read_text())
model_id = os.environ["APPLE_MODEL_ID"]
apple = next(model for model in models["data"] if model["id"] == model_id)
summary = {
    "status": "pass",
    "model": model_id,
    "replicas": apple["metadata"]["replicas"],
    "max_concurrent_requests": apple["metadata"]["max_concurrent_requests"],
    "provider_model_versions": apple["metadata"]["provider_model_versions"],
    "completion_content": completion["choices"][0]["message"]["content"],
    "private_peer_provider_runtime_visible": True,
    "load_aware_remote_dispatch": True,
    "provider_withdrawal_and_failover": True,
    "failover_completion_content": failover["choices"][0]["message"]["content"],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
