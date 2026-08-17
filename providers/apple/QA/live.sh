#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_PATH="$APPLE_ROOT"
OUTPUT_DIR="$REPO_ROOT/target/apple-runtime/live"
BIN_DIR="$(swift build --show-bin-path --package-path "$PACKAGE_PATH")"
BINARY="$BIN_DIR/mesh-apple-runtime"

[[ -x "$BINARY" ]] || {
    echo "missing Apple runtime executable; run just apple::build" >&2
    exit 2
}

mkdir -p "$OUTPUT_DIR"

run_probe() {
    local name="$1"
    shift
    echo "==> $name"
    "$BINARY" "$@" >"$OUTPUT_DIR/$name.jsonl" 2>&1 &
    local probe_pid=$!
    (
        sleep "${MESH_APPLE_QA_PROBE_TIMEOUT:-120}"
        kill -TERM "$probe_pid" 2>/dev/null || true
    ) &
    local killer_pid=$!
    local probe_status=0
    wait "$probe_pid" || probe_status=$?
    kill "$killer_pid" 2>/dev/null || true
    cat "$OUTPUT_DIR/$name.jsonl"
    if [[ "$probe_status" -ne 0 ]]; then
        echo "Apple runtime probe '$name' failed with status $probe_status" >&2
        return "$probe_status"
    fi
}

run_probe status status
run_probe prewarm prewarm --prefix "MeshLLM routes whole requests"
run_probe generate generate \
    --prompt "In one short sentence, explain why whole-model routing avoids per-layer network hops." \
    --max-tokens 64 \
    --temperature 0
run_probe structured structured \
    --prompt "Classify this request as routing, storage, or unrelated: choose a warm mesh replica."
run_probe tool tool --key "milestone-zero"
run_probe cancel cancel --after-ms 25 --max-tokens 512

python3 - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])

def events(name):
    return [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines() if line]

status = events("status")[0]
assert status["runtimeID"] == "apple/runtime", status
system_model = next(model for model in status["models"] if model["modelID"] == "apple/system")
assert system_model["availability"] == "available", system_model
assert system_model["contextSize"] > 0, system_model

generated = events("generate")
assert any(event.get("type") == "delta" for event in generated), generated
completed = next(event for event in generated if event.get("type") == "completed")
assert completed["content"], completed
assert completed["usage"]["outputTokens"] > 0, completed

structured = events("structured")[0]
assert structured["label"], structured
assert 0 <= structured["confidence"] <= 100, structured

tool = events("tool")[0]
assert tool["invokedKeys"] == ["milestone-zero"], tool
assert "mesh-fixture-value-for-milestone-zero" in tool["content"], tool

cancelled = events("cancel")
assert any(event.get("type") == "cancelled" for event in cancelled), cancelled

summary = {
    "status": "pass",
    "runtime": status["runtimeID"],
    "model": system_model["modelID"],
    "variant": system_model["variant"],
    "context_size": system_model["contextSize"],
    "capabilities": system_model["capabilities"],
    "generation": {
        "elapsed_ms": completed["elapsedMilliseconds"],
        "ttft_ms": completed["timeToFirstTokenMilliseconds"],
        "input_tokens": completed["usage"]["inputTokens"],
        "output_tokens": completed["usage"]["outputTokens"],
    },
    "structured_label": structured["label"],
    "tool_invocations": tool["invokedKeys"],
    "cancellation": "observed",
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
