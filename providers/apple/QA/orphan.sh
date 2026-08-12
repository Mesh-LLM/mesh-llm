#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"
BINARY="$PACKAGE_ROOT/bin/mesh-apple-runtime"
OUTPUT_DIR="$REPO_ROOT/target/apple-runtime/orphan"
PID_FILE="$OUTPUT_DIR/provider.pid"
STDOUT_LOG="$OUTPUT_DIR/stdout.jsonl"
STDERR_LOG="$OUTPUT_DIR/stderr.log"
SUPERVISOR_PID=""
CHILD_PID=""

[[ -x "$BINARY" ]] || {
    echo "missing packaged Apple runtime; run just apple::package" >&2
    exit 2
}

cleanup() {
    if [[ "$CHILD_PID" =~ ^[0-9]+$ ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
    fi
    if [[ "$SUPERVISOR_PID" =~ ^[0-9]+$ ]] && kill -0 "$SUPERVISOR_PID" 2>/dev/null; then
        kill -TERM "$SUPERVISOR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"
rm -f "$PID_FILE" "$STDOUT_LOG" "$STDERR_LOG" "$OUTPUT_DIR/summary.json"

BINARY_ENV="$BINARY" \
PID_FILE_ENV="$PID_FILE" \
STDOUT_ENV="$STDOUT_LOG" \
STDERR_ENV="$STDERR_LOG" \
/bin/zsh -c '
    "$BINARY_ENV" generate \
        --prompt "Write a long, detailed history of distributed computing with many sections." \
        --max-tokens 512 \
        --parent-pid "$$" \
        >"$STDOUT_ENV" \
        2>"$STDERR_ENV" &
    child_pid=$!
    echo "$child_pid" >"$PID_FILE_ENV"
    wait "$child_pid"
' &
SUPERVISOR_PID=$!

for _ in $(seq 1 100); do
    [[ -s "$PID_FILE" ]] && break
    sleep 0.05
done
[[ -s "$PID_FILE" ]] || {
    echo "supervisor did not report provider pid" >&2
    exit 1
}

CHILD_PID="$(tr -d '[:space:]' < "$PID_FILE")"
[[ "$CHILD_PID" =~ ^[0-9]+$ ]] || {
    echo "invalid provider pid: $CHILD_PID" >&2
    exit 1
}
kill -0 "$CHILD_PID"

kill -KILL "$SUPERVISOR_PID"
wait "$SUPERVISOR_PID" 2>/dev/null || true
SUPERVISOR_PID=""

for _ in $(seq 1 100); do
    if ! kill -0 "$CHILD_PID" 2>/dev/null; then
        python3 - "$OUTPUT_DIR/summary.json" "$CHILD_PID" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
summary = {
    "status": "pass",
    "runtime_pid": int(sys.argv[2]),
    "result": "Apple runtime exited after supervisor SIGKILL",
}
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
        CHILD_PID=""
        exit 0
    fi
    sleep 0.05
done

echo "Apple runtime $CHILD_PID survived supervisor SIGKILL" >&2
exit 1
