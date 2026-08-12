#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"
BINARY="$PACKAGE_ROOT/bin/mesh-apple-runtime"
OUTPUT_DIR="$REPO_ROOT/target/apple-runtime/launchd"
LABEL="com.mesh-llm.apple-runtime.experimental.$$"
DOMAIN="gui/$(id -u)"
PLIST="$OUTPUT_DIR/$LABEL.plist"
STDOUT_LOG="$OUTPUT_DIR/stdout.jsonl"
STDERR_LOG="$OUTPUT_DIR/stderr.log"

[[ -x "$BINARY" ]] || {
    echo "missing packaged Apple runtime; run just apple::package" >&2
    exit 2
}

mkdir -p "$OUTPUT_DIR"
: > "$STDOUT_LOG"
: > "$STDERR_LOG"

cleanup() {
    launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python3 - "$PLIST" "$LABEL" "$BINARY" "$STDOUT_LOG" "$STDERR_LOG" <<'PY'
import plistlib
import sys

path, label, binary, stdout, stderr = sys.argv[1:]
payload = {
    "Label": label,
    "ProgramArguments": [
        binary,
        "generate",
        "--prompt",
        "Reply with exactly: launchd apple system provider ready",
        "--max-tokens",
        "32",
        "--temperature",
        "0",
    ],
    "RunAtLoad": True,
    "StandardOutPath": stdout,
    "StandardErrorPath": stderr,
    "ProcessType": "Background",
}
with open(path, "wb") as handle:
    plistlib.dump(payload, handle, sort_keys=True)
PY

launchctl bootstrap "$DOMAIN" "$PLIST"

for _ in $(seq 1 120); do
    if grep -q '"type":"completed"' "$STDOUT_LOG"; then
        break
    fi
    if grep -q '"type":"error"' "$STDOUT_LOG"; then
        break
    fi
    sleep 1
done

launchctl print "$DOMAIN/$LABEL" > "$OUTPUT_DIR/launchctl.txt" 2>&1 || true
codesign -d --entitlements :- "$BINARY" > "$OUTPUT_DIR/entitlements.plist" 2>/dev/null || true

python3 - "$STDOUT_LOG" "$STDERR_LOG" "$OUTPUT_DIR/summary.json" <<'PY'
import json
import pathlib
import sys

stdout_path, stderr_path, summary_path = map(pathlib.Path, sys.argv[1:])
events = [json.loads(line) for line in stdout_path.read_text().splitlines() if line]
completed = next((event for event in events if event.get("type") == "completed"), None)
errors = [event for event in events if event.get("type") == "error"]
summary = {
    "status": "pass" if completed else "fail",
    "completed": completed,
    "errors": errors,
    "stderr": stderr_path.read_text(),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if completed is None:
    raise SystemExit(1)
PY
