#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"
BINARY="$PACKAGE_ROOT/bin/mesh-apple-runtime"
OUTPUT_DIR="$REPO_ROOT/target/apple-runtime/instruments"
FOUNDATION_TRACE="$OUTPUT_DIR/foundation-models.trace"
CORE_AI_TRACE="$OUTPUT_DIR/core-ai.trace"

[[ -x "$BINARY" ]] || {
    echo "missing packaged Apple runtime; run just apple::package" >&2
    exit 2
}

mkdir -p "$OUTPUT_DIR"
rm -rf "$FOUNDATION_TRACE" "$CORE_AI_TRACE"
rm -f \
    "$OUTPUT_DIR/foundation-models.jsonl" \
    "$OUTPUT_DIR/core-ai.jsonl" \
    "$OUTPUT_DIR/foundation-models-toc.xml" \
    "$OUTPUT_DIR/core-ai-toc.xml" \
    "$OUTPUT_DIR/core-ai-ane.xml" \
    "$OUTPUT_DIR/summary.json"

xcrun xctrace record \
    --no-prompt \
    --template "Foundation Models" \
    --time-limit 30s \
    --output "$FOUNDATION_TRACE" \
    --target-stdout "$OUTPUT_DIR/foundation-models.jsonl" \
    --launch -- "$BINARY" generate \
    --prompt "Reply with exactly: foundation models provider ready" \
    --max-tokens 32 \
    --temperature 0

xcrun xctrace record \
    --no-prompt \
    --template "Core AI" \
    --time-limit 30s \
    --output "$CORE_AI_TRACE" \
    --target-stdout "$OUTPUT_DIR/core-ai.jsonl" \
    --launch -- "$BINARY" generate \
    --prompt "Reply with exactly: core ai provider ready" \
    --max-tokens 32 \
    --temperature 0

xcrun xctrace export \
    --input "$FOUNDATION_TRACE" \
    --toc \
    --output "$OUTPUT_DIR/foundation-models-toc.xml"
xcrun xctrace export \
    --input "$CORE_AI_TRACE" \
    --toc \
    --output "$OUTPUT_DIR/core-ai-toc.xml"
xcrun xctrace export \
    --input "$CORE_AI_TRACE" \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="ane-hw-intervals"]' \
    --output "$OUTPUT_DIR/core-ai-ane.xml"

python3 - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

root = pathlib.Path(sys.argv[1])


def completed(name):
    events = [
        json.loads(line)
        for line in (root / f"{name}.jsonl").read_text().splitlines()
        if line
    ]
    return next(event for event in events if event.get("type") == "completed")


foundation = completed("foundation-models")
core_ai = completed("core-ai")
ane_root = ET.parse(root / "core-ai-ane.xml").getroot()
ane_rows = ane_root.findall(".//row")
ane_text = " ".join(ane_root.itertext())
assert ane_rows, "Core AI trace contains no ANE intervals"
assert "Apple Neural Engine" in ane_text, ane_text
assert "Prediction" in ane_text, ane_text

summary = {
    "status": "pass",
    "foundation_models": {
        "elapsed_ms": foundation["elapsedMilliseconds"],
        "ttft_ms": foundation["timeToFirstTokenMilliseconds"],
        "output_tokens": foundation["usage"]["outputTokens"],
    },
    "core_ai": {
        "elapsed_ms": core_ai["elapsedMilliseconds"],
        "ttft_ms": core_ai["timeToFirstTokenMilliseconds"],
        "output_tokens": core_ai["usage"]["outputTokens"],
    },
    "accelerator": "Apple Neural Engine",
    "ane_interval_rows": len(ane_rows),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
