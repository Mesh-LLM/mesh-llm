#!/usr/bin/env bash
# ci-mlx-smoke-test.sh — start mesh-llm with a tiny MLX model on macOS,
# run one inference request, verify template selection, then shut down.
#
# Usage:
#   scripts/ci-mlx-smoke-test.sh <mesh-llm-binary> <mlx-model-dir-or-repo> [expected-template-source] [prompt] [expect-contains] [forbid-contains] [thinking-mode] [expect-exact] [prompt-suite-json]

set -euo pipefail

MESH_LLM="$1"
MODEL_SPEC="$2"
EXPECTED_TEMPLATE_SOURCE="${3:-}"
PROMPT_TEXT="${4:-What is 2+2? Reply with one word only.}"
EXPECT_CONTAINS="${5:-}"
FORBID_CONTAINS="${6:-}"
THINKING_MODE="${7:-}"
EXPECT_EXACT="${8:-}"
PROMPT_SUITE_JSON="${9:-}"
pick_free_port() {
    python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

API_PORT="$(pick_free_port)"
CONSOLE_PORT="$(pick_free_port)"
while [ "$API_PORT" = "$CONSOLE_PORT" ]; do
    CONSOLE_PORT="$(pick_free_port)"
done
MAX_WAIT=300
LOG=/tmp/mesh-llm-ci-mlx.log
ARTIFACT_DIR="${VALIDATION_CASE_DIR:-}"

echo "=== CI MLX Smoke Test ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  model:     $MODEL_SPEC"
echo "  api port:  $API_PORT"
echo "  os:        $(uname -s)"
echo "  prompt:    $PROMPT_TEXT"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "❌ MLX smoke test only supports macOS"
    exit 1
fi

if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi

echo "Starting mesh-llm..."
LAUNCH_PREFIX=()
if command -v stdbuf >/dev/null 2>&1; then
    LAUNCH_PREFIX=(stdbuf -oL -eL)
fi
if [ -d "$MODEL_SPEC" ]; then
    RUST_LOG=info "${LAUNCH_PREFIX[@]}" "$MESH_LLM" \
        --mlx-file "$MODEL_SPEC" \
        --no-draft \
        --port "$API_PORT" \
        --console "$CONSOLE_PORT" \
        > "$LOG" 2>&1 &
else
    RUST_LOG=info "${LAUNCH_PREFIX[@]}" "$MESH_LLM" \
        --model "$MODEL_SPEC" \
        --mlx \
        --no-draft \
        --port "$API_PORT" \
        --console "$CONSOLE_PORT" \
        > "$LOG" 2>&1 &
fi
MESH_PID=$!
echo "  PID: $MESH_PID"

cleanup() {
    echo "Shutting down mesh-llm (PID $MESH_PID)..."
    kill "$MESH_PID" 2>/dev/null || true
    pkill -P "$MESH_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$MESH_PID" 2>/dev/null || true
    wait "$MESH_PID" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

record_chat_artifact() {
    local artifact_name="$1"
    local prompt_text="$2"
    local request_json="$3"
    local response_json="$4"
    local content="$5"
    local finish_reason="$6"
    local expect_contains="$7"
    local expect_contains_ci="$8"
    local expect_contains_all_ci_json="$9"
    local expect_any_ci_json="${10}"
    local forbid_contains="${11}"
    local expect_exact="${12}"

    if [ -z "$ARTIFACT_DIR" ]; then
        return
    fi

    mkdir -p "$ARTIFACT_DIR/chat"
    CASE_LABEL="$artifact_name" \
    PROMPT_TEXT="$prompt_text" \
    REQUEST_JSON="$request_json" \
    RESPONSE_JSON="$response_json" \
    CONTENT_TEXT="$content" \
    FINISH_REASON="$finish_reason" \
    EXPECT_CONTAINS="$expect_contains" \
    EXPECT_CONTAINS_CI="$expect_contains_ci" \
    EXPECT_CONTAINS_ALL_CI_JSON="$expect_contains_all_ci_json" \
    EXPECT_ANY_CI_JSON="$expect_any_ci_json" \
    FORBID_CONTAINS="$forbid_contains" \
    EXPECT_EXACT="$expect_exact" \
    ARTIFACT_PATH="$ARTIFACT_DIR/chat/${artifact_name}.json" \
    python3 - <<'PY'
import json, os
payload = {
    "label": os.environ["CASE_LABEL"],
    "prompt": os.environ["PROMPT_TEXT"],
    "request": json.loads(os.environ["REQUEST_JSON"]),
    "raw_response": json.loads(os.environ["RESPONSE_JSON"]),
    "content": os.environ["CONTENT_TEXT"],
    "finish_reason": os.environ["FINISH_REASON"],
    "expectations": {
        "expect_contains": os.environ["EXPECT_CONTAINS"],
        "expect_contains_ci": os.environ["EXPECT_CONTAINS_CI"],
        "expect_contains_all_ci": json.loads(os.environ["EXPECT_CONTAINS_ALL_CI_JSON"] or "[]"),
        "expect_any_ci": json.loads(os.environ["EXPECT_ANY_CI_JSON"] or "[]"),
        "forbid_contains": os.environ["FORBID_CONTAINS"],
        "expect_exact": os.environ["EXPECT_EXACT"],
    },
}
with open(os.environ["ARTIFACT_PATH"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

echo "Waiting for model to load (up to ${MAX_WAIT}s)..."
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "❌ mesh-llm exited unexpectedly"
        echo "--- Log tail ---"
        tail -80 "$LOG" || true
        exit 1
    fi

    READY=$(curl -sf "http://localhost:${CONSOLE_PORT}/api/status" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('llama_ready', False))" 2>/dev/null || echo "False")
    if [ "$READY" = "True" ]; then
        echo "✅ Model loaded in ${i}s"
        break
    fi

    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "❌ Model failed to load within ${MAX_WAIT}s"
        echo "--- Log tail ---"
        tail -80 "$LOG" || true
        exit 1
    fi

    if [ $((i % 15)) -eq 0 ]; then
        echo "  Still waiting... (${i}s)"
    fi
    sleep 1
done

if [ -n "$EXPECTED_TEMPLATE_SOURCE" ]; then
    if ! grep -F "MLX prompt template: loaded HF template from $EXPECTED_TEMPLATE_SOURCE" "$LOG" >/dev/null 2>&1; then
        echo "❌ Expected template source not found in log: $EXPECTED_TEMPLATE_SOURCE"
        echo "--- Log tail ---"
        tail -120 "$LOG" || true
        exit 1
    fi
    echo "✅ Template source matched: $EXPECTED_TEMPLATE_SOURCE"
fi

run_chat_case() {
    local prompt_text="$1"
    local expect_contains="$2"
    local expect_contains_ci="$3"
    local expect_contains_all_ci_json="$4"
    local expect_any_ci_json="$5"
    local forbid_contains="$6"
    local thinking_mode="$7"
    local expect_exact="$8"
    local case_label="$9"
    local max_tokens="${10:-32}"

    echo "Testing /v1/chat/completions ($case_label)..."
    local curl_body
    curl_body=$(python3 - "$prompt_text" "$max_tokens" <<'PY'
import json, sys
prompt = sys.argv[1]
max_tokens = int(sys.argv[2])
print(json.dumps({
    "model": "any",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "seed": 123,
    "enable_thinking": False
}))
PY
)
    local response
    if ! response=$(curl -sf "http://localhost:${API_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$curl_body" 2>&1); then
        echo "❌ Inference request failed"
        echo "$response"
        echo "--- Log tail ---"
        tail -80 "$LOG" || true
        exit 1
    fi

    local content
    content=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    if [ -z "$content" ]; then
        echo "❌ Empty response from inference"
        echo "Raw response: $response"
        exit 1
    fi

    if echo "$content" | grep -F "<think>" >/dev/null 2>&1; then
        echo "❌ Unexpected reasoning output with enable_thinking=false"
        echo "Content: $content"
        exit 1
    fi

    if [ -n "$expect_contains" ] && ! echo "$content" | grep -F "$expect_contains" >/dev/null 2>&1; then
        echo "❌ Response did not contain expected text: $expect_contains"
        echo "Content: $content"
        exit 1
    fi

    if [ -n "$expect_contains_ci" ]; then
        if ! CONTENT="$content" EXPECT_CONTAINS_CI="$expect_contains_ci" python3 - <<'PY'
import os, sys
content = os.environ["CONTENT"].lower()
needle = os.environ["EXPECT_CONTAINS_CI"].lower()
raise SystemExit(0 if needle in content else 1)
PY
        then
            echo "❌ Response did not contain expected text (case-insensitive): $expect_contains_ci"
            echo "Content: $content"
            exit 1
        fi
    fi

    if [ -n "$expect_contains_all_ci_json" ]; then
        if ! CONTENT="$content" EXPECT_CONTAINS_ALL_CI_JSON="$expect_contains_all_ci_json" python3 - <<'PY' >/dev/null
import json, os, sys
content = os.environ["CONTENT"].lower()
needles = json.loads(os.environ["EXPECT_CONTAINS_ALL_CI_JSON"])
missing = [needle for needle in needles if needle.lower() not in content]
if missing:
    print(", ".join(missing))
    raise SystemExit(1)
PY
        then
            local missing_terms
            missing_terms=$(CONTENT="$content" EXPECT_CONTAINS_ALL_CI_JSON="$expect_contains_all_ci_json" python3 - <<'PY'
import json, os
content = os.environ["CONTENT"].lower()
needles = json.loads(os.environ["EXPECT_CONTAINS_ALL_CI_JSON"])
missing = [needle for needle in needles if needle.lower() not in content]
print(", ".join(missing))
PY
)
            echo "❌ Response did not contain all expected terms (case-insensitive): $missing_terms"
            echo "Content: $content"
            exit 1
        fi
    fi

    if [ -n "$expect_any_ci_json" ]; then
        if ! CONTENT="$content" EXPECT_ANY_CI_JSON="$expect_any_ci_json" python3 - <<'PY'
import json, os
content = os.environ["CONTENT"].lower()
needles = json.loads(os.environ["EXPECT_ANY_CI_JSON"])
raise SystemExit(0 if any(needle.lower() in content for needle in needles) else 1)
PY
        then
            echo "❌ Response did not contain any expected text (case-insensitive): $expect_any_ci_json"
            echo "Content: $content"
            exit 1
        fi
    fi

    if [ -n "$expect_exact" ]; then
        local normalized_content normalized_expected
        normalized_content=$(printf '%s' "$content" | python3 -c "import sys; print(sys.stdin.read().strip())")
        normalized_expected=$(printf '%s' "$expect_exact" | python3 -c "import sys; print(sys.stdin.read().strip())")
        if [ "$normalized_content" != "$normalized_expected" ]; then
            echo "❌ Response did not exactly match expected text"
            echo "Expected: $normalized_expected"
            echo "Content:  $normalized_content"
            exit 1
        fi
    fi

    if [ -n "$forbid_contains" ] && echo "$content" | grep -F "$forbid_contains" >/dev/null 2>&1; then
        echo "❌ Response contained forbidden text: $forbid_contains"
        echo "Content: $content"
        exit 1
    fi

    local finish_reason
    finish_reason=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0].get('finish_reason',''))" 2>/dev/null || echo "")
    if [ -z "$finish_reason" ]; then
        echo "❌ Missing finish_reason in response"
        echo "Raw response: $response"
        exit 1
    fi

    record_chat_artifact "$case_label" "$prompt_text" "$curl_body" "$response" "$content" "$finish_reason" "$expect_contains" "$expect_contains_ci" "$expect_contains_all_ci_json" "$expect_any_ci_json" "$forbid_contains" "$expect_exact"

    echo "✅ Inference response: $content"

    if [ -n "$thinking_mode" ]; then
        echo "Testing explicit reasoning output ($case_label)..."
        local thinking_response thinking_content
        thinking_response=$(curl -sf "http://localhost:${API_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$(python3 - "$prompt_text" <<'PY'
import json, sys
prompt = sys.argv[1]
print(json.dumps({
    "model": "any",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "seed": 123,
    "enable_thinking": True
}))
PY
)" 2>&1)

        thinking_content=$(echo "$thinking_response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
        if [ -z "$thinking_content" ]; then
            echo "❌ Empty response from explicit reasoning request"
            echo "Raw response: $thinking_response"
            exit 1
        fi
        case "$thinking_mode" in
            tagged)
                if ! echo "$thinking_content" | grep -F "<think>" >/dev/null 2>&1; then
                    echo "❌ Explicit reasoning response did not contain <think> tags"
                    echo "Content: $thinking_content"
                    exit 1
                fi
                ;;
            multiline)
                if [ "$thinking_content" = "$content" ]; then
                    echo "❌ Explicit reasoning response matched non-thinking response"
                    echo "Content: $thinking_content"
                    exit 1
                fi
                if ! printf '%s' "$thinking_content" | python3 -c "import sys; s=sys.stdin.read(); raise SystemExit(0 if '\n' in s else 1)"; then
                    echo "❌ Explicit reasoning response was not multiline"
                    echo "Content: $thinking_content"
                    exit 1
                fi
                ;;
            *)
                echo "❌ Unknown thinking mode: $thinking_mode"
                exit 1
                ;;
        esac
        record_chat_artifact "${case_label}.thinking" "$prompt_text" "$(python3 - "$prompt_text" <<'PY'
import json, sys
prompt = sys.argv[1]
print(json.dumps({
    "model": "any",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "seed": 123,
    "enable_thinking": True
}))
PY
)" "$thinking_response" "$thinking_content" "stop" "" "" "[]" "[]" "" ""
        echo "✅ Explicit reasoning response: $thinking_content"
    fi
}

run_chat_case "$PROMPT_TEXT" "$EXPECT_CONTAINS" "" "" "" "$FORBID_CONTAINS" "$THINKING_MODE" "$EXPECT_EXACT" "primary" "32"

if [ -n "$PROMPT_SUITE_JSON" ]; then
    echo "Running extra prompt suite..."
    python3 - "$PROMPT_SUITE_JSON" <<'PY' | while IFS=$'\t' read -r label prompt expect_contains expect_contains_ci expect_contains_all_ci_json expect_any_ci_json forbid_contains thinking_mode expect_exact max_tokens; do
import json, sys
suite = json.loads(sys.argv[1])
EMPTY = "__EMPTY__"
for index, case in enumerate(suite, start=1):
    print("\t".join([
        str(case.get("label", f"case-{index}")) or EMPTY,
        str(case.get("prompt", "")) or EMPTY,
        str(case.get("expect_contains", "")) or EMPTY,
        str(case.get("expect_contains_ci", "")) or EMPTY,
        json.dumps(case.get("expect_contains_all_ci", []), separators=(",", ":")) if "expect_contains_all_ci" in case else EMPTY,
        json.dumps(case.get("expect_any_ci", []), separators=(",", ":")) if "expect_any_ci" in case else EMPTY,
        str(case.get("forbid_contains", "")) or EMPTY,
        str(case.get("thinking_mode", "")) or EMPTY,
        str(case.get("expect_exact", "")) or EMPTY,
        str(case.get("max_tokens", "")) or EMPTY,
    ]))
PY
        [ "$label" = "__EMPTY__" ] && label=""
        [ "$prompt" = "__EMPTY__" ] && prompt=""
        [ "$expect_contains" = "__EMPTY__" ] && expect_contains=""
        [ "$expect_contains_ci" = "__EMPTY__" ] && expect_contains_ci=""
        [ "$expect_contains_all_ci_json" = "__EMPTY__" ] && expect_contains_all_ci_json=""
        [ "$expect_any_ci_json" = "__EMPTY__" ] && expect_any_ci_json=""
        [ "$forbid_contains" = "__EMPTY__" ] && forbid_contains=""
        [ "$thinking_mode" = "__EMPTY__" ] && thinking_mode=""
        [ "$expect_exact" = "__EMPTY__" ] && expect_exact=""
        [ "$max_tokens" = "__EMPTY__" ] && max_tokens=""
        [ -z "$max_tokens" ] && max_tokens="32"
        run_chat_case "$prompt" "$expect_contains" "$expect_contains_ci" "$expect_contains_all_ci_json" "$expect_any_ci_json" "$forbid_contains" "$thinking_mode" "$expect_exact" "$label" "$max_tokens"
    done
fi

echo "Testing /v1/models..."
MODELS=$(curl -sf "http://localhost:${API_PORT}/v1/models" 2>&1)
MODEL_COUNT=$(echo "$MODELS" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "0")
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "❌ No models in /v1/models"
    echo "$MODELS"
    exit 1
fi
if [ -n "$ARTIFACT_DIR" ]; then
    mkdir -p "$ARTIFACT_DIR/models"
    printf '%s\n' "$MODELS" > "$ARTIFACT_DIR/models/v1-models.json"
fi
echo "✅ /v1/models returned $MODEL_COUNT model(s)"

echo ""
echo "=== MLX smoke test passed ==="
