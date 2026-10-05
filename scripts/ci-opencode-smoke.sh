#!/usr/bin/env bash
# ci-opencode-smoke.sh - exercise OpenCode's filesystem tools with a coding model.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCODE_AUTOMATION_HOME="${HOME:-}"
# Frozen automation selection begins.
opencode_automation=(env "HOME=$OPENCODE_AUTOMATION_HOME" just --justfile "$ROOT/Justfile" automation-run)
if [[ "${MESH_LLM_AUTOMATION_BIN+set}" == set ]]; then
    if [[ "$MESH_LLM_AUTOMATION_BIN" != /* || ! -f "$MESH_LLM_AUTOMATION_BIN" || ! -x "$MESH_LLM_AUTOMATION_BIN" ]]; then
        echo 'MESH_LLM_AUTOMATION_BIN must be an absolute executable' >&2
        exit 1
    fi
    opencode_automation=("$MESH_LLM_AUTOMATION_BIN")
fi
# Frozen automation selection ends.

if [[ -n "${MESH_OPENCODE_BASE_URL:-}" ]]; then
    MESH_BASE_URL="$MESH_OPENCODE_BASE_URL"
elif [[ -n "${MESH_CLIENT_API_BASE:-}" ]]; then
    MESH_BASE_URL="${MESH_CLIENT_API_BASE%/}/v1"
else
    MESH_BASE_URL="http://127.0.0.1:9337/v1"
fi
MESH_MODEL="${MESH_OPENCODE_MODEL:-${MESH_SDK_MODEL_ID:-}}"
MODEL="${OPENCODE_SMOKE_MODEL:-}"
TIMEOUT_SECONDS="${OPENCODE_SMOKE_TIMEOUT:-300}"
WORK_DIR="${OPENCODE_SMOKE_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/mesh-opencode-smoke.XXXXXX")}"
OUTPUT_JSONL="${OPENCODE_SMOKE_OUTPUT:-${WORK_DIR}/opencode-output.jsonl}"
TURN1_JSONL="${OPENCODE_SMOKE_TURN1_OUTPUT:-${WORK_DIR}/opencode-turn1.jsonl}"
TURN2_JSONL="${OPENCODE_SMOKE_TURN2_OUTPUT:-${WORK_DIR}/opencode-turn2.jsonl}"
ERROR_LOG="${OPENCODE_SMOKE_ERROR_LOG:-${WORK_DIR}/opencode-stderr.log}"
SURFACE_LOG="${OPENCODE_SMOKE_SURFACE_LOG:-${WORK_DIR}/openai-surface.jsonl}"
SURFACE_PROXY_LOG="${OPENCODE_SMOKE_SURFACE_PROXY_LOG:-${WORK_DIR}/openai-surface-proxy.log}"
SURFACE_CAPTURE="${OPENCODE_SMOKE_CAPTURE_SURFACE:-true}"
LONG_PROMPT_CHARS="${OPENCODE_SMOKE_LONG_PROMPT_CHARS:-65536}"

if ! command -v opencode >/dev/null 2>&1; then
    echo "opencode is not installed or is not on PATH" >&2
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    MODELS_JSON="$(curl -sf "${MESH_BASE_URL%/}/models" 2>/dev/null || true)"
    if [[ -z "$MODELS_JSON" ]]; then
        echo "::notice::Skipping OpenCode smoke because mesh endpoint is not reachable at ${MESH_BASE_URL%/}/models."
        echo "::notice::Start mesh-llm or set MESH_OPENCODE_BASE_URL to an OpenAI-compatible mesh /v1 endpoint."
        exit 0
    fi

    if [[ -z "$MESH_MODEL" ]]; then
        MESH_MODEL="$(
            printf '%s' "$MODELS_JSON" | "${opencode_automation[@]}" automation agent-pick-model 2>/dev/null || echo ""
        )"
    fi

    if [[ -z "$MESH_MODEL" ]]; then
        echo "Mesh endpoint returned no models from ${MESH_BASE_URL%/}/models" >&2
        exit 1
    fi

    MODEL="mesh/${MESH_MODEL}"
fi

resolve_opencode_model_identity() {
    if [[ -n "$MESH_MODEL" ]]; then
        return 0
    fi
    if [[ "${OPENCODE_SMOKE_MODEL:-}" == mesh/* ]]; then
        MESH_MODEL="${OPENCODE_SMOKE_MODEL#mesh/}"
        if [[ -n "$MESH_MODEL" ]]; then
            return 0
        fi
        echo 'OPENCODE_SMOKE_MODEL=mesh/ requires a nonempty model identity' >&2
        return 1
    fi
    echo 'OpenCode compatibility smoke requires MESH_OPENCODE_MODEL or MESH_SDK_MODEL_ID for a non-mesh provider' >&2
    return 1
}
resolve_opencode_model_identity

CONFIG_BASE_URL="$MESH_BASE_URL"
SURFACE_PROXY_PID=""
cleanup_surface_proxy() {
    if [[ -n "$SURFACE_PROXY_PID" ]]; then
        kill "$SURFACE_PROXY_PID" 2>/dev/null || true
        wait "$SURFACE_PROXY_PID" 2>/dev/null || true
    fi
}
trap cleanup_surface_proxy EXIT

if [[ "$SURFACE_CAPTURE" == "true" && "$MODEL" == mesh/* ]]; then
    SURFACE_READY="${WORK_DIR}/openai-surface-proxy.ready"
    rm -f -- "$SURFACE_READY"
    "${opencode_automation[@]}" automation agent-recording-proxy "$MESH_BASE_URL" "$SURFACE_LOG" "$SURFACE_READY" 3600 >"$SURFACE_PROXY_LOG" 2>&1 &
    SURFACE_PROXY_PID=$!

    for _ in $(seq 1 100); do
        if [[ -s "$SURFACE_READY" ]]; then
            CONFIG_BASE_URL="$(cat "$SURFACE_READY")"
            break
        fi
        if ! kill -0 "$SURFACE_PROXY_PID" 2>/dev/null; then
            echo "OpenAI surface capture proxy exited unexpectedly" >&2
            cat "$SURFACE_PROXY_LOG" >&2 || true
            exit 1
        fi
        sleep 0.1
    done

    if [[ "$CONFIG_BASE_URL" == "$MESH_BASE_URL" ]]; then
        echo "Timed out starting OpenAI surface capture proxy" >&2
        cat "$SURFACE_PROXY_LOG" >&2 || true
        exit 1
    fi
fi

prepare_opencode_config() {
    if [[ -n "${OPENCODE_CONFIG_CONTENT:-}" ]]; then
        return 0
    fi
    if [[ "$MODEL" == mesh/* ]]; then
        export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
        OPENCODE_CONFIG_CONTENT="$("${opencode_automation[@]}" automation agent-client-config opencode "$CONFIG_BASE_URL" "$MESH_MODEL")"
    else
        OPENCODE_CONFIG_CONTENT="$("${opencode_automation[@]}" automation agent-client-config opencode)"
    fi
    export OPENCODE_CONFIG_CONTENT
}
prepare_opencode_config

export OPENCODE_DISABLE_AUTOUPDATE="${OPENCODE_DISABLE_AUTOUPDATE:-true}"
export OPENCODE_DISABLE_PRUNE="${OPENCODE_DISABLE_PRUNE:-true}"
export OPENCODE_DISABLE_LSP_DOWNLOAD="${OPENCODE_DISABLE_LSP_DOWNLOAD:-true}"

INITIAL_IMPL_SHA="$("${opencode_automation[@]}" automation agent-fixture-inputs coding-setup "$WORK_DIR")"

TURN1_PROMPT='You are running turn 1 of a CI smoke test in a throwaway project. Use filesystem tools, inspect the repository, read the tests, and implement src/smoke_calc.rs so the tests are intended to pass. This is a real coding task: edit the file, do not just describe the edit. Keep the implementation small and dependency-free. End your response with TURN1_DONE.'

TURN2_PROMPT='This is turn 2 of the same CI smoke test. Continue from the prior work. Run just test, fix src/smoke_calc.rs if anything fails, then answer exactly these four lines with no Markdown and no extra text:
CODEWORD=<the CODEWORD value from facts/signal.md>
CHECKSUM=<the checksum value from src/matrix.txt>
PRIME_SUM=<the sum of prime numbers from the numbers line in src/matrix.txt>
QUESTION=<comma-separated relative paths whose file name contains signal>'

echo "=== CI OpenCode Smoke Test ==="
echo "  model:      ${MODEL}"
if [[ "$MODEL" == mesh/* ]]; then
    echo "  mesh:       ${MESH_BASE_URL%/}"
    if [[ "$SURFACE_CAPTURE" == "true" ]]; then
        echo "  capture:    ${CONFIG_BASE_URL%/}"
    fi
fi
echo "  opencode:   $(opencode --version 2>/dev/null || echo unknown)"
echo "  work dir:   ${WORK_DIR}"
echo "  output:     ${OUTPUT_JSONL}"

if [[ "$SURFACE_CAPTURE" == "true" && "$MODEL" == mesh/* ]]; then
    curl -sf "${CONFIG_BASE_URL%/}/models" >/dev/null
    SURFACE_PROBE_PAYLOAD="${WORK_DIR}/openai-surface-probe.json"
    SURFACE_PROBE_RESPONSE="${WORK_DIR}/openai-surface-probe-response.json"
    "${opencode_automation[@]}" automation agent-fixture-inputs surface "$MESH_MODEL" "$SURFACE_PROBE_PAYLOAD"
    curl -fsS --max-time 120 \
        "${CONFIG_BASE_URL%/}/chat/completions" \
        -H 'content-type: application/json' \
        -d @"$SURFACE_PROBE_PAYLOAD" \
        -o "$SURFACE_PROBE_RESPONSE"
    "${opencode_automation[@]}" automation agent-fixture-evidence probe "$SURFACE_PROBE_RESPONSE" OpenCode

    if [[ "$LONG_PROMPT_CHARS" -gt 0 ]]; then
        LONG_PROMPT_PAYLOAD="${WORK_DIR}/openai-long-prompt-probe.json"
        LONG_PROMPT_RESPONSE="${WORK_DIR}/openai-long-prompt-probe-response.json"
        "${opencode_automation[@]}" automation agent-fixture-inputs soak "$MESH_MODEL" "$LONG_PROMPT_CHARS" "$LONG_PROMPT_PAYLOAD"
        curl -fsS --max-time 180 \
            "${CONFIG_BASE_URL%/}/chat/completions" \
            -H 'content-type: application/json' \
            -d @"$LONG_PROMPT_PAYLOAD" \
            -o "$LONG_PROMPT_RESPONSE"
        "${opencode_automation[@]}" automation agent-fixture-evidence soak "$LONG_PROMPT_RESPONSE" OpenCode
    fi
fi

BASE_RUN_ARGS=(run --format json --model "${MODEL}" --dir "${WORK_DIR}")
if [[ -n "${OPENCODE_SMOKE_VARIANT:-}" ]]; then
    BASE_RUN_ARGS+=(--variant "${OPENCODE_SMOKE_VARIANT}")
fi

if command -v timeout >/dev/null 2>&1; then
    OPENCODE_COMMAND=(timeout "${TIMEOUT_SECONDS}" opencode)
else
    OPENCODE_COMMAND=(opencode)
fi

if ! "${OPENCODE_COMMAND[@]}" "${BASE_RUN_ARGS[@]}" "${TURN1_PROMPT}" >"${TURN1_JSONL}" 2>"${ERROR_LOG}"; then
    echo "OpenCode smoke turn 1 failed" >&2
    echo "--- opencode stderr ---" >&2
    tail -120 "${ERROR_LOG}" >&2 || true
    echo "--- opencode json output ---" >&2
    tail -120 "${TURN1_JSONL}" >&2 || true
    exit 1
fi

if ! SESSION_ID="$("${opencode_automation[@]}" automation agent-fixture-evidence opencode-session "${TURN1_JSONL}")"; then
    echo "OpenCode turn 1 did not emit a usable sessionID" >&2
    tail -120 "${TURN1_JSONL}" >&2 || true
    exit 1
fi

if ! "${OPENCODE_COMMAND[@]}" "${BASE_RUN_ARGS[@]}" --session "$SESSION_ID" "${TURN2_PROMPT}" >"${TURN2_JSONL}" 2>>"${ERROR_LOG}"; then
    echo "OpenCode smoke turn 2 failed" >&2
    echo "--- opencode stderr ---" >&2
    tail -120 "${ERROR_LOG}" >&2 || true
    echo "--- opencode json output ---" >&2
    tail -120 "${TURN2_JSONL}" >&2 || true
    exit 1
fi

cat "${TURN1_JSONL}" "${TURN2_JSONL}" >"${OUTPUT_JSONL}"

if ! "${opencode_automation[@]}" automation agent-fixture-inputs coding-verify "$WORK_DIR" "$INITIAL_IMPL_SHA" "$(command -v just)" "$(command -v rustc)"; then
    echo "OpenCode smoke fixture implementation verification failed" >&2
    sed -n '1,220p' "${WORK_DIR}/src/smoke_calc.rs" >&2 || true
    exit 1
fi

if ! "${opencode_automation[@]}" automation agent-fixture-evidence opencode-result "${OUTPUT_JSONL}"; then
    echo "--- output tail ---" >&2
    tail -80 "${OUTPUT_JSONL}" >&2 || true
    exit 1
fi

if [[ "$SURFACE_CAPTURE" == "true" && "$MODEL" == mesh/* ]]; then
    "${opencode_automation[@]}" automation agent-fixture-evidence surface "$SURFACE_LOG" "$LONG_PROMPT_CHARS"
fi
