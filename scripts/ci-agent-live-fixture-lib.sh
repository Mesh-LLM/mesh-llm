#!/usr/bin/env bash

# Preserve toolchain discovery when Pi later exports its isolated client HOME.
AGENT_SMOKE_AUTOMATION_HOME="${HOME:-}"
AGENT_SMOKE_AUTOMATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

agent_smoke_automation() {
    if [[ "${MESH_LLM_AUTOMATION_BIN+set}" == set ]]; then
        if [[ "$MESH_LLM_AUTOMATION_BIN" != /* || ! -f "$MESH_LLM_AUTOMATION_BIN" || ! -x "$MESH_LLM_AUTOMATION_BIN" ]]; then
            echo "MESH_LLM_AUTOMATION_BIN must be an absolute executable" >&2
            return 1
        fi
        env HOME="$AGENT_SMOKE_AUTOMATION_HOME" "$MESH_LLM_AUTOMATION_BIN" automation "$@"
    else
        env HOME="$AGENT_SMOKE_AUTOMATION_HOME" just --justfile "$AGENT_SMOKE_AUTOMATION_ROOT/Justfile" automation-run automation "$@"
    fi
}

agent_smoke_evidence() {
    agent_smoke_automation agent-fixture-evidence "$@"
}

agent_smoke_normalize_v1_base() {
    local base_url="${1:?base URL required}"
    base_url="${base_url%/}"
    if [[ "$base_url" != */v1 ]]; then
        base_url="${base_url}/v1"
    fi
    printf '%s\n' "$base_url"
}

agent_smoke_pick_model() {
    local base_url="${1:?base URL required}"
    local requested="${2:-}"

    if [[ -n "$requested" ]]; then
        printf '%s\n' "$requested"
        return 0
    fi

    curl -sf "${base_url%/}/models" |
        agent_smoke_automation agent-pick-model
}

agent_smoke_write_fixture() {
    local work_dir="${1:?work dir required}"
    agent_smoke_automation agent-fixture-inputs coding-setup "$work_dir"
}

agent_smoke_prompt() {
    cat <<'EOF'
You are running a CI smoke test in a throwaway project. Use filesystem and coding tools; do not answer from memory.

Tasks:
1. Inspect the repository.
2. Read facts/signal.md, src/matrix.txt, and tests/smoke_calc.rs.
3. Implement src/smoke_calc.rs using only the Rust standard library so the tests pass.
4. Run just test.
5. Answer exactly these four lines with no Markdown and no extra text:
CODEWORD=<the CODEWORD value from facts/signal.md>
CHECKSUM=<the checksum value from src/matrix.txt>
PRIME_SUM=<the sum of prime numbers from the numbers line in src/matrix.txt>
QUESTION=<comma-separated relative paths whose file name contains signal>
EOF
}

agent_smoke_long_prompt_soak() {
    local base_url="${1:?base URL required}"
    local model="${2:?model required}"
    local work_dir="${3:?work dir required}"
    local label="${4:?label required}"
    local target_chars="${AGENT_SMOKE_LONG_PROMPT_CHARS:-${OPENCODE_SMOKE_LONG_PROMPT_CHARS:-65536}}"
    local max_time="${AGENT_SMOKE_LONG_PROMPT_MAX_TIME:-180}"
    local slug

    if [[ ! "$target_chars" =~ ^[0-9]+$ ]]; then
        echo "${label} long prompt char count must be numeric: ${target_chars}" >&2
        return 1
    fi
    if [[ "$target_chars" -le 0 ]]; then
        echo "${label} long prompt soak skipped"
        return 0
    fi
    slug="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]_' '-')"

    local payload="${work_dir}/${slug}-long-prompt-payload.json"
    local response="${work_dir}/${slug}-long-prompt-response.json"

    agent_smoke_automation agent-fixture-inputs soak "$model" "$target_chars" "$payload"

    curl -fsS --max-time "$max_time" \
        "${base_url%/}/chat/completions" \
        -H 'content-type: application/json' \
        -d @"$payload" \
        -o "$response"

    agent_smoke_evidence soak "$response" "$label"
}

agent_smoke_validate_fixture() {
    local work_dir="${1:?work dir required}"
    local initial_sha="${2:?initial sha required}"
    local output_path="${3:?output path required}"
    local label="${4:?label required}"
    local require_tool_events="${5:-false}"

    if ! agent_smoke_automation agent-fixture-inputs coding-verify "$work_dir" "$initial_sha" "$(command -v just)" "$(command -v rustc)"; then
        echo "${label} fixture implementation verification failed after the coding session" >&2
        sed -n '1,220p' "${work_dir}/src/smoke_calc.rs" >&2 || true
        return 1
    fi

    agent_smoke_evidence result "$output_path" "$label" "$require_tool_events"
}
