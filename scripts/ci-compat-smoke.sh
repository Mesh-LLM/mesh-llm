#!/usr/bin/env bash
# ci-compat-smoke.sh - run OpenAI client compatibility probes against mesh-llm.
#
# Usage: scripts/ci-compat-smoke.sh <mesh-llm-binary> <bin-dir> <model-path>

set -euo pipefail
# Required SDK environment admission begins.
SDK_PYTHON="${MESH_REQUIRED_SDK_PYTHON:-}"
if [[ "$SDK_PYTHON" != /* || ! -f "$SDK_PYTHON" || ! -x "$SDK_PYTHON" ]]; then
    echo "required compatibility SDK smoke needs an absolute executable MESH_REQUIRED_SDK_PYTHON from the locked environment" >&2
    exit 1
fi
# Required SDK environment admission ends.
automation=(cargo xtool)
if [[ -n "${MESH_LLM_AUTOMATION_BIN:-}" ]]; then
    automation=("$MESH_LLM_AUTOMATION_BIN")
fi

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
MODEL="${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
DEVICE="${MESH_COMPAT_DEVICE:-CPU}"
API_PORT="${MESH_COMPAT_API_PORT:-9348}"
CONSOLE_PORT="${MESH_COMPAT_CONSOLE_PORT:-3142}"
MAX_WAIT="${MESH_COMPAT_MAX_WAIT:-180}"
LOG="${MESH_COMPAT_LOG:-/tmp/mesh-llm-compat-ci.log}"
BATCH_SIZE="${MESH_COMPAT_BATCH_SIZE:-}"
UBATCH_SIZE="${MESH_COMPAT_UBATCH_SIZE:-}"
ATTESTATION_PUBLIC_KEY_FILE="${MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE:-}"
ATTESTATION_EXPECTED_STATUS="${MESH_RELEASE_ATTESTATION_EXPECTED_STATUS:-valid}"
# Admit immutable external SDK source before starting the product or SDK children.
"${automation[@]}" automation smoke-observation sdk-source --kind root >/dev/null
SMOKE_STATE_DIR="$(mktemp -d "${MESH_COMPAT_STATE_PARENT:-/tmp}/mesh-llm-compat-state.XXXXXX")"
SDK_EVIDENCE_DIR="$(mktemp -d "${MESH_COMPAT_SDK_EVIDENCE_PARENT:-/tmp}/mesh-llm-sdk-evidence.XXXXXX")"
SMOKE_CONFIG_PATH="$SMOKE_STATE_DIR/config.toml"
SMOKE_RUNTIME_ROOT="$SMOKE_STATE_DIR/runtime"

inspect_release_attestation() {
    if [[ -z "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
        echo "Release attestation inspect skipped (set MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE to verify stamped binaries)"
        return 0
    fi

    local inspect_json
    local inspect_status
    inspect_json="$("${automation[@]}" release-attestation inspect --binary "$MESH_LLM" --public-key-file "$ATTESTATION_PUBLIC_KEY_FILE" --json)"
    echo "Release attestation inspect: $inspect_json"
    inspect_status="$(printf '%s' "$inspect_json" | "${automation[@]}" product attestation-status)"
    if [[ "$inspect_status" != "$ATTESTATION_EXPECTED_STATUS" ]]; then
        echo "Unexpected embedded release-attestation status: expected $ATTESTATION_EXPECTED_STATUS, got $inspect_status" >&2
        exit 1
    fi
}

echo "=== CI OpenAI Compatibility Smoke ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  bin-dir:   $BIN_DIR (compatibility placeholder)"
echo "  model:     $MODEL"
echo "  api port:  $API_PORT"
echo "  console:   $CONSOLE_PORT"
echo "  device:    $DEVICE"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "Missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Missing model file: $MODEL" >&2
    exit 1
fi

inspect_release_attestation

RUNTIME_BUNDLE="${MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR:-$(cd "$(dirname "$MESH_LLM")" && pwd)/native-runtimes}"
if [[ ! -d "$RUNTIME_BUNDLE" ]]; then
    echo "Missing packaged native runtime beside mesh-llm: $RUNTIME_BUNDLE" >&2
    exit 1
fi
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$RUNTIME_BUNDLE"

if [[ -n "$BATCH_SIZE" || -n "$UBATCH_SIZE" ]]; then
    if [[ -z "$BATCH_SIZE" || -z "$UBATCH_SIZE" ]]; then
        echo "MESH_COMPAT_BATCH_SIZE and MESH_COMPAT_UBATCH_SIZE must be set together" >&2
        exit 1
    fi
    cat >"$SMOKE_CONFIG_PATH" <<EOF
version = 1

[defaults.model_fit]
batch = $BATCH_SIZE
ubatch = $UBATCH_SIZE
EOF
fi

# TEMPORARY DIAGNOSTIC (PR #2472 Linux CPU smoke crash): the product dies with a
# bare SIGSEGV inside the embedded native runtime and prints nothing. Install an
# in-process crash reporter (no ptrace, so it works in the job container) that
# prints module + offset frames, and run under gdb when the container permits it.
CRASH_DIR="$(mktemp -d "${MESH_COMPAT_STATE_PARENT:-/tmp}/mesh-llm-crash.XXXXXX")"
CRASH_SOURCE="$CRASH_DIR/crash_trace.c"
cat >"$CRASH_SOURCE" <<'CRASH_TRACE_C'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char alt_stack[256 * 1024];

static void emit(const char *text) {
    size_t length = strlen(text);
    while (length > 0) {
        ssize_t written = write(STDERR_FILENO, text, length);
        if (written <= 0) {
            return;
        }
        text += written;
        length -= (size_t) written;
    }
}

static void on_crash(int signal_number) {
    void * frames[96];
    char   line[512];
    char   warm[1];
    emit("\n=== NATIVE CRASH TRACE ===\n");
    snprintf(line, sizeof line, "signal=%d\n", signal_number);
    emit(line);
    int count = backtrace(frames, 96);
    for (int index = 0; index < count; ++index) {
        Dl_info info;
        if (dladdr(frames[index], &info) != 0 && info.dli_fbase != NULL && info.dli_fname != NULL) {
            unsigned long long offset =
                (unsigned long long) ((const char *) frames[index] - (const char *) info.dli_fbase);
            snprintf(line, sizeof line, "TRACE %s 0x%llx %s\n", info.dli_fname, offset,
                     info.dli_sname != NULL ? info.dli_sname : "-");
        } else {
            snprintf(line, sizeof line, "TRACE ? 0x%llx -\n", (unsigned long long) (size_t) frames[index]);
        }
        emit(line);
    }
    emit("=== native backtrace_symbols ===\n");
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    emit("=== END NATIVE CRASH TRACE ===\n");
    (void) warm;
    signal(signal_number, SIG_DFL);
    raise(signal_number);
}

__attribute__((constructor)) static void install_crash_trace(void) {
    stack_t stack;
    memset(&stack, 0, sizeof stack);
    stack.ss_sp = alt_stack;
    stack.ss_size = sizeof alt_stack;
    (void) sigaltstack(&stack, NULL);

    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_handler = on_crash;
    action.sa_flags = SA_RESETHAND | SA_NODEFER | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, NULL);
    sigaction(SIGBUS, &action, NULL);
    sigaction(SIGABRT, &action, NULL);
    sigaction(SIGILL, &action, NULL);
    sigaction(SIGFPE, &action, NULL);
}
CRASH_TRACE_C

if command -v cc >/dev/null 2>&1 && cc -shared -fPIC -O1 -o "$CRASH_DIR/crash_trace.so" "$CRASH_SOURCE" -ldl; then
    echo "diagnostic: crash reporter installed ($CRASH_DIR/crash_trace.so)"
    export LD_PRELOAD="$CRASH_DIR/crash_trace.so${LD_PRELOAD:+:$LD_PRELOAD}"
else
    echo "diagnostic: crash reporter unavailable (no working C compiler)"
fi

LAUNCH=()
if command -v gdb >/dev/null 2>&1 && gdb -q -batch -ex run --args /bin/true >/dev/null 2>&1; then
    echo "diagnostic: ptrace permitted, launching the product under gdb"
    LAUNCH=(
        gdb -q -batch
        -ex "set pagination off"
        -ex "set print thread-events off"
        -ex "handle SIGPIPE nostop noprint pass"
        -ex "run"
        -ex "echo \n=== GDB STOPPED ===\n"
        -ex "info program"
        -ex "thread apply all bt 40"
        -ex "echo \n=== END GDB BACKTRACE ===\n"
        -ex "quit"
        --args
    )
fi

env MESH_LLM_CONFIG="$SMOKE_CONFIG_PATH" MESH_LLM_RUNTIME_ROOT="$SMOKE_RUNTIME_ROOT" \
    "${LAUNCH[@]}" \
    "$MESH_LLM" \
    --log-format json \
    serve \
    --model "$MODEL" \
    --no-draft \
    --device "$DEVICE" \
    --ctx-size "${MESH_COMPAT_CTX_SIZE:-256}" \
    --port "$API_PORT" \
    --console "$CONSOLE_PORT" \
    --headless \
    >"$LOG" 2>&1 &
MESH_PID=$!

cleanup() {
    local status=$?
    kill "$MESH_PID" 2>/dev/null || true
    pkill -P "$MESH_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$MESH_PID" 2>/dev/null || true
    wait "$MESH_PID" 2>/dev/null || true
    echo "--- compat mesh-llm log tail ---"
    tail -100 "$LOG" 2>/dev/null || true
    echo "--- end log ---"
    if grep -q "NATIVE CRASH TRACE" "$LOG" 2>/dev/null; then
        echo "--- symbolized native crash frames ---"
        awk '/^TRACE /{ if ($2 == "?") next; if (!($2 in seen)) { seen[$2] = 1; order[++count] = $2 } offs[$2] = offs[$2] " " $3 }
             END { for (i = 1; i <= count; i++) print order[i], offs[order[i]] }' "$LOG" |
            while read -r module offsets; do
                echo "### $module"
                # shellcheck disable=SC2086
                addr2line -f -C -e "$module" $offsets 2>/dev/null | paste - - || true
            done
        echo "--- end symbolized native crash frames ---"
    fi
    # TEMPORARY DIAGNOSTIC: keep the embedded native runtime log so a crash in
    # native code is visible (it is otherwise deleted with the smoke state dir).
    for native_log in "$SMOKE_RUNTIME_ROOT"/*/logs/*.log; do
        [ -f "$native_log" ] || continue
        echo "--- native runtime log tail: $native_log ---"
        tail -120 "$native_log" 2>/dev/null || true
        echo "--- end native runtime log ---"
    done
    echo "SDK evidence retained: $SDK_EVIDENCE_DIR"
    if [ "$status" -ne 0 ]; then
        # SDK child output is only retained on disk; surface it so CI failures are diagnosable.
        for capture in "$SDK_EVIDENCE_DIR"/*.json "$SDK_EVIDENCE_DIR"/sdk-capture-*/*.log; do
            [ -f "$capture" ] || continue
            echo "--- $capture ---"
            tail -c 8192 "$capture" || true
        done
    fi
    rm -rf "$SMOKE_STATE_DIR"
}
trap cleanup EXIT

BASE_URL="http://127.0.0.1:${API_PORT}/v1"
sdk_owned_model_ready() {
    kill -0 "$MESH_PID" 2>/dev/null || return 1
    MODEL_ID="$("${automation[@]}" automation smoke-observation sdk-ready --base-url "$BASE_URL" --timeout-secs "$MAX_WAIT")" || return 1
    kill -0 "$MESH_PID" 2>/dev/null || return 1
}
sdk_owned_model_ready

RUNTIME_ATTESTATION_STATUS="$({
    curl --max-time 5 --connect-timeout 5 -sf "http://127.0.0.1:${CONSOLE_PORT}/api/status" 2>/dev/null |
        "${automation[@]}" automation smoke-observation attestation 2>/dev/null ||
        echo ""
})"
echo "Runtime release attestation status: ${RUNTIME_ATTESTATION_STATUS:-unknown}"
if [[ -n "$ATTESTATION_PUBLIC_KEY_FILE" && "$RUNTIME_ATTESTATION_STATUS" != "$ATTESTATION_EXPECTED_STATUS" ]]; then
    echo "Unexpected runtime release-attestation status: expected $ATTESTATION_EXPECTED_STATUS, got ${RUNTIME_ATTESTATION_STATUS:-<empty>}" >&2
    exit 1
fi

"${automation[@]}" automation smoke-observation sdk-client --client openai --python "$SDK_PYTHON" --base-url "$BASE_URL" --timeout-secs "${MESH_COMPAT_SDK_TIMEOUT_SECS:-240}" --receipt "$SDK_EVIDENCE_DIR/openai-sdk.json"
"${automation[@]}" automation smoke-observation sdk-client --client litellm --python "$SDK_PYTHON" --base-url "$BASE_URL" --model "$MODEL_ID" --timeout-secs "${MESH_COMPAT_SDK_TIMEOUT_SECS:-240}" --receipt "$SDK_EVIDENCE_DIR/litellm-sdk.json"
"${automation[@]}" automation smoke-observation sdk-client --client langchain --python "$SDK_PYTHON" --base-url "$BASE_URL" --model "$MODEL_ID" --timeout-secs "${MESH_COMPAT_SDK_TIMEOUT_SECS:-240}" --receipt "$SDK_EVIDENCE_DIR/langchain-sdk.json"
NODE_PATH="${NODE_PATH:-$(npm root -g 2>/dev/null || true)}" node scripts/ci-openai-node-smoke.cjs --base-url "$BASE_URL"

echo "OpenAI compatibility smoke passed"
