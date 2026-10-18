#!/usr/bin/env bash
# TEMPORARY DIAGNOSTIC (PR #2472 Linux CPU smoke crash).
#
# The product dies with a bare SIGSEGV inside the embedded native runtime and
# prints nothing, so the failing steps are undiagnosable. This helper installs an
# in-process crash reporter (no ptrace required, so it works inside the job
# container) that prints `TRACE <module> <offset> <symbol>` frames and re-raises
# the signal, plus a symbolizer that turns those frames into file:line pairs with
# addr2line in the same job.
#
# Delete this file and its two callers once the crash is root-caused.

# Installs the crash reporter and exports LD_PRELOAD. Prints the directory that
# holds the compiled shim so callers can keep it for the symbolizer.
mesh_llm_install_crash_trace() {
    local directory source

    directory="$(mktemp -d "${TMPDIR:-/tmp}/mesh-llm-crash.XXXXXX")" || return 1
    source="$directory/crash_trace.c"
    cat >"$source" <<'CRASH_TRACE_C'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static char alternate_stack[256 * 1024];

static void emit(const char *text) {
    size_t remaining = strlen(text);
    while (remaining > 0) {
        ssize_t written = write(STDERR_FILENO, text, remaining);
        if (written <= 0) {
            return;
        }
        text += written;
        remaining -= (size_t) written;
    }
}

static void on_crash(int signal_number) {
    void * frames[96];
    char   line[512];
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
    signal(signal_number, SIG_DFL);
    raise(signal_number);
}

__attribute__((constructor)) static void install_crash_trace(void) {
    stack_t stack;
    memset(&stack, 0, sizeof stack);
    stack.ss_sp = alternate_stack;
    stack.ss_size = sizeof alternate_stack;
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

    if command -v cc >/dev/null 2>&1 &&
        cc -shared -fPIC -O1 -o "$directory/crash_trace.so" "$source" -ldl; then
        export LD_PRELOAD="$directory/crash_trace.so${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "diagnostic: native crash reporter installed ($directory/crash_trace.so)"
        printf '%s\n' "$directory"
        return 0
    fi
    echo "diagnostic: native crash reporter unavailable (no working C compiler)" >&2
    return 1
}

# Prints symbolized frames for every TRACE frame found in the given logs.
mesh_llm_symbolize_crash_logs() {
    local log

    for log in "$@"; do
        [ -f "$log" ] || continue
        grep -q "=== NATIVE CRASH TRACE ===" "$log" 2>/dev/null || continue
        echo "--- symbolized native crash frames: $log ---"
        awk '/^TRACE /{ if ($2 == "?") next; if (!($2 in seen)) { seen[$2] = 1; order[++count] = $2 } offs[$2] = offs[$2] " " $3 }
             END { for (i = 1; i <= count; i++) print order[i], offs[order[i]] }' "$log" |
            while read -r module offsets; do
                echo "### $module"
                # shellcheck disable=SC2086
                addr2line -f -C -e "$module" $offsets 2>/dev/null | paste - - || true
            done
        echo "--- raw crash frames ---"
        grep -E "^TRACE |^signal=" "$log" 2>/dev/null | head -60 || true
        echo "--- end symbolized native crash frames ---"
    done
}
