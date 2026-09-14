#!/usr/bin/env bash

# Resolve an LLVM lld that the active toolchain can actually link with.
#
# lld is a build-speed optimization (measured up to 26% faster locally), not a
# correctness requirement. It is upgraded independently of the platform SDK,
# and it has to parse that SDK's text-based stubs. When the SDK moves first,
# lld rejects libSystem.tbd outright and every libSystem symbol comes back
# undefined:
#
#   ld64.lld: error: could not load TAPI file at .../MacOSX.sdk/usr/lib/libSystem.tbd
#   libSystem.tbd:4:20: error: unknown target
#                      arm64e.x1-macos, arm64e.x1-maccatalyst ]
#
# That is SDK-ahead-of-lld skew: downgrading lld makes it worse and pinning an
# older SDK is a version we then have to remember to unpin. So rather than
# assume, probe the resolved linker once with a throwaway link. Use it when it
# works; otherwise say why once and fall back to the platform linker. When an
# lld that understands the newer SDK arrives, the probe passes and lld comes
# back with no repository change.
#
# `cargo check` does not link, so a broken linker is invisible to it and only
# surfaces at the first `cargo build` or `cargo test`. Nothing in this
# repository may commit cargo to a linker it has not probed.

# Print an installed lld (a `-fuse-ld=` value) on stdout, or nothing when none
# is installed. Says nothing about whether it can link; see `lld_links`.
find_lld() {
    case "$(uname -s)" in
        Linux)
            command -v ld.lld >/dev/null 2>&1 && printf 'lld\n'
            ;;
        Darwin)
            _find_lld_darwin
            ;;
        *) ;;
    esac
    return 0
}

_find_lld_darwin() {
    if command -v ld64.lld >/dev/null 2>&1; then
        command -v ld64.lld
        return 0
    fi
    local prefix=""
    if command -v brew >/dev/null 2>&1; then
        prefix="$(brew --prefix lld 2>/dev/null || true)"
        if [[ -n "$prefix" && -x "$prefix/bin/ld64.lld" ]]; then
            printf '%s\n' "$prefix/bin/ld64.lld"
            return 0
        fi
    fi
    local candidate
    for candidate in /opt/homebrew/opt/lld/bin/ld64.lld /usr/local/opt/lld/bin/ld64.lld; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 0
}

# Link a one-line C program with `-fuse-ld=$1` through `cc`, the same driver
# rustc uses. One invocation, tens of milliseconds. The linker's own
# diagnostics are left in LLD_PROBE_OUTPUT so a caller can report the real
# reason instead of guessing one.
lld_links() {
    local linker="$1"
    LLD_PROBE_OUTPUT=""
    if ! command -v cc >/dev/null 2>&1; then
        LLD_PROBE_OUTPUT="no C compiler driver (cc) on PATH to probe with"
        return 1
    fi
    local probe_dir status=0
    probe_dir="$(mktemp -d)" || return 1
    printf 'int main(void) { return 0; }\n' >"$probe_dir/probe.c"
    LLD_PROBE_OUTPUT="$(
        cc "-fuse-ld=$linker" "$probe_dir/probe.c" -o "$probe_dir/probe" 2>&1
    )" || status=1
    rm -rf "$probe_dir"
    return "$status"
}

# Say on stderr that $1 is installed but cannot link, and that this build is
# continuing with the platform linker.
report_lld_fallback() {
    local linker="$1"
    cat >&2 <<MSG
Note: $linker is installed but cannot link against the active SDK, so this
build is using the platform default linker instead. The result is correct,
just slower to link.

${LLD_PROBE_OUTPUT:-}

This is usually the SDK being newer than lld. When a newer lld is installed
(macOS: brew upgrade lld) it is used again automatically.
MSG
    return 0
}

# Find and probe in one step. Prints a usable `-fuse-ld=` value on stdout, or
# nothing, with the reason on stderr when lld is installed but unusable.
# Never fails the caller: an unusable lld is a slower build, not a broken one.
resolve_usable_lld() {
    local lld
    lld="$(find_lld)"
    [[ -n "$lld" ]] || return 0
    if lld_links "$lld"; then
        printf '%s\n' "$lld"
    else
        report_lld_fallback "$lld"
    fi
    return 0
}
