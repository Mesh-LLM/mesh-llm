#!/usr/bin/env bash

# Resolve an LLVM lld the current toolchain can actually link with.
#
# lld is a build-speed optimization (measured up to 26% faster locally), not a
# correctness requirement: the produced binaries are equivalent either way. It
# does, however, have to parse the active SDK's text-based stubs, and lld and
# the SDK are upgraded independently. When the SDK moves first, lld rejects
# `libSystem.tbd` outright and every libSystem symbol comes back undefined:
#
#   ld64.lld: error: could not load TAPI file at .../MacOSX.sdk/usr/lib/libSystem.tbd
#   libSystem.tbd:4:20: error: unknown target
#                      arm64e.x1-macos, arm64e.x1-maccatalyst ]
#
# That is an SDK-ahead-of-lld skew, so downgrading lld makes it worse and
# pinning an older SDK buys a version we then have to remember to unpin. The
# durable answer is to stop assuming: probe the resolved linker once, use it
# when it works, and fall back to the platform linker with a stated reason
# when it does not. When a toolchain that understands the newer SDK arrives,
# the probe starts passing and lld comes back with no further change.
#
# `cargo check` does not link, so a broken linker is invisible to it. Anything
# that trusts lld without probing fails at the first `cargo test` or `cargo
# build` instead, which is a much worse place to discover it.

# Print the path to an installed lld on stdout, or nothing when none is
# installed. Says nothing about whether it can actually link -- use
# `lld_is_usable` for that. Callers that want both in one step can use
# `resolve_usable_lld`.
find_lld() {
    case "$(uname -s)" in
        Linux)
            # The ELF driver takes the portable name; clang resolves it on PATH.
            command -v ld.lld >/dev/null 2>&1 && printf 'lld\n'
            ;;
        Darwin) find_lld_darwin ;;
        *) ;;
    esac
    return 0
}

find_lld_darwin() {
    local lld=""
    if command -v ld64.lld >/dev/null 2>&1; then
        # Portable driver name, not the absolute path: clang resolves
        # `-fuse-ld=lld` to `ld64.lld` on PATH, and some clang builds reject
        # an absolute `-fuse-ld=`. Either way the probe below decides.
        printf 'lld\n'
        return 0
    fi
    if command -v brew >/dev/null 2>&1; then
        local prefix
        prefix="$(brew --prefix lld 2>/dev/null || true)"
        [[ -n "$prefix" && -x "$prefix/bin/ld64.lld" ]] && lld="$prefix/bin/ld64.lld"
    fi
    if [[ -z "$lld" ]]; then
        for candidate in /opt/homebrew/opt/lld/bin/ld64.lld /usr/local/opt/lld/bin/ld64.lld; do
            if [[ -x "$candidate" ]]; then
                lld="$candidate"
                break
            fi
        done
    fi
    [[ -n "$lld" ]] && printf '%s\n' "$lld"
    return 0
}

# Link a two-line C program with `-fuse-ld=$1`. One clang invocation on a
# temporary directory; tens of milliseconds, once per command. Leaves the
# linker's own diagnostics in LLD_PROBE_OUTPUT so a caller can report the
# real reason rather than guessing at one.
lld_is_usable() {
    local linker="$1"
    command -v cc >/dev/null 2>&1 || return 1
    local probe_dir
    probe_dir="$(mktemp -d)" || return 1
    local status=0
    printf 'int main(void) { return 0; }\n' >"$probe_dir/probe.c"
    LLD_PROBE_OUTPUT="$(
        cc "-fuse-ld=$linker" "$probe_dir/probe.c" -o "$probe_dir/probe" 2>&1
    )" || status=1
    export LLD_PROBE_OUTPUT
    rm -rf "$probe_dir"
    return "$status"
}

# Explain, on stderr, that $1 is installed but cannot link, and that the
# build is continuing with the platform linker.
report_unusable_lld() {
    local linker="$1"
    cat >&2 <<EOF
Note: $linker is installed but cannot link against the active SDK, so this
build is using the platform default linker instead. Builds will be correct,
just slower.

${LLD_PROBE_OUTPUT:-}

This is usually an SDK-ahead-of-lld skew. Update lld (macOS: brew upgrade lld)
and it will be used again automatically -- no repository change is needed.
EOF
    return 0
}

# Find and probe in one step. Prints a usable lld on stdout, or nothing --
# with the reason on stderr when one is installed but unusable.
resolve_usable_lld() {
    local lld
    lld="$(find_lld)"
    [[ -n "$lld" ]] || return 0
    if lld_is_usable "$lld"; then
        printf '%s\n' "$lld"
    else
        report_unusable_lld "$lld"
    fi
    return 0
}
