#!/usr/bin/env bash
# MeshLLM builds against the Skippy crates.io release pinned in Cargo.toml and
# Cargo.lock. This script points the build at a local Skippy checkout instead,
# for work that spans both repositories.
#
#   scripts/skippy-source.sh              show what the build uses
#   scripts/skippy-source.sh <checkout>   build against that checkout or worktree
#   scripts/skippy-source.sh --ref <ref> [repository-url]
#                                         fetch that Skippy commit into
#                                         .deps/skippy-ref and build against it.
#                                         <ref> is a branch, tag, full commit
#                                         SHA or pull/<number>; the URL names a
#                                         fork (default: the pinned repository)
#   scripts/skippy-source.sh --pinned     go back to the pinned crates.io release
#
# The override is .cargo/skippy-local.toml (not committed). A checkout at a
# different release temporarily changes the Skippy versions in Cargo.toml.
# Build without --locked; build-host.sh does that for you. Do not commit the
# overridden Cargo.toml or Cargo.lock.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERRIDE="$REPO_ROOT/.cargo/skippy-local.toml"
LOCK_BACKUP="$REPO_ROOT/.cargo/skippy-pinned.lock"
pin="$("$REPO_ROOT/scripts/skippy-pin.sh")"
read -r pin_url pin_rev pin_version <<<"$pin"

set_skippy_versions() {
    local version="$1" temp
    temp="$(mktemp "$REPO_ROOT/.cargo/skippy-manifest.XXXXXX")"
    awk -v version="$version" '
        /^(hf_hub|skippy-[a-z-]+) = \{/ {
            sub(/version = "=[^"]+"/, "version = \"=" version "\"")
        }
        { print }
    ' "$REPO_ROOT/Cargo.toml" >"$temp"
    cat "$temp" >"$REPO_ROOT/Cargo.toml"
    rm -f "$temp"
}

if (($# == 0)); then
    if [[ -f "$OVERRIDE" ]]; then
        dir="$(sed -n 's/^# checkout: //p' "$OVERRIDE")"
        echo "local $dir $(git -C "$dir" rev-parse HEAD)$(git -C "$dir" diff --quiet HEAD -- 2>/dev/null || echo ' (modified)')"
    else
        echo "pinned $pin_url $pin_rev"
    fi
    exit 0
fi

if [[ "$1" == "--pinned" ]]; then
    if [[ ! -f "$OVERRIDE" ]]; then
        echo "already using pinned Skippy v$pin_version ($pin_rev)"
        exit 0
    fi
    [[ -f "$LOCK_BACKUP" ]] || {
        echo "cannot restore pinned Cargo.lock: $LOCK_BACKUP is missing" >&2
        exit 1
    }
    set_skippy_versions "$pin_version"
    rm -f "$OVERRIDE"
    cp "$LOCK_BACKUP" "$REPO_ROOT/Cargo.lock"
    rm -f "$LOCK_BACKUP"
    echo "using pinned Skippy v$pin_version ($pin_rev)"
    exit 0
fi

if [[ "$1" == "--ref" ]]; then
    (($# == 2 || $# == 3)) || { echo "usage: $0 --ref <ref> [repository-url]" >&2; exit 2; }
    ref_url="${3:-$pin_url}"
    sha="$(SKIPPY_GIT_URL="$ref_url" "$REPO_ROOT/scripts/ci-resolve-skippy.sh" "$2" | sed -n 's/^sha=//p')"
    checkout="$REPO_ROOT/.deps/skippy-ref"
    mkdir -p "$checkout"
    [[ -d "$checkout/.git" ]] || git -C "$checkout" init -q
    git -C "$checkout" fetch -q --depth 1 "$ref_url" "$sha"
    git -C "$checkout" checkout -q --detach FETCH_HEAD
    set -- "$checkout"
fi

source_dir="$(cd "$1" && pwd -P)"
cargo_source_dir="$source_dir"
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*)
        # Cargo is a native Windows executable and does not apply Git Bash's
        # /c/... path translation to paths read from TOML.
        cargo_source_dir="$(cygpath -m "$source_dir")"
        ;;
esac
case "$source_dir" in
    "$REPO_ROOT"/.deps/* | "$REPO_ROOT"/target/*) ;;
    "$REPO_ROOT"/*)
        echo "put an in-repository Skippy checkout under .deps/ or target/ so Cargo uses Skippy's own workspace version" >&2
        exit 1
        ;;
esac
[[ -f "$source_dir/crates/skippy-ffi/Cargo.toml" ]] || {
    echo "$source_dir is not a Skippy checkout (crates/skippy-ffi/Cargo.toml is missing)" >&2
    exit 1
}
source_version="$(sed -n 's/^version = "\([0-9][0-9.]*\)"/\1/p' "$source_dir/Cargo.toml" | head -1)"
[[ -n "$source_version" ]] || { echo "no Skippy release version in $source_dir/Cargo.toml" >&2; exit 1; }
[[ -f "$OVERRIDE" ]] || cp "$REPO_ROOT/Cargo.lock" "$LOCK_BACKUP"
set_skippy_versions "$source_version"
{
    echo "# checkout: $source_dir"
    echo "[patch.crates-io]"
    for manifest in "$source_dir"/crates/*/Cargo.toml; do
        name="$(sed -n 's/^name = "\(.*\)"/\1/p' "$manifest" | head -1)"
        echo "$name = { path = \"$cargo_source_dir/crates/$name\" }"
    done
} >"$OVERRIDE"
"$REPO_ROOT/scripts/verify-skippy-source.sh" "$source_dir"
echo "building against $source_dir ($(git -C "$source_dir" rev-parse HEAD))"
