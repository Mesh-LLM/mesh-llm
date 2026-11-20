#!/usr/bin/env bash
# Tests for scripts/ci-resolve-skippy.sh against a local Git repository.
set -euo pipefail

resolve="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ci-resolve-skippy.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
repo="$work/skippy"
git init -q -b main "$repo"
g() { git -C "$repo" -c user.name=test -c user.email=test@example.invalid "$@"; }
commit() { g commit -q --allow-empty -m "$1" && g rev-parse HEAD; }

main="$(commit main)"
g checkout -q -b feature/main
feature="$(commit feature)"
g checkout -q -b both
both_branch="$(commit both)"
g tag both "$main"
g tag -a v1 -m v1 "$main"
g update-ref refs/pull/7/head "$feature"
g checkout -q main
export SKIPPY_GIT_URL="$repo"

failures=0
expect() { # expect <want-sha-or-FAIL> <ref>
    local got
    got="$("$resolve" "$2" 2>/dev/null | sed -n 's/^sha=//p')" || got=FAIL
    if [[ "$got" != "$1" ]]; then
        echo "FAIL: '$2' resolved to '$got', want '$1'" >&2
        failures=$((failures + 1))
    fi
}
expect "$main" main                 # not feature/main, which ends in the same name
expect "$feature" feature/main
expect "$main" refs/heads/main
expect "$main" v1                   # the commit, not the annotated tag object
expect "$main" refs/tags/v1
expect "$feature" pull/7
expect "$feature" "$feature"        # a full SHA is used as given
expect FAIL both                    # a branch and a tag share the name
expect "$both_branch" refs/heads/both
expect "$main" refs/tags/both
expect FAIL missing
expect FAIL pull/8
expect FAIL 'a;b'
# shellcheck disable=SC2016
expect FAIL '$(id)'
expect FAIL 'a..b'
expect FAIL '--upload-pack=x'
[[ "$(SKIPPY_GIT_URL='' "$resolve" | sed -n 's/^sha=//p')" =~ ^[0-9a-f]{40}$ ]] || { echo "FAIL: pinned revision" >&2; failures=$((failures + 1)); }
((failures == 0)) || exit 1
echo "ci-resolve-skippy: ok"
