#!/usr/bin/env bash
# Print the Skippy repository and commit a CI run should build against.
#   scripts/ci-resolve-skippy.sh [ref]
# With no ref: the revision pinned in Cargo.toml. A ref is a branch, a tag, a
# full commit SHA, or `pull/<number>` for a Skippy pull request.
# Output: two lines, `repository=<owner/name>` and `sha=<40 hex>`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pin_url="$(sed -n 's/^skippy-ffi = .*git = "\([^"]*\)".*/\1/p' "$REPO_ROOT/Cargo.toml")"
pin_rev="$(sed -n 's/^skippy-ffi = .*rev = "\([^"]*\)".*/\1/p' "$REPO_ROOT/Cargo.toml")"
[[ -n "$pin_url" && "$pin_rev" =~ ^[0-9a-f]{40}$ ]] || { echo "no Skippy pin in Cargo.toml" >&2; exit 1; }
repository="$(sed -E 's#^(ssh://git@|https://)github\.com/##; s#\.git$##' <<<"$pin_url")"

ref="${1:-}"
if [[ -z "$ref" ]]; then
    sha="$pin_rev"
elif [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then
    sha="$ref"
elif [[ "$ref" =~ ^[A-Za-z0-9._/-]+$ && "$ref" != *..* ]]; then
    [[ "$ref" =~ ^pull/[0-9]+$ ]] && ref="refs/pull/${ref#pull/}/head"
    sha="$(git ls-remote "$pin_url" "$ref" | awk 'NR == 1 {print $1}')"
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "Skippy ref not found: $ref" >&2; exit 1; }
else
    echo "invalid Skippy ref: $ref" >&2
    exit 2
fi
echo "repository=$repository"
echo "sha=$sha"
