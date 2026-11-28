#!/usr/bin/env bash
# Print the Skippy repository and commit a CI run should build against.
#   scripts/ci-resolve-skippy.sh [ref]
# With no ref: the release commit pinned in scripts/skippy-pin.sh. A ref is a full commit SHA,
# `pull/<number>` for a Skippy pull request, a branch or tag name, or a full
# `refs/heads/...` or `refs/tags/...` name. A name that is both a branch and a
# tag is refused; use the full name.
# Output: two lines, `repository=<owner/name>` and `sha=<40 hex>`.
# SKIPPY_GIT_URL replaces the pinned repository URL (used by the tests).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pin="$("$REPO_ROOT/scripts/skippy-pin.sh")"
read -r pin_url pin_rev _ <<<"$pin"
repository="$(sed -E 's#^https://github\.com/##; s#\.git$##' <<<"$pin_url")"
url="${SKIPPY_GIT_URL:-$pin_url}"

# Commit an exactly named ref points at; empty when it does not exist.
# ls-remote matches name suffixes, so compare the full name. A peeled line
# (`<ref>^{}`) is the commit behind an annotated tag and wins over the tag.
commit_of() {
    git ls-remote "$url" "$1" "$1^{}" |
        awk -v ref="$1" '$2 == ref "^{}" {peeled = $1} $2 == ref {plain = $1} END {print peeled ? peeled : plain}'
}

ref="${1:-}"
if [[ -z "$ref" ]]; then
    sha="$pin_rev"
elif [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then
    sha="$ref"
elif [[ ! "$ref" =~ ^[A-Za-z0-9._/-]+$ || "$ref" == *..* ]]; then
    echo "invalid Skippy ref: $ref" >&2
    exit 2
elif [[ "$ref" =~ ^pull/[0-9]+$ ]]; then
    sha="$(commit_of "refs/$ref/head")"
elif [[ "$ref" == refs/heads/* || "$ref" == refs/tags/* ]]; then
    sha="$(commit_of "$ref")"
else
    branch="$(commit_of "refs/heads/$ref")"
    tag="$(commit_of "refs/tags/$ref")"
    if [[ -n "$branch" && -n "$tag" ]]; then
        echo "Skippy ref $ref is both a branch and a tag; use refs/heads/$ref or refs/tags/$ref" >&2
        exit 1
    fi
    sha="$branch$tag"
fi
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "Skippy ref not found: $ref" >&2; exit 1; }
echo "repository=$repository"
echo "sha=$sha"
