#!/usr/bin/env bash
# Print the crates.io roster in dependency order for publishing and docs.
set -euo pipefail

(($# == 0)) || { echo "usage: $0" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
metadata="$(cd "$root" && cargo metadata --format-version 1 --no-deps --locked)"
not_released=(mesh-llm mesh-llm-ffi mesh-llm-nodejs mesh-llm-test-harness)
publishable="$(jq -r --args \
    '.packages[] | select(.publish != []) | .name | select(IN($ARGS.positional[]) | not)' \
    "${not_released[@]}" <<<"$metadata")"
[[ -n "$publishable" ]] || { echo "cargo metadata returned no published crates" >&2; exit 1; }
dep_pairs="$(jq -r --arg names "$publishable" '
    ($names | split("\n")) as $names
    | (.packages | map({key: (.manifest_path | sub("/Cargo.toml$"; "")), value: .name}) | from_entries) as $dirs
    | .packages[] | select(.name | IN($names[])) | .name as $name
    | .dependencies[] | select(.kind != "dev" and .path != null)
    | $dirs[.path] // empty | select(IN($names[])) | "\(.) \($name)"
' <<<"$metadata")"
{
    printf '%s\n' "$publishable" | awk '{ print $1, $1 }'
    [[ -z "$dep_pairs" ]] || printf '%s\n' "$dep_pairs"
} | tsort
