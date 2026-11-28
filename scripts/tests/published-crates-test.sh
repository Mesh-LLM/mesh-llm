#!/usr/bin/env bash
# The shared roster must contain every published workspace crate exactly once.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
roster="$("$root/scripts/published-crates.sh")"
[[ -n "$roster" ]]
[[ "$(printf '%s\n' "$roster" | sort -u | wc -l | tr -d ' ')" == "$(printf '%s\n' "$roster" | wc -l | tr -d ' ')" ]]
if grep -Eq '^(mesh-llm|mesh-llm-ffi|mesh-llm-nodejs|mesh-llm-test-harness)$' <<<"$roster"; then
    echo "private crate entered the published roster" >&2
    exit 1
fi
grep -qx 'mesh-llm-sdk' <<<"$roster"
grep -qx 'mesh-llm-system' <<<"$roster"
echo 'published crate roster: ok'
