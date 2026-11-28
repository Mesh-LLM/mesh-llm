#!/usr/bin/env bash
# Read the test-only Skippy-Ref directive. With --merge-ready, admit only a
# successful pinned run whose PR description still has no override.
set -euo pipefail

ref=""
seen=false
while IFS= read -r line; do
    line="${line%$'\r'}"
    if [[ "$line" =~ ^[[:space:]]*Skippy-Ref: ]]; then
        [[ "$seen" == false ]] || { echo 'Duplicate Skippy-Ref directives' >&2; exit 1; }
        seen=true
        read -r ref extra <<<"${line#*Skippy-Ref:}"
        [[ "$ref" =~ ^[A-Za-z0-9._/-]+$ && "$ref" != *..* && "$ref" != -* && -z "$extra" ]] || {
            echo 'Skippy-Ref needs exactly one branch, tag, full SHA or pull/N' >&2
            exit 1
        }
    fi
done <<<"${PR_BODY:-}"

case "${1:-}" in
    '') printf '%s\n' "$ref" ;;
    --merge-ready)
        [[ "${OVERRIDE:-}" == false && -z "$ref" ]] || {
            echo 'Skippy-Ref is test-only. Update the committed pin, remove the directive, and run fresh pinned CI before merging.' >&2
            exit 1
        }
        [[ "${RESULTS:-}" == 'success success success success success success success' ]] || {
            echo "CI did not fully pass: ${RESULTS:-missing results}" >&2
            exit 1
        }
        ;;
    *) echo "usage: $0 [--merge-ready]" >&2; exit 2 ;;
esac
