#!/usr/bin/env bash
set -euo pipefail
script="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ci-skippy-ref.sh"
export PR_BODY='' OVERRIDE=false RESULTS='success success success success success success success'
expect_ref() {
    local got
    got="$(PR_BODY="$2" "$script" 2>/dev/null)" || got=FAIL
    [[ "$got" == "$1" ]] || { echo "Expected '$1', got '$got'" >&2; exit 1; }
}
expect_ref '' 'An ordinary PR'
expect_ref main $'Description\nSkippy-Ref: main\nMore text'
expect_ref pull/123 $'  Skippy-Ref: pull/123\r\n'
expect_ref refs/heads/feature 'Skippy-Ref: refs/heads/feature'
expect_ref v0.80.1 'Skippy-Ref: v0.80.1'
expect_ref 0123456789012345678901234567890123456789 'Skippy-Ref: 0123456789012345678901234567890123456789'
expect_ref FAIL 'Skippy-Ref:'
expect_ref FAIL 'Skippy-Ref: main extra'
expect_ref FAIL $'Skippy-Ref: main\nSkippy-Ref: main'
expect_ref FAIL 'Skippy-Ref: --upload-pack=x'
expect_ref FAIL 'Skippy-Ref: a..b'
# shellcheck disable=SC2016
expect_ref FAIL 'Skippy-Ref: $(id)'

reject() {
    if "$@" "$script" --merge-ready >/dev/null 2>&1; then
        echo "Merge guard unexpectedly passed: $*" >&2
        exit 1
    fi
}
"$script" --merge-ready
reject env OVERRIDE=true PR_BODY='Skippy-Ref: main'
# Removing the directive does not turn an override run into pinned evidence.
reject env OVERRIDE=true PR_BODY=''
# Adding a directive while an older pinned run is finishing must fail it.
reject env OVERRIDE=false PR_BODY='Skippy-Ref: main'
reject env OVERRIDE=false PR_BODY='Skippy-Ref:'
reject env OVERRIDE=''
for result in failure cancelled skipped ''; do
    reject env RESULTS="success success $result success success success success"
done
# Newly required website and platform smoke jobs cannot be missing or skipped.
reject env RESULTS='success success success success success'
reject env RESULTS='success success success success success success skipped'
reject env RESULTS='success success success success success success failure'
# An override still explains the test-only rejection with its platform smokes skipped.
message="$(OVERRIDE=true RESULTS='success success success success success success skipped' "$script" --merge-ready 2>&1)" && exit 1
[[ "$message" == *'Skippy-Ref is test-only'* ]]
# Fresh complete pinned run after removing the directive passes.
"$script" --merge-ready

echo 'ci-skippy-ref: ok'
