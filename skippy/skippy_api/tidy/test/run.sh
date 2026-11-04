#!/usr/bin/env bash
# run.sh <clang-tidy> <FileCheck> <plugin> <fixture.cpp> <stubs dir>
# A fixture names its checks on a `// CHECKS:` line, may override options on a
# `// CONFIG:` line (clang-tidy --config syntax), and pins diagnostics with
# FileCheck `// CHECK:` lines. Any warning without a CHECK line fails the test.
# A fixture with no CHECK lines is pure-negative: it must produce no output.
set -euo pipefail
tidy="$1"; filecheck="$2"; plugin="$3"; fixture="$4"; stubs="$5"
checks="$(sed -n 's|^// CHECKS: ||p' "$fixture")"
config="$(sed -n 's|^// CONFIG: ||p' "$fixture")"
out="$("$tidy" --load="$plugin" -checks="-*,${checks}" --config="${config}" --header-filter='.*' \
    "$fixture" -- -std=c++23 -I "$stubs" 2>/dev/null || true)"
if grep -Eq '^[[:space:]]*// CHECK:' "$fixture"; then
    printf '%s\n' "$out" \
      | "$filecheck" "$fixture" --check-prefix=CHECK --implicit-check-not='warning: ' --allow-empty
else
    if printf '%s\n' "$out" | grep -q 'warning:'; then
        printf '%s\n' "unexpected warnings in negative fixture $fixture:" "$out" >&2
        exit 1
    fi
fi
