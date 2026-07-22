#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

check_logs="$(mktemp -d "${TMPDIR:-/tmp}/mesh-test-all.XXXXXX")"
cleanup_check_logs() {
    rm -rf "$check_logs"
}
trap cleanup_check_logs EXIT

ui_lint() {
    cd "$ROOT/crates/mesh-llm-ui"
    pnpm run lint
}

ui_typecheck_and_build() {
    cd "$ROOT/crates/mesh-llm-ui"
    pnpm run typecheck
    # TypeScript already passed above, so do not repeat it through `pnpm run build`.
    pnpm exec vite build
}

ui_unit_tests() {
    cd "$ROOT/crates/mesh-llm-ui"
    pnpm test
}

website_build() {
    cd "$ROOT/website"
    npm run build
}

script_and_sdk_tests() {
    python3 -m unittest discover -s scripts/tests -p 'test_*.py'
    node --test scripts/console-format.test.js
    npm test --prefix sdk/node
    scripts/check-sdk-contract.sh

    if command -v java >/dev/null 2>&1 && java -version >/dev/null 2>&1; then
        (cd sdk/kotlin && ./gradlew test --no-daemon)
    else
        echo "SKIP Kotlin SDK tests: Java runtime is not installed."
    fi

    if command -v swift >/dev/null 2>&1 \
        && [[ -d sdk/swift/Generated/MeshLLMFFI.xcframework ]]; then
        swift test
    else
        echo "SKIP Swift SDK tests: local MeshLLMFFI.xcframework is not built."
    fi
}

check_names=(ui-lint ui-typecheck-build ui-unit website-build script-sdk)
check_commands=(ui_lint ui_typecheck_and_build ui_unit_tests website_build script_and_sdk_tests)
check_pids=()

for i in "${!check_names[@]}"; do
    "${check_commands[$i]}" >"$check_logs/${check_names[$i]}.log" 2>&1 &
    check_pids+=("$!")
done

check_failed=0
for i in "${!check_names[@]}"; do
    if wait "${check_pids[$i]}"; then
        echo "PASS ${check_names[$i]}"
    else
        echo "FAIL ${check_names[$i]}"
        check_failed=1
    fi
    sed "s/^/[${check_names[$i]}] /" "$check_logs/${check_names[$i]}.log"
done

if [[ "$check_failed" != "0" ]]; then
    exit 1
fi
