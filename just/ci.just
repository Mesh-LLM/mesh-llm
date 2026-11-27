# Checks run by CI and before a pull request. Every recipe is a plain command;
# nothing here builds a helper tool first.

# Format, lint, and shell checks.
lint:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    shellcheck scripts/*.sh scripts/tests/*.sh mesh/scripts/*.sh
    scripts/tests/skippy-pin-test.sh
    scripts/tests/skippy-source-test.sh
    scripts/tests/build-skippy-runtime-test.sh
    scripts/tests/fetch-skippy-runtime-test.sh
    scripts/tests/ci-resolve-skippy-test.sh
    scripts/tests/ci-skippy-runtime-test.sh
    python3 scripts/tests/ci-cpu-host-smoke-test.py
    scripts/tests/install-asset-test.sh
    scripts/tests/check-host-imports-test.sh
    scripts/tests/wait-packaging-receipt-test.sh
    scripts/tests/package-runtime-platform-test.sh
    scripts/tests/native-sdk-package-test.sh
    scripts/tests/build-ui-staleness-test.sh
    scripts/tests/published-crates-test.sh
    scripts/tests/ci-website-test.sh
    scripts/tests/create-compatibility-aliases-test.sh

# Rust tests for the whole workspace.
test-rust *ARGS:
    cargo test --workspace {{ ARGS }}

# Console lint, type check, unit tests, and build.
test-ui:
    cd "{{ ui_dir }}" && pnpm install --frozen-lockfile && pnpm run validate

# Real inference through the MeshLLM host on Skippy's CPU runtime.
# `runtime` is a directory of packaged Skippy runtimes; `model` is a GGUF file.
[unix]
smoke runtime model:
    @scripts/ci-smoke-test.sh "{{ runtime }}" "{{ model }}"

# The tests that call into a native runtime, which `test-rust` leaves ignored.
# `runtime` is a directory of packaged Skippy runtimes.
[unix]
test-with-runtime runtime:
    SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR="$(cd "{{ runtime }}" && pwd -P)" cargo test -p mesh-llm-system --features dynamic-native-runtime --lib test_skippy_backend -- --ignored

# Everything a pull request must pass, except the smoke (it needs a runtime and a model).
test-all: lint test-rust test-ui
