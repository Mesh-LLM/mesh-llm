# Checks run by CI and before a pull request. Every recipe is a plain command;
# nothing here builds a helper tool first.

# Format, lint, and shell checks.
lint:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    shellcheck scripts/*.sh scripts/tests/*.sh mesh/scripts/*.sh
    scripts/tests/ci-resolve-skippy-test.sh
    scripts/tests/ci-skippy-runtime-test.sh
    scripts/tests/install-asset-test.sh

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

# Everything a pull request must pass, except the smoke (it needs a runtime and a model).
test-all: lint test-rust test-ui
