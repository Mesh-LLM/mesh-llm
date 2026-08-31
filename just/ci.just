# ── Full Validation Gate ───────────────────────────────────────

# Populate the trusted Linux compiler-object seed with the dominant host graph.
[linux]
ci-sccache-seed-build:
    cargo clippy --locked -p mesh-llm --all-targets -- -D warnings

# Validate CI definitions, planner fixtures, and repository consistency.
ci-validate:
    actionlint -config-file .github/actionlint.yaml
    git diff --check
    python3 -m unittest discover -s scripts/tests -p 'test_*.py'
    just ci-crate-lists
    just check-release
    just no-console-print
    just publish-crates

# Run CI/workspace crate-list consistency checks.
ci-crate-lists:
    just with-lld cargo run -p xtask -- repo-consistency ci-crate-lists

# Run crates.io publish-chain consistency checks.
publish-crates:
    just with-lld cargo run -p xtask -- repo-consistency publish-crates

# Ratchet on println!/eprintln!/print!/eprint! in Rust files under crates/
# (including crate test targets; build.rs is skipped): every occurrence must be
# explicitly listed in tools/xtask/data/console_print_allowlist.json.
no-console-print:
    cargo run -p xtask -- repo-consistency no-console-print

# Shellcheck the explicitly supplied changed shell scripts.
ci-shellcheck *scripts:
    shellcheck {{ scripts }}

# Run all checks: repo consistency, Rust tests, author exemplars, fmt, clippy, UI/docs builds, and E2E smoke.
test-all:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "=== 0/11 Test-all rust crate coverage preflight ==="
    just with-lld cargo run -p xtask -- repo-consistency test-all-rust-crate-coverage

    # A full workspace gate otherwise leaves hundreds of incompatible
    # incremental feature graphs behind. CI already builds non-incrementally;
    # use the same bounded-disk behavior here unless explicitly overridden.
    export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
    dynamic_target_dir="${MESH_TEST_ALL_DYNAMIC_TARGET_DIR:-$PWD/target/test-all-dynamic}"

    native_backend="${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-${LLAMA_BACKEND:-}}}"
    if [[ -z "$native_backend" ]]; then
        case "$(uname -s)" in
            Darwin) native_backend="metal" ;;
            *) native_backend="cpu" ;;
        esac
    fi
    export LLAMA_STAGE_BACKEND="$native_backend"

    if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
        LLAMA_STAGE_BUILD_DIR="$(scripts/build-llama.sh --print-build-dir)"
        export LLAMA_STAGE_BUILD_DIR
    fi

    echo "=== Native llama.cpp ABI ($LLAMA_STAGE_BACKEND) ==="
    echo "Build dir: $LLAMA_STAGE_BUILD_DIR"
    scripts/prepare-llama.sh
    scripts/build-llama.sh
    echo ""

    # Each UI step runs in a subshell so cd doesn't leak between steps.
    echo "=== 1/11 Repo consistency ==="
    just with-lld cargo run -p xtask -- repo-consistency ci-crate-lists
    just with-lld cargo run -p xtask -- repo-consistency publish-crates
    echo ""
    echo "=== 2/11 Rust format check ==="
    just with-lld cargo fmt --all -- --check
    echo ""
    echo "=== 3/11 GPU bench crate check ==="
    just with-lld cargo check -p mesh-llm-gpu-bench
    echo ""
    echo "=== 4-5/11 Rust validation ==="
    # Keep Clippy and tests adjacent for each compatible feature graph. Switching
    # dynamic -> static -> dynamic -> static forces Cargo to relink both graphs.
    echo "--- Dynamic-runtime bindings: Clippy ---"
    CARGO_TARGET_DIR="$dynamic_target_dir" just with-lld cargo clippy \
        -p mesh-llm-ffi \
        -p mesh-llm-nodejs \
        -p skippy-ffi \
        -p skippy-quantize \
        --all-targets -- -D warnings
    echo "--- Dynamic-runtime bindings: tests ---"
    CARGO_TARGET_DIR="$dynamic_target_dir" just with-lld cargo test \
        -p mesh-llm-ffi \
        -p mesh-llm-nodejs \
        -p skippy-ffi \
        -p skippy-quantize
    echo "--- Static development workspace: Clippy ---"
    just with-lld cargo clippy --workspace --all-targets \
        --exclude mesh-llm-ffi \
        --exclude mesh-llm-nodejs \
        --exclude skippy-ffi \
        --exclude skippy-quantize \
        -- -D warnings
    echo "--- Static development workspace: tests ---"
    just with-lld cargo test --workspace \
        --exclude mesh-llm-ffi \
        --exclude mesh-llm-nodejs \
        --exclude skippy-ffi \
        --exclude skippy-quantize \
        --exclude skippy-runtime \
        --exclude skippy-server
    echo "--- Static Skippy runtime tests ---"
    just with-lld cargo test --package skippy-runtime --no-default-features --lib
    echo "--- Static Skippy server tests ---"
    just with-lld cargo test --package skippy-server --no-default-features --lib
    echo ""
    echo "=== 6/11 Plugin author exemplar ==="
    just with-lld cargo run --quiet --manifest-path docs/plugins/exemplars/web-ui/Cargo.toml -- --print-package-manifest > target/web-ui-exemplar-manifest.json
    diff -u <(jq -S . docs/plugins/exemplars/web-ui/plugin.package.json) <(jq -S . target/web-ui-exemplar-manifest.json)
    node --check docs/plugins/exemplars/web-ui/bundle/register-mesh-plugin-ui.js
    (cd "{{ ui_dir }}" && pnpm exec tsc --ignoreConfig --noEmit --target ES2022 --module ESNext --moduleResolution Bundler --lib ES2022,DOM ../../docs/plugins/exemplars/web-ui/bundle/register-mesh-plugin-ui.ts)
    echo ""
    echo "=== 7-10/11 Parallel portable checks and builds ==="
    scripts/test-portable.sh
    echo ""
    echo "=== 11/11 E2E smoke tests (Playwright) ==="
    (cd "{{ ui_dir }}" && pnpm run test:e2e)
    echo ""
    echo "All checks passed."
