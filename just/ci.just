# ── Full Validation Gate ───────────────────────────────────────

# Populate the trusted Linux compiler-object seed with the dominant Clippy,
# release-host, and isolated test graphs used by high-fanout CI jobs.
[linux]
ci-sccache-seed-build:
    cargo clippy --locked -p mesh-llm --all-targets -- -D warnings
    cargo build --release --locked -p mesh-llm --bin mesh-llm --no-default-features --features web-ui,dynamic-native-runtime
    cargo test --locked -p mesh-llm-cli --no-run

# Validate CI definitions, planner fixtures, and repository consistency.
ci-validate:
    actionlint -config-file .github/actionlint.yaml
    git diff --check
    just ci-automation-contracts
    just ci-legacy-contracts
    just ci-crate-lists
    just check-release
    just no-console-print
    just publish-crates

# Explicit surviving transitional coverage; retire modules only with intent owners.
ci-legacy-contracts:
    python3 -m unittest scripts.tests.test_cuda_toolkit scripts.tests.test_prepare_llama scripts.tests.test_package_native_runtime scripts.tests.test_depot_authority_sentinel scripts.tests.test_generate_split_certified scripts.tests.test_skippy_llama_parity scripts.tests.test_ci_windows_composition scripts.tests.test_skippy_scheduler_fixtures scripts.tests.test_ci_workflow_artifacts scripts.tests.test_reusable_workflow_runner_trust scripts.tests.test_workload_lane_execution scripts.tests.test_depot_registry_canary_workflow scripts.tests.test_skippy_waiting_prefix_ab scripts.tests.test_build_accelerator_defaults scripts.tests.test_build_windows scripts.tests.test_skippy_radix_cache_ab scripts.tests.test_agentic_replay_nightly_workflow scripts.tests.test_ci_pr_canary_workflow scripts.tests.test_install_sh scripts.tests.test_ci_lane_workflows scripts.tests.test_ci_product_composer scripts.tests.test_skippy_native_evidence_contract scripts.tests.test_run_event_benchmark_matrix scripts.tests.test_ci_two_node_split_smoke scripts.tests.test_ci_native_toolchain_routing scripts.tests.test_ci_prepare_native_runtime scripts.tests.test_skippy_competitive_benchmark scripts.tests.test_release_version_script scripts.tests.test_pr_workflow_artifacts scripts.tests.test_hf_skippy_mtp_certify_job scripts.tests.test_ci_source_layout scripts.tests.test_ci_openai_embeddings_smoke scripts.tests.test_resume_crates_release_workflow scripts.tests.test_compare_event_benchmark_matrix scripts.tests.test_install_ps1 scripts.tests.test_hf_xet_portability_smoke scripts.tests.test_kv_restart_replay scripts.tests.test_llama_canary_agent_repair_contract scripts.tests.test_agentic_trajectory_manifest scripts.tests.test_llama_upstream_canary_contract scripts.tests.test_llama_review_regressions scripts.tests.test_skippy_adaptive_prefill_ab scripts.tests.test_skippy_agentic_prompt_manifest scripts.tests.test_skippy_mixed_prefill_decode_ab scripts.tests.test_ci_sdk_producers scripts.tests.test_llama_native_full_replay scripts.tests.test_ci_artifact_actions scripts.tests.test_depot_canary_workflow scripts.tests.test_skippy_system_one_smoke scripts.tests.test_skippy_static_link scripts.tests.test_justfile_release_runtime scripts.tests.test_skippy_dynamic_link scripts.tests.test_release_workflow_artifacts

# Run CI/workspace crate-list consistency checks.
ci-crate-lists:
    just with-lld cargo run -p xtask -- repo-consistency ci-crate-lists

# Build only the xtask automation tool, then prove it is Cargo's target output
# with no product/native/UI dependency (no llama.cpp or UI preparation).
automation-bootstrap:
    just with-lld cargo build --locked -p xtask --bin xtask
    just with-lld cargo run --locked -p xtask --bin xtask -- automation bootstrap

[private]
[unix]
[positional-arguments]
automation-run *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    bootstrap="$(just automation-bootstrap)"
    binary="$(printf '%s\n' "$bootstrap" | awk -F= '$1 == "binary_path" { print substr($0, index($0, "=") + 1) }')"
    test -n "$binary"
    exec "$binary" "$@"

# Run crates.io publish-chain consistency checks.
publish-crates:
    just with-lld cargo run -p xtask -- repo-consistency publish-crates

# Gate on println!/eprintln!/print!/eprint! and direct io::stdout()/io::stderr()
# handles in Rust files under crates/ (excluding tests, non-product crates, helper
# bins and build.rs). There is no allowlist: route output through mesh-llm-events.
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
    just with-lld cargo check -p skippy-gpu-bench
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
        --exclude skippy-package-builder \
        --exclude skippy-runtime \
        --exclude skippy-serving \
        --exclude skippy-cli
    echo "--- Static Skippy package-builder tests ---"
    just with-lld cargo test --package skippy-package-builder --no-default-features
    echo "--- Static Skippy runtime tests ---"
    just with-lld cargo test --package skippy-runtime --no-default-features --lib
    echo "--- Static Skippy server tests ---"
    just with-lld cargo test --package skippy-serving --no-default-features --lib
    echo ""
    echo "--- Static Skippy CLI tests ---"
    just with-lld cargo test --package skippy-cli --no-default-features
    echo "=== 6/11 Plugin author exemplar ==="
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) echo "--- Installed OpenAI exchange conformance is Unix-only; skipping on Windows ---" ;;
        *) just test-openai-exchange-conformance ;;
    esac
    just with-lld cargo run --quiet --manifest-path mesh/docs/plugins/exemplars/web-ui/Cargo.toml -- --print-package-manifest > target/web-ui-exemplar-manifest.json
    diff -u <(jq -S . mesh/docs/plugins/exemplars/web-ui/plugin.package.json) <(jq -S . target/web-ui-exemplar-manifest.json)
    node --check mesh/docs/plugins/exemplars/web-ui/bundle/register-mesh-plugin-ui.js
    (cd "{{ ui_dir }}" && pnpm exec tsc --ignoreConfig --noEmit --target ES2022 --module ESNext --moduleResolution Bundler --lib ES2022,DOM ../../docs/plugins/exemplars/web-ui/bundle/register-mesh-plugin-ui.ts)
    echo ""
    echo "=== 7-10/11 Parallel portable checks and builds ==="
    scripts/test-portable.sh
    echo ""
    echo "=== 11/11 E2E smoke tests (Playwright) ==="
    (cd "{{ ui_dir }}" && pnpm run test:e2e)
    echo ""
    echo "All checks passed."

# Portable automation contracts replace accepted legacy test intentions.
# Build only xtask and its inert fixture binaries/examples before serial tests.
ci-automation-contracts:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v node >/dev/null 2>&1 || { echo "ci-automation-contracts requires the Node component for finite local action fixtures; install Node before this required gate." >&2; exit 1; }
    bootstrap="$(just automation-bootstrap)"
    binary="$(printf '%s\n' "$bootstrap" | awk -F= '$1 == "binary_path" { count++; print substr($0, index($0, "=") + 1) } END { if (count != 1) exit 1 }')"
    target_directory="$(printf '%s\n' "$bootstrap" | awk -F= '$1 == "target_directory" { count++; print substr($0, index($0, "=") + 1) } END { if (count != 1) exit 1 }')"
    test -n "$binary"
    test -n "$target_directory"
    export CARGO_TARGET_DIR="$target_directory"
    just with-lld cargo build --locked -p xtask --bins --examples
    fixture_suffix=""
    case "$binary" in *.exe) fixture_suffix=".exe" ;; esac
    export SPV_ADAPTER_FIXTURE="$target_directory/debug/examples/swift_privacy_lint_fixture$fixture_suffix"
    git_path="${MIGRATION_TEST_GIT-$(command -v git)}"
    if command -v cygpath >/dev/null 2>&1; then
        git_path="$(cygpath -w "$git_path")"
        test -f "$(cygpath -u "$SPV_ADAPTER_FIXTURE")"
        test -x "$(cygpath -u "$SPV_ADAPTER_FIXTURE")"
    else
        test -f "$SPV_ADAPTER_FIXTURE"
        test -x "$SPV_ADAPTER_FIXTURE"
    fi
    export MIGRATION_TEST_GIT="$git_path"
    export GIT_MASTER=1 GIT_OPTIONAL_LOCKS=0
    just ci-snapshot-promotion-contracts
    just with-lld cargo test --locked -p xtask --bin xtask \
        --test automation_selector \
        --test canary_cache_configuration \
        --test canary_controller_plan \
        --test canary_result_gate \
        --test ci_runner_cache_contract \
        --test family_battery_environment \
        --test family_battery_prepared_source \
        --test l4_producers \
        --test lane_l11_optional_tooling \
        --test lane_l12_build_llama \
        --test lane_l1_ui_delivery \
        --test lane_l2_authority \
        --test lane_l2_hosted_bootstrap \
        --test lane_l2_quality_contracts \
        --test lane_l2_registry_pulls \
        --test lane_l2_runtime_lifecycle \
        --test lane_l2_runtime_seed \
        --test lane_l3_abi_restore \
        --test lane_l3_cache_metadata \
        --test lane_l3_display_projection \
        --test lane_l3_l2_dispatch \
        --test lane_l5_adapter_contracts \
        --test lane_l5_release_product \
        --test lane_l6_sdk_swift \
        --test lane_l7_control_plane \
        --test lane_l7_daemon_lifecycle \
        --test lane_l7_logging_console \
        --test lane_l7_logging_recovery \
        --test lane_l7_plans \
        --test lane_l7_sdk_fixture \
        --test lane_l7_split_recovery \
        --test lane_l7_standalone \
        --test lane_l8_smoke \
        --test laya_cli_fixture \
        --test local_ports \
        --test migration_archives \
        --test migration_bootstrap \
        --test migration_canary_aggregate_cli \
        --test migration_canary_receipts \
        --test migration_canary_receipts_parity \
        --test migration_cargo_metadata \
        --test migration_cargo_packages_cli \
        --test migration_ci_graph \
        --test migration_ci_operations \
        --test migration_ci_plan \
        --test migration_cli \
        --test migration_family_plan \
        --test migration_hf_artifact_cli \
        --test migration_hf_artifact_controls \
        --test migration_hf_converted_artifact \
        --test migration_lifecycle \
        --test migration_stability \
        --test migration_models \
        --test migration_native_fixtures \
        --test migration_native_generator \
        --test migration_native_policy \
        --test migration_optional_isolation \
        --test migration_package_source_version \
        --test migration_patch_bytes \
        --test migration_patch_shards \
        --test migration_prepared_inputs \
        --test migration_process \
        --test migration_product \
        --test migration_qualification_cli \
        --test migration_release \
        --test migration_report_contract \
        --test migration_report_safety \
        --test migration_repository \
        --test migration_run_family \
        --test migration_workload_oracle_cli \
        --test migration_xcframework_cli \
        --test openai_smoke_config \
        --test opencode_config_adapter \
        --test opencode_fixture_evidence \
        --test prepared_source \
        --test release_inventory_evidence \
        --test release_inventory_provenance \
        --test release_inventory_publication \
        --test replay_acceptance_gates \
        --test replay_arm_pass \
        --test replay_build_arm \
        --test replay_context_preflight \
        --test replay_external_engines \
        --test replay_external_run \
        --test replay_history \
        --test replay_history_hub \
        --test replay_interruption \
        --test replay_l3 \
        --test replay_manifest_preflight \
        --test replay_manual_profile \
        --test replay_manual_resume_refs \
        --test replay_model_preflight \
        --test replay_pass_schedule \
        --test replay_pooled_rows \
        --test replay_progress \
        --test replay_publication_prepare \
        --test replay_report \
        --test replay_retirement \
        --test replay_run_execution \
        --test replay_server_cell \
        --test replay_session_evidence \
        --test replay_trajectory_execution \
        --test swift_api_checksum \
        --test swift_privacy_adapter \
        --test swift_producer_environment \
        --test workload_certify_contract \
        --test workload_oracle_producer_contract \
        --test workload_smoke_config \
        --test workload_start_retry \
        -- --test-threads=1

# Native component owner and actual CLI, including the embedded job caller.
ci-snapshot-promotion-contracts:
    just with-lld cargo test --locked -p model-package --lib --test snapshot_promotion_cli -- --test-threads=1
