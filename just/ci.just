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
    cargo test -p xtask --test migration_ci_operations --test migration_prepared_inputs --test migration_models --test lane_l3_l2_dispatch --test lane_l3_display_projection --test lane_l3_abi_restore --test lane_l3_cache_metadata
    just ci-legacy-contracts
    just ci-crate-lists
    just check-release
    just no-console-print
    just publish-crates

# Explicit surviving transitional coverage; retire modules only with intent owners.
ci-legacy-contracts:
    python3 -m unittest scripts.tests.test_cuda_toolkit scripts.tests.test_ci_runner_selection_and_cache_policy scripts.tests.test_prepare_llama scripts.tests.test_publish_crates scripts.tests.test_llama_oracle_source scripts.tests.test_package_native_runtime scripts.tests.test_depot_authority_sentinel scripts.tests.test_generate_split_certified scripts.tests.test_skippy_llama_parity scripts.tests.test_ci_compute_changes_expression_limit scripts.tests.test_ci_windows_composition scripts.tests.test_skippy_scheduler_fixtures scripts.tests.test_configure_canary_cache scripts.tests.test_ci_workflow_artifacts scripts.tests.test_llama_canary_handoff_contract scripts.tests.test_workload_monolithic_oracle scripts.tests.test_safe_extract_tar scripts.tests.test_agentic_replay scripts.tests.test_reusable_workflow_runner_trust scripts.tests.test_pr_builds_summary scripts.tests.test_qa_agent_tool_call_reliability scripts.tests.test_static_abi_artifacts scripts.tests.test_promote_layer_package_snapshot scripts.tests.test_runtime_events_native_gate scripts.tests.test_performance_history scripts.tests.test_workload_lane_execution scripts.tests.test_depot_registry_canary_workflow scripts.tests.test_skippy_waiting_prefix_ab scripts.tests.test_claude_compatibility_workflow scripts.tests.test_build_accelerator_defaults scripts.tests.test_build_ui scripts.tests.test_build_windows scripts.tests.test_skippy_workload_oracles_build scripts.tests.test_skippy_radix_cache_ab scripts.tests.test_kv_nightly_workflow_contract scripts.tests.test_sccache_evidence scripts.tests.test_agentic_replay_engines scripts.tests.test_compute_changes_justfiles scripts.tests.test_generate_native_runtime_release_manifest scripts.tests.test_ci_batch_crate_filter scripts.tests.test_download_skippy_parity_candidates scripts.tests.test_agentic_replay_nightly_workflow scripts.tests.test_ci_pr_canary_workflow scripts.tests.test_install_sh scripts.tests.test_ci_lane_workflows scripts.tests.test_build_linux scripts.tests.test_skippy_tts_oracle scripts.tests.test_ci_product_composer scripts.tests.test_justfile_shell_portability scripts.tests.test_skippy_native_evidence_contract scripts.tests.test_run_event_benchmark_matrix scripts.tests.test_ci_two_node_split_smoke scripts.tests.test_ci_sdk_json_consumer scripts.tests.test_ci_native_toolchain_routing scripts.tests.test_ci_prepare_native_runtime scripts.tests.test_justfile_source scripts.tests.test_skippy_competitive_benchmark scripts.tests.test_qa_kv_tool_loop_stability scripts.tests.test_llama_canary_agent_manifest_policy scripts.tests.test_native_artifact_verifiers scripts.tests.test_release_version_script scripts.tests.test_select_release_notes_base scripts.tests.test_model_artifact_registry scripts.tests.test_skippy_workload_certify scripts.tests.test_pr_workflow_artifacts scripts.tests.test_hf_skippy_mtp_certify_job scripts.tests.test_install_sccache scripts.tests.test_ci_source_layout scripts.tests.test_ci_openai_embeddings_smoke scripts.tests.test_logging_api_docs scripts.tests.test_check_skippy_workload_candidate scripts.tests.test_resume_crates_release_workflow scripts.tests.test_compare_event_benchmark_matrix scripts.tests.test_llama_canary_family_evidence scripts.tests.test_build_release scripts.tests.test_install_ps1 scripts.tests.test_agentic_replay_evidence scripts.tests.test_hf_xet_portability_smoke scripts.tests.test_kv_restart_replay scripts.tests.test_llama_canary_agent_repair_contract scripts.tests.test_logging_module_boundaries scripts.tests.test_release_script scripts.tests.test_run_command_with_timeout scripts.tests.test_synthetic_graph_registry scripts.tests.test_affected_crates_layout scripts.tests.test_package_release scripts.tests.test_agentic_trajectory_manifest scripts.tests.test_justfile_layout scripts.tests.test_plan_family_battery scripts.tests.test_macos_deployment_target scripts.tests.test_llama_upstream_canary_contract scripts.tests.test_llama_review_regressions scripts.tests.test_skippy_adaptive_prefill_ab scripts.tests.test_canary_family_memory scripts.tests.test_skippy_agentic_prompt_manifest scripts.tests.test_allocate_local_ports scripts.tests.test_skippy_mixed_prefill_decode_ab scripts.tests.test_ci_sdk_producers scripts.tests.test_cleanup_self_hosted scripts.tests.test_lld_lib scripts.tests.test_safe_extract_zip scripts.tests.test_llama_canary_mesh_ref scripts.tests.test_llama_native_full_replay scripts.tests.test_build_development_product scripts.tests.test_ci_artifact_actions scripts.tests.test_depot_canary_workflow scripts.tests.test_ci_cargo_packages scripts.tests.test_swift_xcframework_env scripts.tests.test_skippy_system_one_smoke scripts.tests.test_qa_nightly_stability scripts.tests.test_pr_sibling_cancellation scripts.tests.test_skippy_ocr_asr_oracle scripts.tests.test_skippy_static_link scripts.tests.test_justfile_release_runtime scripts.tests.test_skippy_dynamic_link scripts.tests.test_cli_inventory_contract scripts.tests.test_agentic_replay_history scripts.tests.test_release_workflow_artifacts scripts.tests.test_collect_ci_metrics

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
automation-run *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    bootstrap="$(just automation-bootstrap)"
    binary="$(printf '%s\n' "$bootstrap" | awk -F= '$1 == "binary_path" { print substr($0, index($0, "=") + 1) }')"
    test -n "$binary"
    exec "$binary" {{ ARGS }}

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
