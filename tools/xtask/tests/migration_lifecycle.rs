#![cfg(unix)]

#[path = "migration_lifecycle/ci_batch_filter/mod.rs"]
mod ci_batch_filter;
#[path = "migration_lifecycle/macos_deployment_target.rs"]
mod macos_deployment_target;
#[path = "migration_lifecycle/package_release_adapter.rs"]
mod package_release_adapter;
#[path = "migration_lifecycle/pr_canary_catalog.rs"]
mod pr_canary_catalog;
#[path = "migration_lifecycle/replay_repair_step.rs"]
mod replay_repair_step;
#[path = "migration_lifecycle/replay_runner_guard.rs"]
mod replay_runner_guard;
#[path = "migration_lifecycle/runtime_events_gate/mod.rs"]
mod runtime_events_gate;

#[path = "migration_lifecycle/just_automation_argv.rs"]
mod just_automation_argv;

#[path = "../src/automation/command_interrupt/mod.rs"]
pub(crate) mod command_interrupt;

#[path = "../src/automation"]
mod automation {
    pub(crate) mod client_readiness;
    pub(crate) use crate::command_interrupt;
    pub(crate) mod daemon_readiness;
    mod private_state;
    pub(crate) mod retained_session;
    #[cfg(test)]
    #[path = "../../tests/migration_lifecycle/shared_owners.rs"]
    pub(crate) mod shared_owner_tests;
}
#[path = "migration_lifecycle/blocked_signals.rs"]
mod blocked_signals;
#[path = "migration_lifecycle/cleanup_owner.rs"]
mod cleanup_owner;
#[cfg(unix)]
#[path = "migration_lifecycle/cli.rs"]
mod cli;
#[path = "migration_lifecycle/daemon/mod.rs"]
mod daemon;
#[path = "migration_lifecycle/daemon/header_bound.rs"]
mod daemon_header_bound;
#[path = "migration_lifecycle/interruption.rs"]
mod interruption;
#[path = "migration_lifecycle/observation.rs"]
mod observation;
#[path = "../src/process/mod.rs"]
pub mod process;
#[path = "migration_lifecycle/protocol.rs"]
mod protocol;
#[path = "migration_lifecycle/pty.rs"]
mod pty;
#[path = "migration_lifecycle/runner_cleanup.rs"]
mod runner_cleanup;
#[path = "migration_lifecycle/runner_cleanup_contracts.rs"]
mod runner_cleanup_contracts;
#[path = "migration_lifecycle/selected_ref.rs"]
mod selected_ref;
#[path = "migration_lifecycle/support.rs"]
mod support;

#[path = "migration_lifecycle/agent_client_config_cli.rs"]
mod agent_client_config_cli;
#[path = "migration_lifecycle/agent_fixture_evidence_cli.rs"]
mod agent_fixture_evidence_cli;
#[path = "migration_lifecycle/agent_fixture_inputs_cli.rs"]
mod agent_fixture_inputs_cli;
#[path = "migration_lifecycle/automation_producer_admission.rs"]
mod automation_producer_admission;
#[path = "migration_lifecycle/automation_restore_admission.rs"]
mod automation_restore_admission;
#[path = "migration_lifecycle/battery_cache_cli.rs"]
mod battery_cache_cli;
#[path = "migration_lifecycle/battery_timeout_cli.rs"]
mod battery_timeout_cli;
#[path = "migration_lifecycle/cache_family_report_cli.rs"]
mod cache_family_report_cli;
#[path = "migration_lifecycle/canary_controller_battery.rs"]
mod canary_controller_battery;
#[path = "migration_lifecycle/canary_timeout_cli.rs"]
mod canary_timeout_cli;
#[path = "migration_lifecycle/family_battery_policy_cli.rs"]
mod family_battery_policy_cli;
#[path = "migration_lifecycle/family_model_identity_cli.rs"]
mod family_model_identity_cli;
#[path = "migration_lifecycle/frozen_selector_integration.rs"]
mod frozen_selector_integration;
#[path = "migration_lifecycle/required_sdk_environment_admission.rs"]
mod required_sdk_environment_admission;
#[path = "migration_lifecycle/workload_provenance_cli.rs"]
mod workload_provenance_cli;

#[path = "migration_lifecycle/binary_stage_readiness_cli.rs"]
mod binary_stage_readiness_cli;
#[path = "migration_lifecycle/runtime_release_manifest_wrapper.rs"]
mod runtime_release_manifest_wrapper;
#[path = "migration_lifecycle/skippy_cache_smoke_config_cli.rs"]
mod skippy_cache_smoke_config_cli;
#[path = "migration_lifecycle/skippy_ci_smoke_control_cli.rs"]
mod skippy_ci_smoke_control_cli;
#[path = "migration_lifecycle/workload_media_comparison_cli.rs"]
mod workload_media_comparison_cli;
#[path = "migration_lifecycle/workload_monolithic_cli.rs"]
mod workload_monolithic_cli;
#[path = "migration_lifecycle/workload_tts_cli.rs"]
mod workload_tts_cli;

#[path = "migration_lifecycle/safetensors_workflow.rs"]
mod safetensors_workflow;

#[path = "migration_lifecycle/publish_dry_run.rs"]
mod publish_dry_run;

#[path = "migration_lifecycle/repair_timeout_adapter.rs"]
mod repair_timeout_adapter;

#[path = "migration_lifecycle/build_product/mod.rs"]
mod build_product;

#[path = "migration_lifecycle/sdk_json_consumer/mod.rs"]
mod sdk_json_consumer;

#[path = "migration_lifecycle/system_one_cases/mod.rs"]
mod system_one_cases;

#[path = "migration_lifecycle/repair_family_plan.rs"]
mod repair_family_plan;

#[path = "migration_lifecycle/local_repair_inspection.rs"]
mod local_repair_inspection;

#[path = "migration_lifecycle/package_input_admission.rs"]
mod package_input_admission;

#[path = "../src/ci_validation/lane_results/workflow_yaml.rs"]
mod workflow_yaml;

#[path = "migration_lifecycle/verification_source_cli.rs"]
mod verification_source_cli;

#[path = "migration_lifecycle/certification_producer_admission.rs"]
mod certification_producer_admission;

#[path = "migration_lifecycle/affected_crates_relocated_cli.rs"]
mod affected_crates_relocated_cli;

#[path = "migration_lifecycle/compute_changes_justfiles.rs"]
mod compute_changes_justfiles;

#[path = "migration_lifecycle/family_evidence_workflow.rs"]
mod family_evidence_workflow;

#[path = "migration_lifecycle/family_build_dispatch.rs"]
mod family_build_dispatch;

#[path = "migration_lifecycle/family_battery_no_python.rs"]
mod family_battery_no_python;

#[path = "migration_lifecycle/canary_mode_dispatch.rs"]
mod canary_mode_dispatch;

#[cfg(unix)]
#[path = "migration_lifecycle/native_sdk_restore.rs"]
mod native_sdk_restore;

#[path = "migration_lifecycle/static_abi_sdk_prebuilt.rs"]
mod static_abi_sdk_prebuilt;

#[path = "migration_lifecycle/static_abi_dynamic_outputs.rs"]
mod static_abi_dynamic_outputs;

#[path = "migration_lifecycle/static_abi_ffi_boundary.rs"]
mod static_abi_ffi_boundary;

#[path = "migration_lifecycle/native_release_recipes.rs"]
mod native_release_recipes;

#[path = "migration_lifecycle/lld_shell_contract.rs"]
mod lld_shell_contract;

#[path = "migration_lifecycle/static_abi_build_policy.rs"]
mod static_abi_build_policy;

#[path = "migration_lifecycle/product_composition_adapter.rs"]
mod product_composition_adapter;

#[path = "migration_lifecycle/sccache_installer.rs"]
mod sccache_installer;

#[path = "migration_lifecycle/release_script.rs"]
mod release_script;

#[path = "migration_lifecycle/publication_diagnostics.rs"]
mod publication_diagnostics;

#[path = "migration_lifecycle/parity_download_owner.rs"]
mod parity_download_owner;

#[path = "migration_lifecycle/pr_sibling_cancellation.rs"]
mod pr_sibling_cancellation;

#[path = "migration_lifecycle/ui_package_contracts.rs"]
mod ui_package_contracts;

#[path = "migration_lifecycle/just_layout_contracts.rs"]
mod just_layout_contracts;
