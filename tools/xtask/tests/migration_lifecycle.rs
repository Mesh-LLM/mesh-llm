#![cfg(unix)]

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
