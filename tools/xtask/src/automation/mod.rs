pub(crate) mod agent_model;
#[path = "canary_receipts/command.rs"]
pub(crate) mod canary_aggregate_command;
#[path = "canary_receipts/build.rs"]
pub(crate) mod canary_build;
#[path = "canary_receipts/handoff.rs"]
pub(crate) mod canary_handoff;
#[path = "canary_receipts/package_closure.rs"]
pub(crate) mod canary_package_closure;
#[expect(
    dead_code,
    reason = "Direct component tests also consume receipt parsing and error accessors"
)]
pub(crate) mod canary_receipts;
#[path = "canary_receipts/source_plan.rs"]
pub(crate) mod canary_source_plan;
pub(crate) mod client_readiness;
mod codepoint_json;
pub(crate) mod control_plane_qa;
pub(crate) mod daemon_lifecycle;
pub(crate) mod laya;
pub(crate) mod logging_console;
pub(crate) mod logging_recovery;
pub(crate) mod runtime_install;
pub(crate) mod sdk_fixture;
pub(crate) mod startup_recovery;
pub(crate) use crate::command_interrupt;
pub(crate) mod daemon_readiness;
pub(crate) mod hf_converted_artifact;
pub(crate) mod hf_xet_smoke;
pub(crate) mod native_contracts;
pub(crate) mod native_generator;
mod private_state;
pub(crate) mod qualification;
mod replay_matrix;
pub(crate) mod required_smoke;
#[expect(
    dead_code,
    reason = "retained session adapter awaits certification port"
)]
pub(crate) mod retained_session;
mod rewriter_patch;
pub(crate) mod rewriter_report;
pub(crate) mod rollout;
pub(crate) mod sdk_advisory;
#[cfg(test)]
#[path = "../../tests/migration_lifecycle/shared_owners.rs"]
pub(crate) mod shared_owner_tests;
pub(crate) mod smoke_inputs;
pub(crate) mod smoke_observation;
pub(crate) mod split_evidence;
pub(crate) mod split_probe;
pub(crate) mod workload_oracle_evidence;
pub(crate) mod workload_smoke;

use crate::command::DynResult;

pub(crate) const REPLAY_EXPORT_USAGE: &str = "cargo xtool automation replay-matrix export --matrix <path> [--json-output <path>] [--github-env <path>] [--print-shell]";
pub(crate) const REPLAY_RUN_FAMILY_USAGE: &str = "cargo xtool automation replay-matrix run-family --matrix <path> --run-family <family> --ref <label=ref>... --dataset-file <path> --model-file <path> --output <path> --python <absolute-executable> [--json-output <path>] [--github-env <path>] [--print-shell] [--timeout <seconds>]";
pub(crate) const WORKLOAD_ORACLE_EVIDENCE_USAGE: &str =
    "cargo xtool automation workload-oracle-evidence {write|verify} ...";

pub(crate) fn run_workload_oracle_evidence(args: &[String]) -> DynResult<()> {
    workload_oracle_evidence::run(args)
}

pub(crate) fn run_replay_matrix(args: &[String], root: Option<&std::path::Path>) -> DynResult<()> {
    match args {
        [verb, rest @ ..] if verb == "plan" => replay_matrix::manual_replay::run(root, rest, false),
        [verb, rest @ ..] if verb == "run" => replay_matrix::manual_replay::run(root, rest, true),
        [verb, rest @ ..] if verb == "hardware" => replay_matrix::hardware::run(rest),
        [verb, rest @ ..] if verb == "card" => replay_matrix::card::run(rest),
        [verb, rest @ ..] if verb == "history" => replay_matrix::history::run(rest),
        [verb, rest @ ..] if verb == "history-fetch" => replay_matrix::history_fetch::run(rest),
        [verb, rest @ ..] if verb == "history-upload" => replay_matrix::history_upload::run(rest),
        [verb, rest @ ..] if verb == "external-config" => replay_matrix::external_probe::run(rest),
        [verb, rest @ ..] if verb == "external-cell" => replay_matrix::external_cell::run(root, rest),
        [verb, rest @ ..] if verb == "l3-plan" => replay_matrix::l3_plan::run(root, rest),
        [verb, rest @ ..] if verb == "l3-run" => replay_matrix::l3_run::run(root, rest),
        [verb, rest @ ..] if verb == "l3-report" => replay_matrix::l3_report::run(rest),
        [verb, rest @ ..] if verb == "execute-run" => replay_matrix::run_execution::run(root, rest),
        [verb, rest @ ..] if verb == "context-preflight" => replay_matrix::context_preflight::run(root, rest),
        [verb, rest @ ..] if verb == "arm-pass" => replay_matrix::arm_pass::run(root, rest),
        [verb, rest @ ..] if verb == "build-arm" => replay_matrix::build_arm::run(rest),
        [verb, rest @ ..] if verb == "pooled-rows" => replay_matrix::pooled_rows::run(rest),
        [verb, rest @ ..] if verb == "report" => replay_matrix::report::run(rest),
        [verb, rest @ ..] if verb == "acceptance-gates" => replay_matrix::acceptance_gates::run(rest),
        [verb, rest @ ..] if verb == "manifest-preflight" => replay_matrix::manifest_preflight::run(rest),
        [verb, rest @ ..] if verb == "server-cell" => replay_matrix::server_cell::run(root, rest),
        [verb, rest @ ..] if verb == "server-cell-worker" => replay_matrix::server_cell_worker::run(rest),
        [verb, rest @ ..] if verb == "pass-schedule" => replay_matrix::pass_schedule::run(rest),
        [verb, rest @ ..] if verb == "model-preflight" => replay_matrix::model_preflight::run(rest),
        [verb, rest @ ..] if verb == "execute-cell" => replay_matrix::cell_execution::run(rest),
        [verb, rest @ ..] if verb == "execute-trajectory" => replay_matrix::trajectory_execution::run(rest),
        [verb, rest @ ..] if verb == "validate" => replay_matrix::run(rest),
        [verb, rest @ ..] if verb == "export" => replay_matrix::export::run(rest),
        [verb, rest @ ..] if verb == "run-family" => replay_matrix::run_family::run(rest, root),
        [verb, rest @ ..] if verb == "pins" => replay_matrix::pins::run(rest),
        [verb, rest @ ..] if verb == "verify-digest" => replay_matrix::digest::run(rest),
        [verb, rest @ ..] if verb == "publication-verify" => replay_matrix::publication::run(rest),
        [verb, rest @ ..] if verb == "session-evidence" => replay_matrix::session_evidence::run(rest),
        [verb, rest @ ..] if verb == "recorded-requests" => replay_matrix::recorded_requests_command::run(rest),
        [verb, rest @ ..] if verb == "trajectory-reader" => replay_matrix::trajectory_reader::run(root, rest),
        _ => Err(format!(
            "usage: cargo xtool automation replay-matrix {{plan|run|validate|export|run-family|execute-run|history|history-fetch|history-upload|card|hardware|l3-plan|l3-run|l3-report|external-config|external-cell|pins|verify-digest|publication-verify}} ...\n  {}",
            REPLAY_RUN_FAMILY_USAGE
        )
        .into()),
    }
}
