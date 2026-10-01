#[path = "canary_receipts/command.rs"]
pub(crate) mod canary_aggregate_command;
#[expect(
    dead_code,
    unused_imports,
    reason = "Aggregate-only caller does not consume receipt writing and all API accessors"
)]
pub(crate) mod canary_receipts;
pub(crate) mod client_readiness;
mod codepoint_json;
pub(crate) use crate::command_interrupt;
pub(crate) mod daemon_readiness;
pub(crate) mod hf_converted_artifact;
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
pub(crate) mod split_evidence;
pub(crate) mod workload_oracle_evidence;

use crate::command::DynResult;

pub(crate) const REPLAY_EXPORT_USAGE: &str = "cargo xtool automation replay-matrix export --matrix <path> [--json-output <path>] [--github-env <path>] [--print-shell]";
pub(crate) const REPLAY_RUN_FAMILY_USAGE: &str = "cargo xtool automation replay-matrix run-family --matrix <path> --run-family <family> --ref <label=ref>... --dataset-file <path> --output <path> --python <absolute-executable> [--json-output <path>] [--github-env <path>] [--print-shell] [--timeout <seconds>]";
pub(crate) const WORKLOAD_ORACLE_EVIDENCE_USAGE: &str =
    "cargo xtool automation workload-oracle-evidence {write|verify} ...";

pub(crate) fn run_workload_oracle_evidence(args: &[String]) -> DynResult<()> {
    workload_oracle_evidence::run(args)
}

pub(crate) fn run_replay_matrix(args: &[String], root: Option<&std::path::Path>) -> DynResult<()> {
    match args {
        [verb, rest @ ..] if verb == "validate" => replay_matrix::run(rest),
        [verb, rest @ ..] if verb == "export" => replay_matrix::export::run(rest),
        [verb, rest @ ..] if verb == "run-family" => replay_matrix::run_family::run(rest, root),
        _ => Err(format!(
            "usage: cargo xtool automation replay-matrix {{validate|export|run-family}} --matrix <path>\n  {}",
            REPLAY_RUN_FAMILY_USAGE
        )
        .into()),
    }
}
