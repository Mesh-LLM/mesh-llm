pub(super) mod acceptance_gates;
pub(super) mod arm_pass;
pub(super) mod build_arm;
mod build_runtime;
pub(super) mod card;
pub(super) mod cell_execution;
mod cell_qualification;
mod cell_summary;
mod cell_workload;
mod cohort_identity;
mod context_eligibility;
pub(super) mod context_preflight;
pub(super) mod digest;
mod executable_resolution;
pub(super) mod export;
mod family_model;
mod family_workload;
pub(super) mod hardware;
mod input;
mod integer;
pub(super) mod manifest_preflight;
mod measured_prefix;
pub(super) mod model_preflight;
mod parameters;
mod pass_artifacts;
mod pass_identity;
mod pass_lifecycle;
mod pass_recurrent;
pub(super) mod pass_schedule;
pub(super) mod pins;
mod pooled_metrics;
pub(super) mod pooled_rows;
mod progress;
pub(super) mod publication;
pub(super) mod recorded_requests;
pub(super) mod recorded_requests_command;
mod recurrent_evidence;
pub(super) mod report;
mod report_chart;
mod report_csv;
mod report_escape;
mod report_input;
mod report_markdown;
mod report_method;
mod report_row;
mod report_table;
mod run_builds;
pub(super) mod run_execution;
pub(super) mod run_family;
mod run_input;
mod run_resume;
mod run_snapshot;
mod run_transport;
mod run_workload;
mod serialization;
pub(super) mod server_cell;
pub(super) mod server_cell_worker;
pub(super) mod session_evidence;
mod session_evidence_command;
mod session_summary;
mod stream_evidence;
pub(super) mod trajectory_execution;
pub(super) mod trajectory_reader;
mod value;

use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix validate --matrix <path>",
    values: &["--matrix"],
    flags: &["--help"],
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let report = match GRAMMAR.parse(args) {
        Err(report) => report,
        Ok(parsed) if parsed.flag("--help") => CheckReport::success(format!(
            "usage: {}\n\nValidate replay parameters and print one tab-delimited shell line.\nReads only --matrix, relative to the invocation directory. No replay is executed.\n\nOptions:\n  --matrix <path>  Required matrix input\n  --help           Show this help\n",
            GRAMMAR.usage
        )),
        Ok(parsed) if !parsed.positionals.is_empty() => GRAMMAR.error(&format!(
            "unrecognized arguments: {}",
            parsed.positionals.join(" ")
        )),
        Ok(parsed) => match parsed.last("--matrix") {
            None => GRAMMAR.error("the following arguments are required: --matrix"),
            Some(path) => validate(&PathBuf::from(path))?,
        },
    };
    report.emit()
}

fn validate(path: &Path) -> DynResult<CheckReport> {
    with_worker(|| match input::load(path) {
        Ok(loaded) => CheckReport::success(loaded.parameters.shell_line()),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    })
}

fn with_worker<T: Send>(operation: impl FnOnce() -> T + Send) -> DynResult<T> {
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("replay-matrix".into())
            .stack_size(64 * 1024 * 1024)
            .spawn_scoped(scope, operation)?;
        match worker.join() {
            Ok(report) => Ok(report),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

pub(super) mod history;
mod history_artifacts;
mod history_baseline;
mod history_input;
mod history_rows;

mod arm_pass_launch;
mod arm_pass_result;
mod arm_pass_workload;
pub(super) mod external_cell;
mod external_command;
mod external_config;
pub(super) mod external_probe;
mod l3_contract;
mod l3_driver;
mod l3_execution;
mod l3_gates;
mod l3_identity;
mod l3_management;
mod l3_manifest;
pub(super) mod l3_plan;
pub(super) mod l3_report;
mod l3_requests;
pub(super) mod l3_run;
mod l3_server;
#[cfg(test)]
mod l3_tests;
mod run_arms;
mod run_budget;
mod run_completion;
mod run_measurement;
mod run_qualification;
mod run_resume_context;
mod run_resume_pass;
mod run_setup;

pub(super) mod history_fetch;
mod history_hub;
#[cfg(test)]
mod history_hub_tests;
pub(super) mod history_upload;

mod manual_options;
pub(super) mod manual_replay;
#[cfg(test)]
mod manual_tests;
mod replay_profile;
mod resume_profile;

#[cfg(test)]
#[path = "replay_matrix/retirement_tests.rs"]
mod retirement_tests;
