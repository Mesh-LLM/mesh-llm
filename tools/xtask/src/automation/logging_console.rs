pub(super) mod checks;
mod coordinator;
mod evidence;
mod execution;
pub(super) mod http;
mod options;
mod setup;
use crate::command::DynResult;
use serde::Serialize;
use std::{io::Write, path::Path};

#[derive(Serialize)]
struct ResultRow {
    status: &'static str,
    name: &'static str,
    message: String,
}

const CHECKS: &[&str] = &[
    "real_embedded_console_bundle",
    "real_openai_lifecycle_and_detail",
    "restart_persistence",
    "trusted_local_rejection",
    "dedicated_sse_replay_gap_and_authoritative_hydration",
    "real_console_accessibility_and_responsive_modes",
    "cleanup",
];

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(
            crate::cli_output::stdout(),
            "automation logging-console --current-binary PATH [--evidence-dir DIR] [--base-port PORT] [--max-wait SECONDS] [--keep-state] [--print-plan]"
        )?;
        return Ok(());
    }
    let options = options::Options::parse(args)?;
    if options.print_plan {
        serde_json::to_writer(
            crate::cli_output::stdout(),
            &serde_json::json!({
                "script":"qa-logging-console-e2e.sh", "binary":options.binary,
                "evidence_root":options.evidence_root,"base_port":options.base_port,
                "max_wait_seconds":options.max_wait_seconds,"checks":CHECKS,"no_mocked_logs_routes":true
            }),
        )?;
        writeln!(crate::cli_output::stdout())?;
        return Ok(());
    }
    let prepared = setup::prepare(root, &options)?;
    let directory = prepared.directory.clone();
    let work = prepared.work.clone();
    let (report, joined, rows) = execution::execute(prepared, &options);
    let success = evidence::finish(&directory, &report, joined, rows)?;
    if success && !options.keep_state {
        std::fs::remove_dir_all(work)?;
    }
    if !success {
        return Err(execution::Failure { report, joined }.into());
    }
    writeln!(crate::cli_output::stdout(), "{}", directory.display())?;
    Ok(())
}
