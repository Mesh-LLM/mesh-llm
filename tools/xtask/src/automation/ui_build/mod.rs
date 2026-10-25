//! UI build command, argument admission and supervised outcome reporting.
mod execution;
#[cfg(test)]
mod execution_tests;
mod failure;
mod options;
#[cfg(test)]
mod options_tests;
mod policy;
#[cfg(test)]
mod policy_tests;
mod setup;
use crate::{
    command::DynResult, command_interrupt::Interrupt, repository::check_report::CheckReport,
};
use std::io::Write;

const USAGE: &str = "cargo xtool automation ui-build --ui-dir PATH [--profile debug|dev|release] [--logs-dir PATH] [--timeout-secs 1..3600] [--pnpm-command PATH [--pnpm-script PATH]]";

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(crate::cli_output::stdout(), "{USAGE}")?;
        return Ok(());
    }
    let options = options::Options::parse(args)?;
    let environment = options::BuildEnvironment::from_environment(options.profile)?;
    let build = environment.projection();
    let logs = options
        .logs
        .clone()
        .unwrap_or_else(|| options.ui.join(".mesh-llm-ui-build-logs"));
    let interrupt = Interrupt::install()?;
    let result = execution::run(
        &options.ui,
        build,
        || setup::prepare(&options),
        options.timeout,
        &interrupt.cancellation(),
        &logs,
    );
    let signal = finish(interrupt)?;
    let report = match result {
        Ok(decision) if signal.is_none() => {
            let action = match decision {
                policy::Decision::Reuse => "Skipping mesh-llm UI build; dist is up to date",
                policy::Decision::Build | policy::Decision::InstallAndBuild => "Built mesh-llm UI",
            };
            CheckReport::success(format!(
                "{action} (profile: {}, debug UI: {}).\n",
                build.profile.name(),
                build.debug_ui()
            ))
        }
        Ok(_) => CheckReport {
            stdout: String::new(),
            stderr: "UI build interrupted after cleanup\n".into(),
            code: 128 + signal.unwrap_or(2),
        },
        Err(error) => {
            let code = error.downcast_ref::<failure::Failure>().map_or_else(
                || {
                    error.downcast_ref::<failure::Admission>().map_or_else(
                        || signal.map_or(1, |value| 128 + value),
                        |admission| admission.code(signal),
                    )
                },
                |failure| failure.code(signal),
            );
            CheckReport {
                stdout: String::new(),
                stderr: format!("{error}\n"),
                code,
            }
        }
    };
    report.emit()
}

#[cfg(unix)]
fn finish(interrupt: Interrupt) -> DynResult<Option<i32>> {
    Ok(interrupt.finish_signal()?)
}
#[cfg(windows)]
fn finish(interrupt: Interrupt) -> DynResult<Option<i32>> {
    match interrupt.finish() {
        Ok(()) => Ok(None),
        Err(crate::command_interrupt::Reason::Interrupted) => Ok(Some(2)),
        Err(error) => Err(error.into()),
    }
}
