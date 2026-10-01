use super::super::parameters::Parameters;
use super::run_family_invocation::PreparedInvocation;
use super::run_family_process_failure;
use super::run_family_report::RunFamilyReport;
use super::{Options, input};
use crate::automation::codepoint_json::value::Value;
use crate::automation::command_interrupt::Interrupt;
use crate::process::{self, Completion, Limits, ProcessSpec, Readiness};
use std::path::Path;
use std::time::Duration;

pub(super) struct InvocationContext<'a> {
    pub(super) family: &'a str,
    pub(super) python: &'a Path,
    pub(super) dataset: &'a Path,
    pub(super) output: &'a Path,
    pub(super) cwd: &'a Path,
    pub(super) repo_root: &'a Path,
    pub(super) timeout: Duration,
}

pub(super) struct ChildContext<'a> {
    pub(super) options: &'a Options,
    pub(super) first: &'a input::LoadedReplay,
    pub(super) invocation: InvocationContext<'a>,
}

pub(super) fn run_selected(
    options: &Options,
    first: &input::LoadedReplay,
    parameters: &Parameters,
    matrix: &Value,
    context: InvocationContext<'_>,
) -> RunFamilyReport {
    let prepared =
        match super::run_family_invocation::prepare(options, parameters, matrix, &context) {
            Ok(prepared) => prepared,
            Err(report) => return report,
        };
    run_child(
        ChildContext {
            options,
            first,
            invocation: context,
        },
        prepared,
    )
}

fn run_child(child: ChildContext<'_>, prepared: PreparedInvocation) -> RunFamilyReport {
    let spec =
        super::run_family_spec::child_spec(&child.invocation, &prepared.python, prepared.arguments);
    let report = match supervised_child(&spec, child.invocation.timeout) {
        Ok(report) => report,
        Err(report) => return report,
    };
    if !report.success() {
        return run_family_process_failure::report(report);
    }
    run_family_process_failure::success_output(
        child.options.parsed.flag("--print-shell"),
        child.first.parameters.shell_line(),
        report,
    )
}

fn supervised_child(
    spec: &ProcessSpec,
    timeout: Duration,
) -> Result<process::ProcessReport, RunFamilyReport> {
    let limits = Limits {
        execution: timeout,
        graceful_shutdown: Duration::from_secs(2),
        forced_shutdown: Duration::from_secs(3),
        retained_bytes_per_stream: 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let interrupt = match Interrupt::install() {
        Ok(interrupt) => interrupt,
        Err(error) => return Err(RunFamilyReport::failure(format!("run-family: {error}\n"))),
    };
    let cancellation = interrupt.cancellation();
    let report = match process::supervise(
        spec,
        &limits,
        &cancellation,
        process::OutputFiles::default(),
    ) {
        Ok(report) => report,
        Err(error) => {
            return Err(RunFamilyReport::failure(format!(
                "run-family process: {error}\n"
            )));
        }
    };
    let interrupted = cancellation.is_cancelled();
    let interruption = interrupt.finish();
    if interrupted {
        return Err(run_family_process_failure::interrupted(report));
    }
    if let Err(error) = interruption {
        return Err(RunFamilyReport::failure(format!("run-family: {error}\n")));
    }
    Ok(report)
}
