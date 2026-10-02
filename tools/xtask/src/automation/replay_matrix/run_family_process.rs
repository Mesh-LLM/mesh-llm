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
    pub(super) model: &'a Path,
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
    _parameters: &Parameters,
    _matrix: &Value,
    context: InvocationContext<'_>,
) -> RunFamilyReport {
    let prepared = match super::run_family_invocation::prepare(options, &context) {
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
    let PreparedInvocation {
        executable,
        arguments,
        workload,
    } = prepared;
    let _workload = workload;
    let spec = super::run_family_spec::child_spec(&child.invocation, &executable, arguments);
    let report = match supervised_child(spec, child.invocation.timeout) {
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
    spec: ProcessSpec,
    timeout: Duration,
) -> Result<process::ProcessReport, RunFamilyReport> {
    let limits = Limits {
        execution: timeout,
        // The Rust child owns nested supervisors with their own 2s+3s cleanup.
        // Let that cleanup finish before forcing the coordinating process down.
        graceful_shutdown: Duration::from_secs(10),
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
    let session = run_session(spec, timeout, &limits, &cancellation);
    let interrupted = cancellation.is_cancelled();
    let interruption = interrupt.finish();
    let report = session_report(session)?;
    if interrupted {
        return Err(run_family_process_failure::interrupted(report));
    }
    if let Err(error) = interruption {
        return Err(RunFamilyReport::failure(format!("run-family: {error}\n")));
    }
    Ok(report)
}

fn run_session(
    spec: ProcessSpec,
    timeout: Duration,
    limits: &Limits,
    cancellation: &process::Cancellation,
) -> Result<process::retained::Report<String>, process::Failure> {
    std::thread::scope(|scope| {
        let forwarder = crate::automation::replay_matrix::progress::Forwarder::new(scope);
        let launch = process::retained::Launch {
            member: process::retained::MemberId::Seed,
            spec,
            files: process::OutputFiles::default(),
            readiness_deadline: timeout,
        };
        let policy = process::retained::ExpectedExit::new(&[0, 1, 2, 101], timeout)?;
        let mut owner = super::run_family_progress::Owner {
            launch: Some(launch),
            policy,
            forwarder: &forwarder,
        };
        let result = process::retained::run(&mut owner, limits, cancellation);
        drop(owner);
        let forwarded = forwarder.finish();
        let mut result = result?;
        if let Err(error) = forwarded {
            result.outcome = process::Outcome::IoFailure;
            result.rejection = Some(format!("replay progress output failed: {error}"));
        }
        Ok::<_, process::Failure>(result)
    })
}

fn session_report(
    session: Result<process::retained::Report<String>, process::Failure>,
) -> Result<process::ProcessReport, RunFamilyReport> {
    let session = match session {
        Ok(report) => report,
        Err(error) => {
            return Err(RunFamilyReport::failure(format!(
                "run-family process: {error}\n"
            )));
        }
    };
    let mut report = match session
        .members
        .into_iter()
        .find(|member| member.member == process::retained::MemberId::Seed)
    {
        Some(member) => member.process,
        None => {
            return Err(RunFamilyReport::failure(
                "run-family child produced no process report\n".into(),
            ));
        }
    };
    if session.outcome != process::Outcome::Ready {
        report.outcome = session.outcome;
    }
    if let Some(failure) = session.failure {
        report.failure = Some(failure);
    }
    if let Some(rejection) = session.rejection {
        return Err(RunFamilyReport::failed_child(
            report.stdout.bytes_retained,
            format!("run-family progress failed: {rejection}\n"),
        ));
    }
    Ok(report)
}
