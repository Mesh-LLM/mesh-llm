use super::{Error, GitDiff};
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawBytes, RawCaptureOptions, Readiness,
    Value,
};
use std::ffi::OsString;
use std::path::Path;
use std::time::Duration;

pub(super) struct Invocation<'a> {
    pub(super) executable: &'a Path,
    pub(super) arguments: Vec<OsString>,
    pub(super) phase: &'static str,
    pub(super) capture: bool,
}

pub(super) fn execute(
    input: &GitDiff,
    invocation: Invocation<'_>,
    cancellation: &Cancellation,
) -> Result<Option<RawBytes>, Error> {
    if invocation.arguments.len() > 16384
        || invocation
            .arguments
            .iter()
            .map(|argument| argument.as_encoded_bytes().len())
            .try_fold(0_usize, usize::checked_add)
            .is_none_or(|length| length > 4 * 1024 * 1024)
    {
        return Err(Error::Arguments("generator argv exceeds safety limit"));
    }
    let spec = ProcessSpec {
        executable: invocation.executable.to_owned(),
        arguments: invocation
            .arguments
            .into_iter()
            .map(Value::Public)
            .collect(),
        cwd: input.source_root.clone(),
        environment: input
            .environment
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    Value::Public(value) => Value::Public(value.clone()),
                    Value::Secret(value) => Value::Secret(value.clone()),
                };
                (key.clone(), value)
            })
            .collect(),
    };
    let limits = Limits {
        execution: input.timeout,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        cancellation,
        RawCaptureOptions {
            stdout: invocation.capture.then_some(input.max_bytes),
            stderr: None,
        },
    )?;
    if !report.process.success() {
        return Err(Error::Command {
            phase: invocation.phase,
            report: Box::new(report.process),
        });
    }
    if invocation.capture && report.stdout.is_none() {
        return Err(Error::MissingPayload);
    }
    Ok(report.stdout)
}

pub(super) fn git(
    input: &GitDiff,
    arguments: &[&str],
    cancellation: &Cancellation,
) -> Result<RawBytes, Error> {
    execute(
        input,
        Invocation {
            executable: &input.executable,
            arguments: arguments.iter().map(OsString::from).collect(),
            phase: "git",
            capture: true,
        },
        cancellation,
    )?
    .ok_or(Error::MissingPayload)
}
