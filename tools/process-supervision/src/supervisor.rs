use super::capture::Capture;
use super::observed::{self, Admission, Decision};
use super::output::{Output, output_file};
use super::report::StopObservation;
use super::{
    Cancellation, Completion, Failure, Limits, Outcome, OutputFiles, ProcessReport, ProcessSpec,
    Readiness, ReadinessObservation, ReadinessStop, Stream, control, platform,
};
use std::thread;
use std::time::Instant;

/// Runs one owned tree and returns its exit status even on timeout/cancellation.
/// Startup/setup errors return Err with RAII tree cleanup; monitored failures
/// belong to the report and never lose the observed exit status.
/// Mark every credential argument/env value Secret. Known sensitive log-key
/// lines and oversized lines are suppressed; arbitrary undeclared secrets are
/// not detectable. Files contain the same capped, sanitized bytes as reports.
pub fn supervise(
    spec: &ProcessSpec,
    limits: &Limits,
    cancellation: &Cancellation,
    files: OutputFiles,
) -> Result<ProcessReport, Failure> {
    supervise_owned(
        spec,
        limits,
        cancellation,
        files,
        |child, output, started| {
            let (outcome, ready, observation) =
                monitor_readiness(child, output, (limits, cancellation, started));
            (outcome, ready, observation.map(StopObservation::Line))
        },
    )
}

pub(super) fn supervise_owned(
    spec: &ProcessSpec,
    limits: &Limits,
    cancellation: &Cancellation,
    files: OutputFiles,
    monitor: impl FnOnce(
        &mut platform::OwnedChild,
        &mut Output,
        Instant,
    ) -> (Outcome, bool, Option<StopObservation>),
) -> Result<ProcessReport, Failure> {
    supervise_configured(
        spec,
        limits,
        cancellation,
        (files, super::RawCaptureOptions::default()),
        None,
        |child, output, started, _projection| monitor(child, output, started),
    )
    .map(|report| report.process)
}

pub fn supervise_raw(
    spec: &ProcessSpec,
    limits: &Limits,
    cancellation: &Cancellation,
    options: super::RawCaptureOptions,
) -> Result<super::RawProcessReport, Failure> {
    supervise_configured(
        spec,
        limits,
        cancellation,
        (OutputFiles::default(), options),
        None,
        |child, output, started, _projection| {
            let (outcome, ready, observation) =
                monitor_readiness(child, output, (limits, cancellation, started));
            (outcome, ready, observation.map(StopObservation::Line))
        },
    )
}

/// Runs the same owned supervisor with bounded raw in-memory evidence and
/// independently capped, sanitized diagnostic files. RawBytes never supplies
/// file contents and must not be persisted or printed by callers. Each raw
/// stream is admitted only after EOF within its explicit byte bound.
pub fn supervise_raw_with_files(
    spec: &ProcessSpec,
    limits: &Limits,
    cancellation: &Cancellation,
    files: OutputFiles,
    options: super::RawCaptureOptions,
) -> Result<super::RawProcessReport, Failure> {
    supervise_configured(
        spec,
        limits,
        cancellation,
        (files, options),
        None,
        |child, output, started, _projection| {
            let (outcome, ready, observation) =
                monitor_readiness(child, output, (limits, cancellation, started));
            (outcome, ready, observation.map(StopObservation::Line))
        },
    )
}

pub(super) fn supervise_configured(
    spec: &ProcessSpec,
    limits: &Limits,
    cancellation: &Cancellation,
    capture: (OutputFiles, super::RawCaptureOptions),
    mut projection: super::projection::Callback<'_>,
    monitor: impl FnOnce(
        &mut platform::OwnedChild,
        &mut Output,
        Instant,
        &mut super::projection::Callback<'_>,
    ) -> (Outcome, bool, Option<StopObservation>),
) -> Result<super::RawProcessReport, Failure> {
    let (files, raw) = capture;
    limits.validate()?;
    let secrets = spec.secrets()?;
    let mut command = spec.command()?;
    if cancellation.is_cancelled() {
        return Err(Failure::InvalidSpec("cancelled before spawn"));
    }
    let stdout_file = output_file(files.stdout.as_deref())?;
    let stderr_file = output_file(files.stderr.as_deref())?;
    let started = Instant::now();
    let mut child = platform::OwnedChild::spawn(&mut command)?;
    let stdout = child
        .stdout()
        .ok_or(Failure::InvalidSpec("stdout pipe missing"))?;
    let stderr = child
        .stderr()
        .ok_or(Failure::InvalidSpec("stderr pipe missing"))?;
    let mut output = Output {
        stdout: Capture::new(
            stdout,
            Stream::Stdout,
            stdout_file,
            limits.retained_bytes_per_stream,
            secrets.clone(),
        )?,
        stderr: Capture::new(
            stderr,
            Stream::Stderr,
            stderr_file,
            limits.retained_bytes_per_stream,
            secrets,
        )?,
        failure: None,
    };
    output.stdout.enable_raw(raw.stdout);
    output.stderr.enable_raw(raw.stderr);
    let (outcome, ready, observation) = monitor(&mut child, &mut output, started, &mut projection);
    let drain = || {
        super::projection::poll(&mut output, &mut projection);
    };
    let (cleanup, status, readiness_stop) = match observation {
        Some(observation) => {
            let (cleanup, status, request) = control::shutdown_ready(&mut child, limits, drain);
            (cleanup, status, observation.receipt(request))
        }
        None => {
            let (cleanup, status) = control::shutdown(&mut child, limits, drain);
            (cleanup, status, ReadinessStop::NotAdmitted)
        }
    };
    let until = Instant::now() + limits.forced_shutdown;
    while !output.stdout.eof() || !output.stderr.eof() {
        super::projection::poll(&mut output, &mut projection);
        if output
            .failure
            .as_ref()
            .is_some_and(|failure| !matches!(failure, Failure::RawCaptureOverflow { .. }))
        {
            break;
        }
        if Instant::now() >= until {
            output.failure.get_or_insert(Failure::CleanupDeadline);
            break;
        }
        thread::sleep(control::POLL);
    }
    let mut raw_failure = None;
    let raw_stdout = match output.stdout.take_raw() {
        Ok(bytes) => bytes,
        Err(error) => {
            raw_failure = Some(error);
            None
        }
    };
    let raw_stderr = match output.stderr.take_raw() {
        Ok(bytes) => bytes,
        Err(error) => {
            raw_failure.get_or_insert(error);
            None
        }
    };
    let (stdout, stdout_failure) = output.stdout.finish();
    let (stderr, stderr_failure) = output.stderr.finish();
    Ok(super::RawProcessReport {
        process: ProcessReport {
            pid: child.id(),
            outcome,
            status,
            ready,
            readiness_stop,
            elapsed: started.elapsed(),
            stdout,
            stderr,
            cleanup,
            failure: output
                .failure
                .or(raw_failure)
                .or(stdout_failure)
                .or(stderr_failure),
        },
        stdout: raw_stdout,
        stderr: raw_stderr,
    })
}

fn monitor_readiness(
    child: &mut platform::OwnedChild,
    output: &mut Output,
    context: (&Limits, &Cancellation, Instant),
) -> (Outcome, bool, Option<ReadinessObservation>) {
    let (limits, cancellation, started) = context;
    match &limits.readiness {
        Readiness::None | Readiness::Line { .. } => {
            let (outcome, ready) = monitor(child, output, context);
            (outcome, ready, None)
        }
        Readiness::ObservedLines { .. } => match observed::monitor(
            child,
            output,
            &Admission {
                limits,
                cancellation,
                elapsed: || started.elapsed(),
            },
        ) {
            Decision::Admitted(observation) => (Outcome::Ready, true, Some(observation)),
            Decision::Terminal(outcome) => (outcome, false, None),
            Decision::Failed(error) => {
                output.failure = Some(error);
                (Outcome::IoFailure, false, None)
            }
        },
    }
}

fn monitor(
    child: &mut platform::OwnedChild,
    output: &mut Output,
    context: (&Limits, &Cancellation, Instant),
) -> (Outcome, bool) {
    let (limits, cancellation, started) = context;
    let mut ready = false;
    loop {
        if cancellation.is_cancelled() {
            return (Outcome::Cancelled, ready);
        }
        if let Some(outcome) = expired(limits, ready, started.elapsed()) {
            return (outcome, ready);
        }
        let candidate = output.poll(&limits.readiness);
        if output.failure.is_some() {
            return (Outcome::IoFailure, ready);
        }
        if cancellation.is_cancelled() {
            return (Outcome::Cancelled, ready);
        }
        match admit(limits, ready, (candidate, started.elapsed())) {
            Ok(admitted) => ready = admitted,
            Err(outcome) => return (outcome, ready),
        }
        let snapshot = match output.snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                output.failure = Some(error);
                return (Outcome::IoFailure, ready);
            }
        };
        match child.exited() {
            Ok(true) => {
                let candidate = match output.buffered_readiness(&limits.readiness, snapshot) {
                    Ok(candidate) => candidate,
                    Err(error) => {
                        output.failure = Some(error);
                        return (Outcome::IoFailure, ready);
                    }
                };
                if cancellation.is_cancelled() {
                    return (Outcome::Cancelled, ready);
                }
                return match admit(limits, ready, (candidate, started.elapsed())) {
                    Ok(ready) => (exit_outcome(&limits.readiness, ready), ready),
                    Err(outcome) => (outcome, ready),
                };
            }
            Ok(false) => (),
            Err(error) => {
                output.failure = Some(error);
                return (Outcome::IoFailure, ready);
            }
        }
        match limits.completion {
            Completion::Exit => (),
            Completion::StopAfterReady if ready => return (Outcome::Ready, ready),
            Completion::StopAfterReady => (),
        }
        if snapshot == (0, 0) {
            thread::sleep(control::POLL);
        }
    }
}

fn expired(limits: &Limits, ready: bool, elapsed: std::time::Duration) -> Option<Outcome> {
    if elapsed >= limits.execution {
        return Some(Outcome::Deadline);
    }
    match &limits.readiness {
        Readiness::Line { deadline, .. } | Readiness::ObservedLines { deadline, .. }
            if !ready && elapsed >= *deadline =>
        {
            Some(Outcome::ReadinessDeadline)
        }
        Readiness::None | Readiness::Line { .. } | Readiness::ObservedLines { .. } => None,
    }
}

fn exit_outcome(readiness: &Readiness, ready: bool) -> Outcome {
    match readiness {
        Readiness::None => Outcome::Exited,
        Readiness::Line { .. } if ready => Outcome::Exited,
        Readiness::Line { .. } => Outcome::EarlyExit,
        Readiness::ObservedLines { .. } => Outcome::EarlyExit,
    }
}

fn admit(
    limits: &Limits,
    ready: bool,
    observation: (bool, std::time::Duration),
) -> Result<bool, Outcome> {
    let (candidate, elapsed) = observation;
    match expired(limits, ready, elapsed) {
        Some(outcome) => Err(outcome),
        None => Ok(ready || candidate),
    }
}

#[cfg(test)]
#[path = "readiness_tests.rs"]
mod tests;

#[cfg(all(test, unix))]
#[path = "raw_file_tests.rs"]
mod raw_file_tests;
