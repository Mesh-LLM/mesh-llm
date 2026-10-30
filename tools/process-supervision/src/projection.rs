//! Complete bounded line projection for finite tool output; never readiness admission.
use super::{
    Cancellation, Completion, Failure, Limits, ObservedLine, Outcome, OutputFiles, ProcessReport,
    ProcessSpec, RawCaptureOptions, Readiness, control,
};
use std::time::Instant;

pub(super) type Callback<'a> = Option<&'a mut dyn FnMut(ObservedLine<'_>)>;

/// Trusted owner receives borrowed lines of at most 8192 bytes, including during
/// post-exit draining and owned-tree cleanup. Do not block, do I/O/logging, panic,
/// signal processes or retain raw lines. Keep only bounded typed facts. A parser
/// rejection must remain latched until the caller admits the complete report.
/// There are no output files, raw byte capture or retained diagnostics. The
/// caller must require successful exit/cleanup AND complete line capture on both
/// streams before using its facts. Retention truncation is intentional here.
pub fn supervise_projected(
    spec: &ProcessSpec,
    limits: &Limits,
    cancellation: &Cancellation,
    projection: &mut dyn FnMut(ObservedLine<'_>),
) -> Result<ProcessReport, Failure> {
    if !matches!(limits.readiness, Readiness::None)
        || !matches!(limits.completion, Completion::Exit)
        || limits.retained_bytes_per_stream != 0
    {
        return Err(Failure::InvalidSpec(
            "projection requires None/Exit and zero retention",
        ));
    }
    super::supervisor::supervise_configured(
        spec,
        limits,
        cancellation,
        (OutputFiles::default(), RawCaptureOptions::default()),
        Some(projection),
        |child, output, started, projection| {
            (
                monitor(child, output, limits, cancellation, started, projection),
                false,
                None,
            )
        },
    )
    .map(|report| report.process)
}

pub(super) fn poll(output: &mut super::output::Output, projection: &mut Callback<'_>) {
    match projection {
        Some(project) => {
            if let Err(error) = output.poll_captured(&mut |line, _| project(line)) {
                output.failure.get_or_insert(error);
            }
        }
        None => {
            output.poll(&Readiness::None);
        }
    }
}

fn monitor(
    child: &mut super::platform::OwnedChild,
    output: &mut super::output::Output,
    limits: &Limits,
    cancellation: &Cancellation,
    started: Instant,
    projection: &mut Callback<'_>,
) -> Outcome {
    loop {
        if cancellation.is_cancelled() {
            return Outcome::Cancelled;
        }
        if started.elapsed() >= limits.execution {
            return Outcome::Deadline;
        }
        poll(output, projection);
        if output.failure.is_some() {
            return Outcome::IoFailure;
        }
        if cancellation.is_cancelled() {
            return Outcome::Cancelled;
        }
        if started.elapsed() >= limits.execution {
            return Outcome::Deadline;
        }
        match child.exited() {
            Ok(true) => return Outcome::Exited,
            Ok(false) => (),
            Err(error) => {
                output.failure = Some(error);
                return Outcome::IoFailure;
            }
        }
        match output.snapshot() {
            Ok((0, 0)) => std::thread::sleep(control::POLL),
            Ok(_) => (),
            Err(error) => {
                output.failure = Some(error);
                return Outcome::IoFailure;
            }
        }
    }
}

#[cfg(all(test, unix))]
#[path = "projection_tests.rs"]
mod tests;
