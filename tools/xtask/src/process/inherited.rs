//! Deadline/tree ownership with inherited stdin and live stdout/stderr. No capture.
use super::{
    Cancellation, Cleanup, Completion, Failure, Limits, Outcome, ProcessSpec, Readiness, control,
    platform,
};
use std::{
    process::{ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

pub struct InheritedReport {
    pub pid: u32,
    pub outcome: Outcome,
    pub status: Option<ExitStatus>,
    pub elapsed: Duration,
    pub cleanup: Cleanup,
    pub failure: Option<Failure>,
}

/// Output remains owned by inherited caller descriptors. This API does not
/// inspect, retain, redact, or report stream bytes; no pipe reader is created.
/// Readiness/capture observations are intentionally unsupported.
pub fn supervise_inherited(
    spec: &ProcessSpec,
    limits: &Limits,
    cancel: &Cancellation,
) -> Result<InheritedReport, Failure> {
    limits.validate()?;
    if !matches!(limits.readiness, Readiness::None)
        || !matches!(limits.completion, Completion::Exit)
    {
        return Err(Failure::InvalidSpec(
            "inherited streams require exit completion without readiness",
        ));
    }
    let mut command = spec.command()?;
    command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    if cancel.is_cancelled() {
        return Err(Failure::InvalidSpec("cancelled before inherited spawn"));
    }
    let started = Instant::now();
    let mut child = platform::OwnedChild::spawn_inherited(&mut command)?;
    let (outcome, failure) = monitor(&mut child, limits, cancel, started);
    let (cleanup, status) = control::shutdown(&mut child, limits, || {});
    Ok(InheritedReport {
        pid: child.id(),
        outcome,
        status,
        elapsed: started.elapsed(),
        cleanup,
        failure,
    })
}
fn monitor(
    child: &mut platform::OwnedChild,
    limits: &Limits,
    cancel: &Cancellation,
    started: Instant,
) -> (Outcome, Option<Failure>) {
    loop {
        if cancel.is_cancelled() {
            return (Outcome::Cancelled, None);
        }
        // Preserve a completed leader at the deadline, matching run_for's poll boundary.
        match child.exited() {
            Ok(true) => return (Outcome::Exited, None),
            Ok(false) => (),
            Err(error) => return (Outcome::IoFailure, Some(error)),
        }
        if cancel.is_cancelled() {
            return (Outcome::Cancelled, None);
        }
        if started.elapsed() >= limits.execution {
            return (Outcome::Deadline, None);
        }
        thread::sleep(control::POLL);
    }
}
#[cfg(all(test, unix))]
#[path = "inherited_tests.rs"]
mod tests;
