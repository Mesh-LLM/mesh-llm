use super::Error;
use crate::process::{GracefulRequest, Outcome, ProcessReport, ReadinessStop};
use std::fmt;

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/receipt.rs"]
mod tests;

#[derive(Debug)]
pub(crate) struct Rejected {
    cause: &'static str,
    report: ProcessReport,
}

impl fmt::Display for Rejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}; outcome={:?}, status={:?}, stop={:?}, cleanup={:?}, failure={:?}\nstdout: {}\nstderr: {}",
            self.cause,
            self.report.outcome,
            self.report.status,
            self.report.readiness_stop,
            self.report.cleanup,
            self.report.failure,
            String::from_utf8_lossy(&self.report.stdout.bytes_retained),
            String::from_utf8_lossy(&self.report.stderr.bytes_retained),
        )
    }
}

pub(crate) fn accept(report: ProcessReport) -> Result<(), Error> {
    let cause = rejection(&report);
    match cause {
        None => Ok(()),
        Some(cause) => Err(Error::Lifecycle(Box::new(Rejected { cause, report }))),
    }
}

pub(crate) fn interrupted(report: ProcessReport) -> Error {
    Error::Lifecycle(Box::new(Rejected {
        cause: "client readiness cancelled by command interruption",
        report,
    }))
}

fn rejection(report: &ProcessReport) -> Option<&'static str> {
    match report.outcome {
        Outcome::Ready => (),
        Outcome::Deadline | Outcome::ReadinessDeadline => {
            return Some("readiness deadline expired");
        }
        Outcome::Exited | Outcome::EarlyExit => return Some("client exited before readiness stop"),
        Outcome::Cancelled => return Some("client readiness cancelled"),
        Outcome::IoFailure => return Some("client observation failed"),
        Outcome::ObservationRejected => return Some("client observation rejected"),
    }
    match &report.readiness_stop {
        ReadinessStop::NotAdmitted => return Some("readiness stop not admitted"),
        ReadinessStop::ProbeAdmitted { .. } => {
            return Some("client requires line readiness receipt");
        }
        ReadinessStop::Admitted { request, .. } => match request {
            GracefulRequest::RequestedAfterLiveObservation => (),
            GracefulRequest::LeaderExitedBeforeRequest => {
                return Some("client exited before stop request");
            }
            GracefulRequest::LeaderObservationFailed(_) => {
                return Some("leader observation failed");
            }
            GracefulRequest::SkippedInactiveTree => return Some("graceful request was skipped"),
            GracefulRequest::TreeObservationFailed => return Some("tree observation failed"),
            GracefulRequest::RequestFailed(_) => return Some("graceful request failed"),
        },
    }
    if !report.ready
        || report.status.is_none_or(|status| status.code() != Some(0))
        || !report.cleanup.complete
        || report.cleanup.forced
        || report.cleanup.graceful_signal_failed
        || report.cleanup.failure.is_some()
        || report.failure.is_some()
    {
        return Some("client did not complete clean unforced shutdown");
    }
    None
}
