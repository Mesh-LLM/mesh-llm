use super::{
    Error, Rejection,
    observer::{Facts, Phase},
};
use crate::process::{GracefulRequest, Outcome, ProbeReport, ReadinessStop};
use std::fmt;

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/daemon/receipt.rs"]
mod tests;

#[derive(Debug)]
pub(crate) struct Rejected {
    cause: &'static str,
    process: crate::process::ProcessReport,
    observation: Option<Rejection>,
    facts: Facts,
}

#[derive(Debug)]
pub(super) struct Accepted {
    process: crate::process::ProcessReport,
    rejection: Option<Rejection>,
    facts: Facts,
}

impl Accepted {
    pub(super) fn finish(
        self,
        interruption: Result<(), crate::automation::command_interrupt::Reason>,
    ) -> Result<(), Error> {
        match interruption {
            Ok(()) => Ok(()),
            Err(reason) => Err(Error::Completion {
                cause: Box::new(Error::from(reason)),
                receipt: Box::new(Rejected {
                    cause: "command interruption scope failed after process cleanup",
                    process: self.process,
                    observation: self.rejection,
                    facts: self.facts,
                }),
            }),
        }
    }
}

impl fmt::Display for Rejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}; observation={:?}, facts={:?}, outcome={:?}, status={:?}, stop={:?}, cleanup={:?}, failure={:?}\nstdout: {}\nstderr: {}",
            self.cause,
            self.observation,
            self.facts,
            self.process.outcome,
            self.process.status,
            self.process.readiness_stop,
            self.process.cleanup,
            self.process.failure,
            String::from_utf8_lossy(&self.process.stdout.bytes_retained),
            String::from_utf8_lossy(&self.process.stderr.bytes_retained)
        )
    }
}

pub(super) fn accept(
    report: ProbeReport<Rejection>,
    facts: Facts,
    completion: (bool, bool),
) -> Result<Accepted, Error> {
    let (interrupted, joined) = completion;
    let cause = if !joined {
        Some("worker_failed")
    } else if interrupted {
        Some("cancelled by command interruption")
    } else {
        rejection(&report, &facts)
    };
    match cause {
        None => Ok(Accepted {
            process: report.process,
            rejection: report.rejection,
            facts,
        }),
        Some(cause) => Err(Error::Lifecycle(Box::new(Rejected {
            cause,
            process: report.process,
            observation: report.rejection,
            facts,
        }))),
    }
}

fn rejection(report: &ProbeReport<Rejection>, facts: &Facts) -> Option<&'static str> {
    if let Some(reason) = report.rejection {
        return Some(match reason {
            Rejection::MalformedStatus => "malformed_status",
            Rejection::OwnershipMismatch => "ownership_mismatch",
            Rejection::ModelsTransferFailed => "models_transfer_failed",
            Rejection::ResponseLimit => "response_limit",
            Rejection::WorkerFailed => "worker_failed",
        });
    }
    let process = &report.process;
    match process.outcome {
        Outcome::Ready => (),
        Outcome::Deadline | Outcome::ReadinessDeadline => {
            return Some(match facts.phase {
                Phase::Attribution => "attribution_unavailable",
                Phase::StatusRetry { .. }
                | Phase::StatusInFlight
                | Phase::ModelsInFlight
                | Phase::Complete => "readiness deadline expired",
            });
        }
        Outcome::Exited | Outcome::EarlyExit => return Some("daemon exited before readiness stop"),
        Outcome::Cancelled => return Some("daemon readiness cancelled"),
        Outcome::IoFailure => return Some("daemon observation failed"),
        Outcome::ObservationRejected => return Some("daemon observation rejected"),
    }
    match &process.readiness_stop {
        ReadinessStop::ProbeAdmitted {
            request: GracefulRequest::RequestedAfterLiveObservation,
            ..
        } => (),
        ReadinessStop::ProbeAdmitted { .. } => return Some("daemon live stop request rejected"),
        ReadinessStop::NotAdmitted | ReadinessStop::Admitted { .. } => {
            return Some("daemon requires probe admission");
        }
    }
    if facts.phase != Phase::Complete
        || facts.status.is_none()
        || facts.models.is_none()
        || !process.ready
        || !process.success()
        || process.status.is_none_or(|status| status.code() != Some(0))
    {
        return Some("daemon did not complete clean attributed unforced shutdown");
    }
    None
}
