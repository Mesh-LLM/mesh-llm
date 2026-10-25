use super::{Rejection, Session, Variant};
use crate::process::{GracefulRequest, Outcome, ProcessReport, ReadinessStop};
use std::fmt;

pub(crate) struct Completed {
    variant: Variant,
    primary: ProcessReport,
    headless: ProcessReport,
}

impl fmt::Debug for Completed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Completed")
            .field("variant", &self.variant)
            .finish_non_exhaustive()
    }
}

pub(crate) struct Rejected {
    pub(crate) reason: Rejection,
    pub(crate) primary: ProcessReport,
    pub(crate) headless: Option<ProcessReport>,
}

impl fmt::Debug for Rejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Rejected")
            .field("reason", &self.reason)
            .field("primary_outcome", &self.primary.outcome)
            .field(
                "headless_outcome",
                &self.headless.as_ref().map(|report| report.outcome),
            )
            .finish()
    }
}

impl fmt::Display for Rejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.reason)
    }
}

impl std::error::Error for Rejected {}

impl Session {
    pub(crate) fn finish(
        self,
        reports: (ProcessReport, Option<ProcessReport>),
        teardown: Result<(), Rejection>,
    ) -> Result<Completed, Box<Rejected>> {
        let (primary, headless) = reports;
        let result = teardown
            .and_then(|()| match self.completion() {
                Ok(()) | Err(Rejection::Incomplete) => Ok(()),
                Err(reason) => Err(reason),
            })
            .and_then(|()| process(&primary))
            .and_then(|()| match &headless {
                Some(report) if report.pid == primary.pid => Err(Rejection::ProcessIdentity),
                Some(report) => process(report),
                None => Err(Rejection::Incomplete),
            })
            .and_then(|()| self.completion());
        match (result, headless) {
            (Ok(()), Some(headless)) => Ok(Completed {
                variant: self.variant,
                primary,
                headless,
            }),
            (Ok(()), None) => Err(Box::new(Rejected {
                reason: Rejection::Incomplete,
                primary,
                headless: None,
            })),
            (Err(reason), headless) => Err(Box::new(Rejected {
                reason,
                primary,
                headless,
            })),
        }
    }
}

impl Completed {
    pub(crate) const fn variant(&self) -> Variant {
        self.variant
    }

    pub(crate) fn process_reports(&self) -> (&ProcessReport, &ProcessReport) {
        (&self.primary, &self.headless)
    }
}

fn process(report: &ProcessReport) -> Result<(), Rejection> {
    if report.pid == 0 {
        return Err(Rejection::ProcessIdentity);
    }
    match report.outcome {
        Outcome::Ready => (),
        Outcome::Exited | Outcome::EarlyExit => return Err(Rejection::EarlyExit),
        Outcome::Deadline | Outcome::ReadinessDeadline => return Err(Rejection::ProcessDeadline),
        Outcome::Cancelled => return Err(Rejection::Cancelled),
        Outcome::IoFailure | Outcome::ObservationRejected => return Err(Rejection::ProcessFailure),
    }
    match &report.readiness_stop {
        ReadinessStop::ProbeAdmitted {
            request: GracefulRequest::RequestedAfterLiveObservation,
            ..
        } => (),
        ReadinessStop::NotAdmitted
        | ReadinessStop::Admitted { .. }
        | ReadinessStop::ProbeAdmitted { .. } => return Err(Rejection::StopNotAdmitted),
    }
    if !report.ready
        || !report.success()
        || !report.cleanup.complete
        || report.cleanup.forced
        || report.cleanup.graceful_signal_failed
        || report.cleanup.failure.is_some()
        || report.status.is_none_or(|status| status.code() != Some(0))
    {
        return Err(Rejection::ProcessCleanup);
    }
    Ok(())
}
