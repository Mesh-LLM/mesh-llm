use super::{Rejection, output::Receipt};
use crate::{automation::command_interrupt::Reason, command::DynResult, process};
use std::{error::Error, fmt, io};

pub(super) enum Failure {
    Coordinated {
        report: process::retained::Report<Rejection>,
    },
    CoordinatedWorkers {
        preceding: DynResult<process::retained::Report<Rejection>>,
        joined: Vec<(bool, bool)>,
    },
    CoordinatedState {
        kind: io::ErrorKind,
        code: Option<i32>,
        preceding: DynResult<Receipt>,
    },
    Interrupt {
        reason: Reason,
        preceding: DynResult<Receipt>,
    },
}

impl Failure {
    pub(super) fn reason(&self) -> String {
        match self {
            Self::Coordinated { .. } => Rejection::ProcessFailure.to_string(),
            Self::CoordinatedWorkers { .. } => Rejection::WorkerFailed.to_string(),
            Self::CoordinatedState { .. } => Rejection::StateCleanup.to_string(),
            Self::Interrupt { reason, .. } => match reason {
                Reason::Interrupted => Rejection::Cancelled.to_string(),
                Reason::ScopeBusy | Reason::Io { .. } => "signal_scope_finalization_failed".into(),
                #[cfg(unix)]
                Reason::ExistingHandler | Reason::BlockedSigint | Reason::BlockedSigterm => {
                    "signal_scope_finalization_failed".into()
                }
            },
        }
    }

    pub(super) fn reports(&self) -> Vec<&process::ProcessReport> {
        match self {
            Self::Coordinated { report } => report
                .members
                .iter()
                .map(|member| &member.process)
                .collect(),
            Self::CoordinatedWorkers { preceding, .. } => match preceding {
                Ok(report) => report
                    .members
                    .iter()
                    .map(|member| &member.process)
                    .collect(),
                Err(error) => super::output::reports(error.as_ref()),
            },
            Self::CoordinatedState { preceding, .. } => match preceding {
                Ok(receipt) => receipt.reports(),
                Err(error) => super::output::reports(error.as_ref()),
            },
            Self::Interrupt { preceding, .. } => match preceding {
                Ok(receipt) => receipt.reports(),
                Err(error) => super::output::reports(error.as_ref()),
            },
        }
    }
}

impl fmt::Debug for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("SmokeFailure");
        debug.field("reason", &self.reason());
        if let Self::CoordinatedState { kind, code, .. } = self {
            debug.field("kind", kind).field("code", code);
        }
        if let Self::CoordinatedWorkers { joined, .. } = self {
            debug.field("joined", joined);
        }
        debug.finish()
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.reason())
    }
}

impl Error for Failure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Coordinated { report } => {
                report.failure.as_ref().map(|error| error as &dyn Error)
            }
            Self::CoordinatedWorkers { preceding, .. } => {
                preceding.as_ref().err().map(|error| error.as_ref())
            }
            Self::CoordinatedState { preceding, .. } => {
                preceding.as_ref().err().map(|error| error.as_ref())
            }
            Self::Interrupt { reason, .. } => Some(reason),
        }
    }
}
