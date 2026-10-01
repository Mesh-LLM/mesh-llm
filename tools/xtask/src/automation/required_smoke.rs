//! Standalone smoke decisions only. No artifact, model, or platform qualification.

mod args;
mod command;
mod contract;
mod coordinated;
mod evidence;
mod execution;
mod failure;
mod http;
mod observer;
mod output;
mod overlap;
mod receipt;
mod session;
mod workers;

pub(crate) use command::{USAGE, run};

pub(crate) use contract::{Attestation, Budget, Check, Model, Stack, Variant};
pub(crate) use evidence::{Response, Transfer};
pub(crate) use receipt::{Completed, Rejected};
pub(crate) use session::{Progress, Session};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Rejection {
    #[error("invalid smoke budget")]
    InvalidBudget,
    #[error("smoke evidence arrived out of order")]
    OutOfOrder,
    #[error("smoke observation clock moved backwards")]
    ClockRegression,
    #[error("smoke deadline expired at {0:?}")]
    Deadline(Check),
    #[error("smoke transfer failed at {0:?}")]
    Transfer(Check),
    #[error("smoke response exceeded its byte bound at {0:?}")]
    ResponseLimit(Check),
    #[error("smoke evidence was rejected at {0:?}")]
    Evidence(Check),
    #[error("smoke checks are incomplete")]
    Incomplete,
    #[error("smoke cancelled")]
    Cancelled,
    #[error("smoke worker failed to join")]
    WorkerFailed,
    #[error("smoke private state cleanup failed")]
    StateCleanup,
    #[error("smoke process exited before completion")]
    EarlyExit,
    #[error("smoke process deadline expired")]
    ProcessDeadline,
    #[error("smoke process observation failed")]
    ProcessFailure,
    #[error("smoke process requires an admitted live probe stop")]
    StopNotAdmitted,
    #[error("smoke process did not complete clean unforced shutdown")]
    ProcessCleanup,
    #[error("standalone and headless process receipts must have distinct live identities")]
    ProcessIdentity,
}

#[cfg(test)]
#[path = "../../tests/migration_required_smoke/mod.rs"]
mod migration_required_smoke;
