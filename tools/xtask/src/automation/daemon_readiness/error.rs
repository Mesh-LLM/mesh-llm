use super::super::{command_interrupt, private_state};
use crate::process;
use std::io;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("daemon readiness: {0}")]
    Invalid(&'static str),
    #[error("daemon readiness {operation} failed ({kind:?}, OS code {code:?})")]
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
        code: Option<i32>,
    },
    #[error("daemon readiness: {0}")]
    Process(#[from] process::Failure),
    #[error("daemon readiness: {0}")]
    Lifecycle(Box<super::receipt::Rejected>),
    #[error("daemon readiness completion failed: {cause}; process history: {receipt}")]
    Completion {
        cause: Box<Error>,
        receipt: Box<super::receipt::Rejected>,
    },
    #[error("{preceding}; interruption teardown failed: {cause}")]
    Interruption {
        preceding: Box<Error>,
        cause: Box<Error>,
    },
    #[error("worker_failed; preceding failure: {0:?}")]
    WorkerJoin(Option<Box<Error>>),
    #[error(
        "private state deletion failed ({kind:?}, OS code {code:?}); preceding failure: {preceding:?}"
    )]
    StateDeletion {
        kind: io::ErrorKind,
        code: Option<i32>,
        preceding: Option<Box<Error>>,
    },
}

impl Error {
    pub(super) fn io(operation: &'static str, error: io::Error) -> Self {
        Self::Io {
            operation,
            kind: error.kind(),
            code: error.raw_os_error(),
        }
    }
}

impl From<private_state::Error> for Error {
    fn from(error: private_state::Error) -> Self {
        match error {
            private_state::Error::EntropyUnavailable => {
                Self::Invalid("private state entropy unavailable")
            }
            #[cfg(windows)]
            private_state::Error::Invalid(message) => Self::Invalid(message),
            private_state::Error::Io {
                operation,
                kind,
                code,
            } => Self::Io {
                operation,
                kind,
                code,
            },
        }
    }
}

impl From<command_interrupt::Reason> for Error {
    fn from(reason: command_interrupt::Reason) -> Self {
        use command_interrupt::Reason;
        match reason {
            Reason::ScopeBusy => Self::Invalid("signal scope already active"),
            Reason::Interrupted => Self::Invalid("cancelled by command interruption"),
            #[cfg(unix)]
            Reason::ExistingHandler => Self::Invalid("refuses an existing signal handler"),
            #[cfg(unix)]
            Reason::BlockedSigint => Self::Invalid("refuses blocked SIGINT"),
            #[cfg(unix)]
            Reason::BlockedSigterm => Self::Invalid("refuses blocked SIGTERM"),
            Reason::Io {
                operation,
                kind,
                code,
            } => Self::Io {
                operation,
                kind,
                code,
            },
        }
    }
}
