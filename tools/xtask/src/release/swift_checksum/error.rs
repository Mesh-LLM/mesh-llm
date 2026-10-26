use crate::automation::command_interrupt::Reason;
use crate::process;
use crate::repository::check_report::CheckReport;

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error("Swift release artifact does not exist: {0}")]
    Artifact(String),
    #[error("Package.swift does not exist: {0}")]
    Manifest(String),
    #[error("Swift checksum I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("Swift checksum is not strict UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error(transparent)]
    Process(#[from] process::Failure),
    #[error("Swift checksum failed: {0:?}")]
    Native(Box<process::RawProcessReport>),
    #[error("Swift checksum returned no complete raw stdout")]
    MissingPayload,
    #[error(transparent)]
    Interrupt(#[from] Reason),
    #[error("{primary}; interruption scope finalization also failed: {reason}")]
    Finalization { primary: Box<Error>, reason: Reason },
}

impl Error {
    fn code(&self) -> i32 {
        match self {
            Self::Native(report) => report
                .process
                .status
                .and_then(|status| status.code())
                .filter(|code| *code != 0)
                .unwrap_or(1),
            Self::Finalization { primary, .. } => primary.code(),
            Self::Artifact(_)
            | Self::Manifest(_)
            | Self::Io(_)
            | Self::Utf8(_)
            | Self::Process(_)
            | Self::MissingPayload
            | Self::Interrupt(_) => 1,
        }
    }

    pub(super) fn report(&self) -> CheckReport {
        CheckReport {
            code: self.code(),
            stderr: format!("{self}\n"),
            stdout: String::new(),
        }
    }
}
