use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("{0}")]
    Contract(String),
    #[error("{}: {source}", .path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid XCFramework Info.plist: {0}")]
    Plist(#[from] plist::Error),
    #[error("failed to inspect XCFramework binary with lipo: {}: {source}", .binary.display())]
    Spawn {
        binary: PathBuf,
        source: crate::process::Failure,
    },
    #[error("failed to inspect XCFramework binary with lipo: {}: outcome={:?}, status={:?}, cleanup={}", .binary.display(), .report.process.outcome, .report.process.status, .report.process.cleanup.complete)]
    Native {
        binary: PathBuf,
        report: Box<crate::process::RawProcessReport>,
    },
    #[error("lipo output is not UTF-8: {0}")]
    Encoding(#[from] std::str::Utf8Error),
    #[error("{0}")]
    Interrupt(#[from] crate::command_interrupt::Reason),
    #[error("{primary}; interruption finalization: {reason}")]
    Finalization {
        primary: Box<Error>,
        reason: crate::command_interrupt::Reason,
    },
}

impl Error {
    pub(super) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_owned(),
            source,
        }
    }
}

pub(super) fn finalize<T>(
    result: Result<T, Error>,
    finish: Result<(), crate::command_interrupt::Reason>,
) -> Result<T, Error> {
    match (result, finish) {
        (result, Ok(())) => result,
        (Ok(_), Err(reason)) => Err(Error::Interrupt(reason)),
        (Err(primary), Err(reason)) => Err(Error::Finalization {
            primary: Box::new(primary),
            reason,
        }),
    }
}
