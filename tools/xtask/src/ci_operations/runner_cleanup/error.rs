use std::{io, path::PathBuf};

#[derive(thiserror::Error)]
pub(crate) enum Failure<T> {
    #[error(transparent)]
    Operation(#[from] Error),
    #[error("cleanup finalization failed: {reason}; preceding={summary}", summary = Summary(preceding))]
    Finalization {
        #[source]
        reason: crate::command_interrupt::Reason,
        preceding: Result<T, Error>,
    },
}

impl<T> std::fmt::Debug for Failure<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Operation(error) => std::fmt::Debug::fmt(error, formatter),
            Self::Finalization { reason, preceding } => formatter
                .debug_struct("Finalization")
                .field("reason", reason)
                .field("preceding", &format_args!("{}", Summary(preceding)))
                .finish(),
        }
    }
}

struct Summary<'a, T>(&'a Result<T, Error>);

impl<T> std::fmt::Display for Summary<'_, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Ok(_) => formatter.write_str("Ok(retained)"),
            Err(Error::Git(report)) => write!(
                formatter,
                "Git(pid={}, outcome={:?}, status={:?}, cleanup={:?}, failure={:?})",
                report.pid, report.outcome, report.status, report.cleanup, report.failure
            ),
            Err(Error::Io(error)) => write!(
                formatter,
                "Io(kind={:?}, OS code={:?})",
                error.kind(),
                error.raw_os_error()
            ),
            Err(Error::Interrupt(reason)) => write!(formatter, "Interrupt({reason})"),
            Err(Error::Process(error)) => write!(formatter, "Process({error})"),
            Err(Error::Missing(_)) => formatter.write_str("Missing(retained)"),
            Err(Error::Input(_)) => formatter.write_str("Input(retained)"),
            Err(Error::Escape(_)) => formatter.write_str("Escape(retained)"),
            Err(Error::ParentSymlink(_)) => formatter.write_str("ParentSymlink(retained)"),
            Err(Error::ReplayRoot(_)) => formatter.write_str("ReplayRoot(retained)"),
            Err(Error::ReplaySymlink(_)) => formatter.write_str("ReplaySymlink(retained)"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("missing required environment: {0}")]
    Missing(&'static str),
    #[error("{0}")]
    Input(&'static str),
    #[error("cleanup escaped its owning directory: {0}")]
    Escape(PathBuf),
    #[error("cleanup parent is a symlink: {0}")]
    ParentSymlink(PathBuf),
    #[error("replay worktree root is a symlink: {0}")]
    ReplayRoot(PathBuf),
    #[error("replay worktree is a symlink: {0}")]
    ReplaySymlink(PathBuf),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Interrupt(#[from] crate::command_interrupt::Reason),
    #[error(transparent)]
    Process(#[from] crate::process::Failure),
    #[error("Git worktree command failed: {0:?}")]
    Git(Box<crate::process::ProcessReport>),
}
