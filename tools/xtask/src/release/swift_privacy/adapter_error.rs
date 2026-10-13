use super::{PrivacyError, native_lint::NativeFailure};
use crate::automation::command_interrupt::Reason;

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error("invalid swift-privacy arguments: {0}")]
    Arguments(&'static str),
    #[error(transparent)]
    Privacy(#[from] PrivacyError),
    #[error(transparent)]
    Native(#[from] NativeFailure),
    #[error("{primary}; additionally diagnostic output failed: {output}")]
    Diagnostic {
        primary: NativeFailure,
        output: std::io::Error,
    },
    #[error(transparent)]
    Interrupt(#[from] Reason),
    #[error("swift-privacy output failed: {0}")]
    Output(#[from] std::io::Error),
    #[error("{primary}; additionally signal finalization failed: {secondary}")]
    Finalization {
        primary: Box<Self>,
        secondary: Reason,
    },
}

pub(super) fn finish<T>(
    primary: Result<T, Error>,
    secondary: Result<(), Reason>,
) -> Result<T, Error> {
    match (primary, secondary) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(Error::Interrupt(error)),
        (Err(primary), Err(secondary)) => Err(Error::Finalization {
            primary: Box::new(primary),
            secondary,
        }),
    }
}
