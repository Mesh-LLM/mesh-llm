use super::{Error, Interrupt};
use crate::automation::command_interrupt::Reason;

#[derive(Debug, thiserror::Error)]
pub(super) enum Failure<T: std::fmt::Debug> {
    #[error(transparent)]
    Operation(#[from] Error),
    #[error("generator finalization failed: {reason}; preceding={preceding:?}")]
    Finalization {
        #[source]
        reason: Reason,
        preceding: Result<T, Error>,
    },
}

pub(super) fn run<T: std::fmt::Debug>(
    operation: impl FnOnce(&Interrupt) -> Result<T, Error>,
) -> Result<T, Failure<T>> {
    let interrupt = Interrupt::install().map_err(Error::Interrupt)?;
    let preceding = operation(&interrupt);
    finalize(preceding, interrupt.finish())
}

pub(super) fn finalize<T: std::fmt::Debug>(
    preceding: Result<T, Error>,
    finish: Result<(), Reason>,
) -> Result<T, Failure<T>> {
    match finish {
        Ok(()) => preceding.map_err(Failure::Operation),
        Err(reason) => Err(Failure::Finalization { reason, preceding }),
    }
}
