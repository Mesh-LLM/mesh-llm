use super::{Error, error::Failure};
use crate::command_interrupt::Reason;

pub(super) fn finalize<T>(
    preceding: Result<T, Error>,
    finish: Result<(), Reason>,
) -> Result<T, Failure<T>> {
    match finish {
        Ok(()) => preceding.map_err(Failure::Operation),
        Err(reason) => Err(Failure::Finalization { reason, preceding }),
    }
}
