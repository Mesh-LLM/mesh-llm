use super::command_interrupt::{Interrupt, Reason};
use crate::process::{
    Failure, Limits,
    retained::{self, Coordinator, Report},
};
#[derive(thiserror::Error)]
pub(crate) enum Error<Rejection> {
    #[error(transparent)]
    Interrupt(#[from] Reason),
    #[error(transparent)]
    Process(#[from] Failure),
    #[error("retained session finalization failed: {reason}")]
    Finalization {
        #[source]
        reason: Reason,
        preceding: Result<Report<Rejection>, Failure>,
    },
}
impl<Rejection> std::fmt::Debug for Error<Rejection> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
    }
}
#[cfg(test)]
#[path = "retained_session/tests.rs"]
pub(crate) mod tests;
pub(crate) fn run<Observer: Coordinator>(
    observer: &mut Observer,
    limits: &Limits,
) -> Result<Report<Observer::Rejection>, Error<Observer::Rejection>> {
    let interrupt = Interrupt::install()?;
    let result = retained::run(observer, limits, &interrupt.cancellation());
    finalize(result, interrupt.finish())
}
fn finalize<Rejection>(
    result: Result<Report<Rejection>, Failure>,
    interruption: Result<(), Reason>,
) -> Result<Report<Rejection>, Error<Rejection>> {
    match interruption {
        Ok(()) => result.map_err(Error::Process),
        Err(reason) => Err(Error::Finalization {
            reason,
            preceding: result,
        }),
    }
}
