//! One deadline/cancellation boundary for process, filesystem and publication observations.
use super::provenance::{Error, Result};
use crate::process::Cancellation;
use std::time::{Duration, Instant};
pub(crate) struct Observation {
    pub(crate) cancellation: Cancellation,
    pub(crate) deadline: Instant,
}
impl Observation {
    pub(crate) fn new(cancellation: Cancellation, budget: Duration) -> Result<Self> {
        let deadline = Instant::now()
            .checked_add(budget)
            .ok_or_else(|| Error("release inventory observation budget invalid".into()))?;
        Ok(Self {
            cancellation,
            deadline,
        })
    }
    pub(crate) fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            return Err(Error("release inventory cancelled".into()));
        }
        if Instant::now() >= self.deadline {
            return Err(Error(
                "release inventory observation deadline expired".into(),
            ));
        }
        Ok(())
    }
}
