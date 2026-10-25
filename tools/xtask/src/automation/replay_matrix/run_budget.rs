use crate::command::DynResult;
use std::time::{Duration, Instant};

pub(super) struct Budget {
    deadline: Option<Instant>,
}
impl Budget {
    pub(super) fn new(input: &super::run_workload::Input) -> Self {
        Self {
            deadline: (!input.context_qualification.is_mesh())
                .then(|| Instant::now() + Duration::from_secs(input.timeout_seconds)),
        }
    }
    pub(super) fn remaining(&self, maximum: Duration) -> DynResult<Duration> {
        let value = self.deadline.map_or(maximum, |deadline| {
            maximum.min(deadline.saturating_duration_since(Instant::now()))
        });
        if value.is_zero() {
            Err("captured replay run deadline exceeded".into())
        } else {
            Ok(value)
        }
    }
    pub(super) fn seconds(&self, maximum: u64) -> DynResult<u64> {
        let seconds = self.remaining(Duration::from_secs(maximum))?.as_secs();
        if seconds == 0 {
            Err("captured replay run has less than one second remaining".into())
        } else {
            Ok(seconds)
        }
    }
}
