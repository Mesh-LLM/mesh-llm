use crate::process::{Failure, ProcessReport};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ExpectedExit {
    statuses: Vec<i32>,
    deadline: Duration,
}

impl ExpectedExit {
    pub fn new(statuses: &[i32], deadline: Duration) -> Result<Self, Failure> {
        if statuses.is_empty() || statuses.len() > 256 || deadline.is_zero() {
            return Err(Failure::InvalidSpec(
                "expected exit needs 1..=256 statuses and a deadline",
            ));
        }
        let mut statuses = statuses.to_vec();
        statuses.sort_unstable();
        statuses.dedup();
        Ok(Self { statuses, deadline })
    }

    pub const fn deadline(&self) -> Duration {
        self.deadline
    }

    pub(super) fn receipt(&self, elapsed: Duration, process: &ProcessReport) -> CompletionReceipt {
        CompletionReceipt {
            statuses: self.statuses.clone(),
            deadline: self.deadline,
            elapsed,
            status: process.status.and_then(|status| status.code()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionReceipt {
    pub statuses: Vec<i32>,
    pub deadline: Duration,
    pub elapsed: Duration,
    pub status: Option<i32>,
}

impl CompletionReceipt {
    pub fn accepted(&self, process: &ProcessReport) -> bool {
        self.elapsed < self.deadline
            && self
                .status
                .is_some_and(|status| self.statuses.contains(&status))
            && process.cleanup.complete
            && !process.cleanup.forced
            && !process.cleanup.graceful_signal_failed
            && process.cleanup.failure.is_none()
            && process.failure.is_none()
    }
}
