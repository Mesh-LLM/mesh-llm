//! Shared cooperative cancellation for bounded maintenance mutations.
//!
//! A timeout cannot stop an already-running `spawn_blocking` task. Both the
//! async route and the blocking store operation therefore share this state:
//! the route marks it cancelled on timeout, and the store checks it before
//! every durable maintenance mutation.

use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use mesh_llm_log_store::MaintenanceExecutionControl;

use super::LogsError;

const CANCELLATION_GRACE: Duration = Duration::from_millis(50);

#[derive(Clone)]
pub(super) struct MaintenanceDeadline {
    deadline: Instant,
    cancelled: Arc<AtomicBool>,
}

impl MaintenanceDeadline {
    pub(super) fn new(time_cap: Duration) -> Self {
        Self {
            // Reserve a small interval for a cooperative store cancellation
            // to cross the blocking boundary before the HTTP route reports a
            // timeout.
            deadline: Instant::now() + time_cap.saturating_sub(CANCELLATION_GRACE),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

impl MaintenanceExecutionControl for MaintenanceDeadline {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || Instant::now() >= self.deadline
    }
}

pub(super) async fn timeout_maintenance<T>(
    time_cap: Duration,
    control: &MaintenanceDeadline,
    operation: impl Future<Output = Result<T, LogsError>>,
) -> Result<T, LogsError> {
    match tokio::time::timeout(time_cap, operation).await {
        Ok(result) => result,
        Err(_) => {
            control.cancel();
            Err(LogsError::MaintenanceCancelled)
        }
    }
}
