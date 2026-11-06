//! One execution-lane reservation owner shared by HTTP and binary frontends.
use anyhow::{Result, bail};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

#[derive(Debug)]
pub(crate) struct RuntimeLaneAdmission {
    permits: Arc<Semaphore>,
    changed: Condvar,
    waiting: Mutex<()>,
    capacity: usize,
}

#[derive(Debug)]
pub(crate) struct RuntimeLanePermit {
    permit: Option<OwnedSemaphorePermit>,
    owner: Arc<RuntimeLaneAdmission>,
}

impl Drop for RuntimeLanePermit {
    fn drop(&mut self) {
        drop(self.permit.take());
        self.owner.changed.notify_all();
    }
}

impl RuntimeLaneAdmission {
    pub(crate) fn new(capacity: u32) -> Arc<Self> {
        let capacity = usize::try_from(capacity).unwrap_or(usize::MAX).max(1);
        Arc::new(Self {
            permits: Arc::new(Semaphore::new(capacity)),
            changed: Condvar::new(),
            waiting: Mutex::new(()),
            capacity,
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) async fn acquire(self: &Arc<Self>) -> Result<RuntimeLanePermit> {
        let permit = self.permits.clone().acquire_owned().await?;
        Ok(RuntimeLanePermit {
            permit: Some(permit),
            owner: self.clone(),
        })
    }

    pub(crate) fn try_acquire(self: &Arc<Self>) -> Result<Option<RuntimeLanePermit>> {
        match self.permits.clone().try_acquire_owned() {
            Ok(permit) => Ok(Some(RuntimeLanePermit {
                permit: Some(permit),
                owner: self.clone(),
            })),
            Err(TryAcquireError::NoPermits) => Ok(None),
            Err(TryAcquireError::Closed) => bail!("runtime lane admission closed"),
        }
    }

    /// Never called while holding the native runtime or ownership-registry lock.
    pub(crate) fn acquire_blocking(
        self: &Arc<Self>,
        cancelled: impl Fn() -> bool,
    ) -> Result<RuntimeLanePermit> {
        let mut waiting = self
            .waiting
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if cancelled() {
                bail!("binary runtime lane admission cancelled");
            }
            if let Some(permit) = self.try_acquire()? {
                return Ok(permit);
            }
            waiting = self
                .changed
                .wait_timeout(waiting, Duration::from_millis(50))
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }
}

impl super::RuntimeState {
    pub(crate) fn lane_admission(&self) -> Arc<RuntimeLaneAdmission> {
        Arc::clone(&self.runtime_lane_admission)
    }
}
