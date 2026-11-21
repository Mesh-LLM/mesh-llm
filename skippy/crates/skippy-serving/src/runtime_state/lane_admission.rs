//! One execution-lane reservation owner shared by HTTP and binary frontends.
use anyhow::{Result, bail};
use std::future::Future;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

#[derive(Debug)]
pub(crate) struct RuntimeLaneAdmission {
    permits: Arc<Semaphore>,
    capacity: usize,
}

#[derive(Debug)]
pub(crate) struct RuntimeLanePermit {
    _permit: OwnedSemaphorePermit,
}

#[derive(Default)]
struct BlockingLaneWaiter {
    notified: Mutex<bool>,
    changed: Condvar,
}

impl Wake for BlockingLaneWaiter {
    fn wake(self: Arc<Self>) {
        *self
            .notified
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        self.changed.notify_one();
    }
}

impl BlockingLaneWaiter {
    fn wait(&self) {
        let notified = self
            .notified
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut notified, _) = self
            .changed
            .wait_timeout_while(notified, Duration::from_millis(50), |notified| !*notified)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *notified = false;
    }
}

impl RuntimeLaneAdmission {
    pub(crate) fn new(capacity: u32) -> Arc<Self> {
        let capacity = usize::try_from(capacity).unwrap_or(usize::MAX).max(1);
        Arc::new(Self {
            permits: Arc::new(Semaphore::new(capacity)),
            capacity,
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) async fn acquire(self: &Arc<Self>) -> Result<RuntimeLanePermit> {
        let permit = self.permits.clone().acquire_owned().await?;
        Ok(RuntimeLanePermit { _permit: permit })
    }

    pub(crate) fn try_acquire(self: &Arc<Self>) -> Result<Option<RuntimeLanePermit>> {
        match self.permits.clone().try_acquire_owned() {
            Ok(permit) => Ok(Some(RuntimeLanePermit { _permit: permit })),
            Err(TryAcquireError::NoPermits) => Ok(None),
            Err(TryAcquireError::Closed) => bail!("runtime lane admission closed"),
        }
    }

    /// Never called while holding the native runtime or ownership-registry lock.
    pub(crate) fn acquire_blocking(
        self: &Arc<Self>,
        cancelled: impl Fn() -> bool,
    ) -> Result<RuntimeLanePermit> {
        // Keep one queued future across cancellation polls so blocking peers
        // retain their FIFO position alongside asynchronous HTTP waiters.
        let mut acquisition = std::pin::pin!(self.permits.clone().acquire_owned());
        let waiter = Arc::new(BlockingLaneWaiter::default());
        let waker = Waker::from(Arc::clone(&waiter));
        let mut context = Context::from_waker(&waker);
        loop {
            if cancelled() {
                bail!("binary runtime lane admission cancelled");
            }
            match acquisition.as_mut().poll(&mut context) {
                Poll::Ready(permit) => return Ok(RuntimeLanePermit { _permit: permit? }),
                Poll::Pending => waiter.wait(),
            }
        }
    }
}

impl super::RuntimeState {
    pub(crate) fn lane_admission(&self) -> Arc<RuntimeLaneAdmission> {
        Arc::clone(&self.runtime_lane_admission)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    };
    use std::thread;

    struct BlockingAcquisition {
        queued: mpsc::Receiver<()>,
        finished: mpsc::Receiver<bool>,
        worker: thread::JoinHandle<Result<RuntimeLanePermit>>,
    }

    fn queue_blocking(
        lanes: Arc<RuntimeLaneAdmission>,
        cancelled: Arc<AtomicBool>,
    ) -> BlockingAcquisition {
        let (queued_tx, queued) = mpsc::channel();
        let (finished_tx, finished) = mpsc::channel();
        let worker = thread::spawn(move || {
            let polls = AtomicUsize::new(0);
            let result = lanes.acquire_blocking(|| {
                // The second cancellation poll follows a Pending poll of the
                // acquisition future, so its FIFO position already exists.
                if polls.fetch_add(1, Ordering::AcqRel) == 1 {
                    let _ = queued_tx.send(());
                }
                cancelled.load(Ordering::Acquire)
            });
            let _ = finished_tx.send(result.is_ok());
            result
        });
        BlockingAcquisition {
            queued,
            finished,
            worker,
        }
    }

    #[tokio::test]
    async fn blocking_waiter_keeps_fifo_position_between_async_waiters() {
        let lanes = RuntimeLaneAdmission::new(1);
        let active = lanes.try_acquire().unwrap().unwrap();
        let mut older_http = Box::pin(lanes.acquire());
        assert!(futures_util::poll!(&mut older_http).is_pending());
        let binary = queue_blocking(lanes.clone(), Arc::new(AtomicBool::new(false)));
        binary.queued.recv_timeout(Duration::from_secs(5)).unwrap();
        let mut newer_http = Box::pin(lanes.acquire());
        assert!(futures_util::poll!(&mut newer_http).is_pending());

        drop(active);
        let older = tokio::time::timeout(Duration::from_secs(5), older_http)
            .await
            .unwrap()
            .unwrap();
        assert!(binary.finished.try_recv().is_err());
        assert!(futures_util::poll!(&mut newer_http).is_pending());
        drop(older);
        assert!(
            binary
                .finished
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
        );
        assert!(futures_util::poll!(&mut newer_http).is_pending());
        let binary = binary.worker.join().unwrap().unwrap();
        drop(binary);
        let newer = tokio::time::timeout(Duration::from_secs(5), newer_http)
            .await
            .unwrap()
            .unwrap();
        drop(newer);
        assert!(lanes.try_acquire().unwrap().is_some());
    }

    #[tokio::test]
    async fn cancelled_blocking_waiter_removes_its_fifo_reservation() {
        let lanes = RuntimeLaneAdmission::new(1);
        let active = lanes.try_acquire().unwrap().unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let binary = queue_blocking(lanes.clone(), cancelled.clone());
        binary.queued.recv_timeout(Duration::from_secs(5)).unwrap();
        let mut http = Box::pin(lanes.acquire());
        assert!(futures_util::poll!(&mut http).is_pending());
        cancelled.store(true, Ordering::Release);
        assert!(
            !binary
                .finished
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
        );
        assert!(binary.worker.join().unwrap().is_err());
        drop(active);
        let admitted = tokio::time::timeout(Duration::from_secs(5), http)
            .await
            .unwrap()
            .unwrap();
        drop(admitted);
        assert!(lanes.try_acquire().unwrap().is_some());
    }

    #[tokio::test]
    async fn blocking_acquisition_can_be_polled_inside_an_existing_runtime() {
        let lanes = RuntimeLaneAdmission::new(1);
        let permit = lanes.acquire_blocking(|| false).unwrap();
        assert!(lanes.try_acquire().unwrap().is_none());
        drop(permit);
        assert!(lanes.try_acquire().unwrap().is_some());
    }
}
