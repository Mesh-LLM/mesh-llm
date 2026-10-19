//! Avoid interleaving independent bulk prefills in the unified KV allocator.
//! Decode can still run between each prefill's bounded native chunks.
use skippy_inference_api::{CancellationToken, InferenceError, InferenceResult};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
pub(super) struct PrefillAdmission {
    waiting: Mutex<VecDeque<Arc<()>>>,
    changed: Condvar,
}

impl PrefillAdmission {
    pub(super) fn acquire(
        &self,
        deadline: Instant,
        cancellation: Option<&CancellationToken>,
    ) -> InferenceResult<PrefillTurn<'_>> {
        let turn = PrefillTurn {
            admission: self,
            ticket: Arc::new(()),
        };
        let mut waiting = self
            .waiting
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        waiting.push_back(turn.ticket.clone());
        self.changed.notify_all();
        loop {
            let error = if cancellation.is_some_and(CancellationToken::is_cancelled) {
                Some(InferenceError::cancelled(
                    "request cancelled waiting for prefill",
                ))
            } else if Instant::now() >= deadline {
                Some(InferenceError::timeout(
                    "cache operation deadline exceeded waiting for prefill",
                ))
            } else {
                None
            };
            if let Some(error) = error {
                drop(waiting);
                return Err(error);
            }
            if waiting
                .front()
                .is_some_and(|first| Arc::ptr_eq(first, &turn.ticket))
            {
                return Ok(turn);
            }
            let wait = deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(10));
            (waiting, _) = self
                .changed
                .wait_timeout(waiting, wait)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

pub(in crate::frontend) struct PrefillTurn<'a> {
    admission: &'a PrefillAdmission,
    ticket: Arc<()>,
}

impl Drop for PrefillTurn<'_> {
    fn drop(&mut self) {
        let mut waiting = self
            .admission
            .waiting
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        waiting.retain(|ticket| !Arc::ptr_eq(ticket, &self.ticket));
        self.admission.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    fn wait_for_queue(admission: &PrefillAdmission, count: usize) {
        let limit = deadline();
        let mut waiting = admission.waiting.lock().unwrap();
        while waiting.len() != count {
            assert!(Instant::now() < limit, "prefill waiter did not enqueue");
            (waiting, _) = admission
                .changed
                .wait_timeout(waiting, Duration::from_millis(10))
                .unwrap();
        }
    }

    #[test]
    fn prefills_are_fifo_without_holding_the_runtime_or_scheduler_worker() {
        let admission = PrefillAdmission::default();
        let first = admission.acquire(deadline(), None).unwrap();
        let (acquired, events) = mpsc::channel();
        let (release, released) = mpsc::channel();
        std::thread::scope(|scope| {
            let admission = &admission;
            let acquired_second = acquired.clone();
            scope.spawn(move || {
                let _turn = admission.acquire(deadline(), None).unwrap();
                acquired_second.send(2).unwrap();
                released.recv_timeout(Duration::from_secs(5)).unwrap();
            });
            wait_for_queue(admission, 2);
            scope.spawn(move || {
                let _turn = admission.acquire(deadline(), None).unwrap();
                acquired.send(3).unwrap();
            });
            wait_for_queue(admission, 3);
            assert!(events.try_recv().is_err());
            drop(first);
            assert_eq!(events.recv_timeout(Duration::from_secs(5)).unwrap(), 2);
            assert!(events.try_recv().is_err());
            release.send(()).unwrap();
            assert_eq!(events.recv_timeout(Duration::from_secs(5)).unwrap(), 3);
        });
        assert!(admission.waiting.lock().unwrap().is_empty());
    }

    #[test]
    fn cancelled_waiter_releases_its_ticket_and_next_waiter_can_run() {
        let admission = PrefillAdmission::default();
        let first = admission.acquire(deadline(), None).unwrap();
        let cancellation = CancellationToken::new();
        std::thread::scope(|scope| {
            let waiter =
                scope.spawn(|| admission.acquire(deadline(), Some(&cancellation)).is_err());
            wait_for_queue(&admission, 2);
            cancellation.cancel();
            assert!(waiter.join().unwrap());
        });
        assert_eq!(admission.waiting.lock().unwrap().len(), 1);
        drop(first);
        assert!(admission.acquire(deadline(), None).is_ok());
    }

    #[test]
    fn timeout_and_unwind_release_prefill_ownership() {
        let admission = PrefillAdmission::default();
        let first = admission.acquire(deadline(), None).unwrap();
        assert!(
            admission
                .acquire(Instant::now() + Duration::from_millis(1), None)
                .is_err()
        );
        assert_eq!(admission.waiting.lock().unwrap().len(), 1);
        drop(first);
        let _ = std::panic::catch_unwind(|| {
            let _turn = admission.acquire(deadline(), None).unwrap();
            panic!("request unwound");
        });
        assert!(admission.waiting.lock().unwrap().is_empty());
        assert!(admission.acquire(deadline(), None).is_ok());
    }
}
