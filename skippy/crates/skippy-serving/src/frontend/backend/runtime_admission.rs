//! Bound frontend work by the execution lanes shared with binary stage clients.
use super::{StageOpenAiBackend, generation_queue_timeout_error, request_cancelled_error};
use crate::runtime_state::lane_admission::RuntimeLanePermit;
use skippy_inference_api::{CancellationToken, InferenceError, InferenceResult};
use std::time::Instant;

impl StageOpenAiBackend {
    pub(super) async fn acquire_runtime_lane(
        &self,
        admission_started: Instant,
        cancellation: &CancellationToken,
    ) -> InferenceResult<RuntimeLanePermit> {
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        let deadline = if self.generation_admission_timeout.is_zero() {
            None
        } else {
            Some(
                admission_started
                    .checked_add(self.generation_admission_timeout)
                    .ok_or_else(|| {
                        InferenceError::backend("runtime lane admission deadline overflow")
                    })?,
            )
        };
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(generation_queue_timeout_error(
                self.generation_admission_timeout,
            ));
        }
        let timeout = async {
            if let Some(deadline) = deadline {
                tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
            } else {
                std::future::pending::<()>().await;
            }
        };
        tokio::select! {
            biased;
            () = cancellation.cancelled() => Err(request_cancelled_error()),
            () = timeout => Err(generation_queue_timeout_error(self.generation_admission_timeout)),
            permit = self.runtime_lane_admission.acquire() => permit.map_err(|error| InferenceError::backend(format!("{error:#}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_state::RuntimeState;
    use std::{
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
    };

    struct HeldRuntime {
        released: Arc<AtomicBool>,
        release: Option<mpsc::Sender<()>>,
        worker: Option<thread::JoinHandle<()>>,
    }

    impl Drop for HeldRuntime {
        fn drop(&mut self) {
            if let Some(release) = self.release.take() {
                let _ = release.send(());
            }
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }

    fn hold_runtime(runtime: Arc<Mutex<RuntimeState>>) -> HeldRuntime {
        let (held_tx, held_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let released = Arc::new(AtomicBool::new(false));
        let observed = released.clone();
        let worker = thread::spawn(move || {
            let runtime = runtime.lock().unwrap();
            held_tx.send(()).unwrap();
            // A regressed synchronous lock releases eventually and fails the
            // state assertion, instead of hanging this single-thread runtime.
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
            drop(runtime);
            observed.store(true, Ordering::Release);
        });
        held_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        HeldRuntime {
            released,
            release: Some(release_tx),
            worker: Some(worker),
        }
    }

    #[tokio::test]
    async fn cancellation_stays_responsive_while_native_runtime_is_locked() {
        let backend = super::super::tests::hooks_test_backend(None);
        let _active = backend
            .runtime_lane_admission
            .try_acquire()
            .unwrap()
            .unwrap();
        let held = hold_runtime(backend.runtime.clone());
        let cancellation = CancellationToken::new();
        let mut waiting = Box::pin(backend.acquire_runtime_lane(Instant::now(), &cancellation));
        assert!(futures_util::poll!(&mut waiting).is_pending());
        assert!(!held.released.load(Ordering::Acquire));
        cancellation.cancel();
        assert!(waiting.await.is_err());
        assert!(!held.released.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn expired_deadline_is_checked_while_native_runtime_is_locked() {
        let backend = super::super::tests::hooks_test_backend(None);
        let held = hold_runtime(backend.runtime.clone());
        let cancellation = CancellationToken::new();
        let started = Instant::now()
            .checked_sub(backend.generation_admission_timeout + Duration::from_secs(1))
            .unwrap();
        let mut waiting = Box::pin(backend.acquire_runtime_lane(started, &cancellation));
        assert!(matches!(
            futures_util::poll!(&mut waiting),
            std::task::Poll::Ready(Err(_))
        ));
        assert!(!held.released.load(Ordering::Acquire));
    }
}
