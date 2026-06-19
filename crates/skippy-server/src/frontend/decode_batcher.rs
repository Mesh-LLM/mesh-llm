use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
        mpsc as std_mpsc,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use super::*;
use crate::runtime_state::RuntimeDecodeBatchRequest;

pub(super) struct DecodeBatcher {
    shared: Arc<DecodeBatcherShared>,
}

struct DecodeBatcherShared {
    runtime: Arc<Mutex<RuntimeState>>,
    state: Mutex<DecodeBatcherState>,
    ready: Condvar,
    max_batch_size: usize,
    owner_count: AtomicUsize,
    worker: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Default)]
struct DecodeBatcherState {
    pending: VecDeque<PendingDecode>,
    stopping: bool,
}

struct PendingDecode {
    session_id: String,
    token_id: i32,
    sampling: Option<SamplingConfig>,
    enqueued_at: Instant,
    reply: std_mpsc::SyncSender<OpenAiResult<DecodeBatchOutcome>>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DecodeBatchOutcome {
    pub predicted: i32,
    pub batch_size: usize,
    pub batch_wait_ms: f64,
    pub runtime_lock_wait_ms: f64,
    pub runtime_lock_hold_ms: f64,
}

impl DecodeBatcher {
    pub(super) fn new(runtime: Arc<Mutex<RuntimeState>>, max_batch_size: usize) -> Self {
        let shared = Arc::new(DecodeBatcherShared {
            runtime,
            state: Mutex::new(DecodeBatcherState::default()),
            ready: Condvar::new(),
            max_batch_size: max_batch_size.max(1),
            owner_count: AtomicUsize::new(1),
            worker: Mutex::new(None),
        });
        let worker = Arc::downgrade(&shared);
        let worker = thread::spawn(move || DecodeBatcherShared::run_worker(worker));
        if let Ok(mut slot) = shared.worker.lock() {
            *slot = Some(worker);
        }
        Self { shared }
    }

    pub(super) fn decode(
        &self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
    ) -> OpenAiResult<DecodeBatchOutcome> {
        let (reply, receiver) = std_mpsc::sync_channel(1);
        self.shared.enqueue(PendingDecode {
            session_id: session_id.to_string(),
            token_id,
            sampling: sampling.cloned(),
            enqueued_at: Instant::now(),
            reply,
        })?;
        receiver
            .recv()
            .map_err(|error| OpenAiError::backend(format!("decode batcher stopped: {error}")))?
    }
}

impl Clone for DecodeBatcher {
    fn clone(&self) -> Self {
        self.shared.owner_count.fetch_add(1, Ordering::Relaxed);
        Self {
            shared: self.shared.clone(),
        }
    }
}

impl Drop for DecodeBatcher {
    fn drop(&mut self) {
        if self.shared.owner_count.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        if let Ok(mut state) = self.shared.state.lock() {
            state.stopping = true;
            self.shared.ready.notify_all();
        }
        let worker = match self.shared.worker.lock() {
            Ok(mut worker) => worker.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(worker) = worker {
            let _ = worker.join();
        }
    }
}

impl DecodeBatcherShared {
    fn enqueue(&self, pending: PendingDecode) -> OpenAiResult<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| OpenAiError::backend("decode batcher lock poisoned"))?;
        state.pending.push_back(pending);
        self.ready.notify_one();
        Ok(())
    }

    fn run_worker(shared: Weak<Self>) {
        while let Some(shared) = shared.upgrade() {
            let Some(batch) = shared.wait_for_batch() else {
                break;
            };
            shared.run_batch(batch);
        }
    }

    fn wait_for_batch(&self) -> Option<Vec<PendingDecode>> {
        let mut state = self.state.lock().expect("decode batcher lock poisoned");
        while state.pending.is_empty() && !state.stopping {
            state = self
                .ready
                .wait(state)
                .expect("decode batcher lock poisoned");
        }
        if state.pending.is_empty() && state.stopping {
            return None;
        }
        let batch_size = self.max_batch_size.min(state.pending.len());
        Some(state.pending.drain(..batch_size).collect())
    }

    fn run_batch(&self, batch: Vec<PendingDecode>) {
        let batch_size = batch.len();
        let batch_wait_ms = batch
            .iter()
            .map(|pending| pending.enqueued_at.elapsed().as_secs_f64() * 1000.0)
            .fold(0.0, f64::max);
        let lock_timer = PhaseTimer::start();
        let runtime_result = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned while running decode batch"));
        let runtime_lock_wait_ms = lock_timer.elapsed_ms();
        let result = runtime_result.and_then(|mut runtime| {
            let hold_timer = PhaseTimer::start();
            let requests = batch
                .iter()
                .map(|pending| RuntimeDecodeBatchRequest {
                    session_id: pending.session_id.as_str(),
                    token_id: pending.token_id,
                    sampling: pending.sampling.as_ref(),
                })
                .collect::<Vec<_>>();
            let predicted = runtime
                .decode_batch_sampled(&requests)
                .map_err(openai_backend_error)?;
            Ok((predicted, hold_timer.elapsed_ms()))
        });
        Self::send_batch_replies(
            batch,
            batch_size,
            batch_wait_ms,
            runtime_lock_wait_ms,
            result,
        );
    }

    fn send_batch_replies(
        batch: Vec<PendingDecode>,
        batch_size: usize,
        batch_wait_ms: f64,
        runtime_lock_wait_ms: f64,
        result: OpenAiResult<(Vec<i32>, f64)>,
    ) {
        match result {
            Ok((predicted, runtime_lock_hold_ms)) => {
                for (pending, predicted) in batch.into_iter().zip(predicted) {
                    let _ = pending.reply.send(Ok(DecodeBatchOutcome {
                        predicted,
                        batch_size,
                        batch_wait_ms,
                        runtime_lock_wait_ms,
                        runtime_lock_hold_ms,
                    }));
                }
            }
            Err(error) => {
                for pending in batch {
                    let _ = pending.reply.send(Err(error.clone()));
                }
            }
        }
    }
}
