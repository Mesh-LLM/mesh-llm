use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
        mpsc as std_mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow};
use skippy_runtime::{ActivationFrame, SamplingConfig};

use crate::runtime_state::{
    RuntimeDecodeFrameBatchRequest, RuntimeSessionAlignStats, RuntimeState,
};

pub(crate) struct DecodeFrameBatcher {
    shared: Arc<DecodeFrameBatcherShared>,
}

struct DecodeFrameBatcherShared {
    runtime: Arc<Mutex<RuntimeState>>,
    state: Mutex<DecodeFrameBatcherState>,
    ready: Condvar,
    max_batch_size: usize,
    collection_window: Duration,
    owner_count: AtomicUsize,
    worker: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Default)]
struct DecodeFrameBatcherState {
    pending: VecDeque<PendingDecodeFrame>,
    stopping: bool,
}

struct PendingDecodeFrame {
    session_id: String,
    target_token_count: u64,
    token_id: i32,
    sampling: Option<SamplingConfig>,
    input: Option<ActivationFrame>,
    enqueued_at: Instant,
    reply: std_mpsc::SyncSender<Result<DecodeFrameBatchOutcome>>,
}

pub(crate) struct DecodeFrameBatchOutcome {
    pub(crate) predicted: i32,
    pub(crate) output: ActivationFrame,
    pub(crate) batch_size: usize,
    pub(crate) batch_wait_ms: f64,
    pub(crate) runtime_lock_wait_ms: f64,
    pub(crate) runtime_lock_hold_ms: f64,
    pub(crate) session_alignment: Option<RuntimeSessionAlignStats>,
}

impl DecodeFrameBatcher {
    pub(crate) fn new(runtime: Arc<Mutex<RuntimeState>>, max_batch_size: usize) -> Self {
        let shared = Arc::new(DecodeFrameBatcherShared {
            runtime,
            state: Mutex::new(DecodeFrameBatcherState::default()),
            ready: Condvar::new(),
            max_batch_size: max_batch_size.max(1),
            collection_window: crate::decode_batch_policy::collection_window(max_batch_size),
            owner_count: AtomicUsize::new(1),
            worker: Mutex::new(None),
        });
        let worker = Arc::downgrade(&shared);
        let worker = thread::spawn(move || DecodeFrameBatcherShared::run_worker(worker));
        if let Ok(mut slot) = shared.worker.lock() {
            *slot = Some(worker);
        }
        Self { shared }
    }

    pub(crate) fn decode(
        &self,
        session_id: &str,
        target_token_count: u64,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<ActivationFrame>,
    ) -> Result<DecodeFrameBatchOutcome> {
        let (reply, receiver) = std_mpsc::sync_channel(1);
        self.shared.enqueue(PendingDecodeFrame {
            session_id: session_id.to_string(),
            target_token_count,
            token_id,
            sampling: sampling.cloned(),
            input,
            enqueued_at: Instant::now(),
            reply,
        })?;
        receiver
            .recv()
            .map_err(|error| anyhow!("decode frame batcher stopped: {error}"))?
    }
}

impl Clone for DecodeFrameBatcher {
    fn clone(&self) -> Self {
        self.shared.owner_count.fetch_add(1, Ordering::Relaxed);
        Self {
            shared: self.shared.clone(),
        }
    }
}

impl Drop for DecodeFrameBatcher {
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

impl DecodeFrameBatcherShared {
    fn enqueue(&self, pending: PendingDecodeFrame) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("decode frame batcher lock poisoned"))?;
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

    fn wait_for_batch(&self) -> Option<Vec<PendingDecodeFrame>> {
        let mut state = self
            .state
            .lock()
            .expect("decode frame batcher lock poisoned");
        while state.pending.is_empty() && !state.stopping {
            state = self
                .ready
                .wait(state)
                .expect("decode frame batcher lock poisoned");
        }
        if state.pending.is_empty() && state.stopping {
            return None;
        }
        state = self.collect_until_deadline(state);
        let batch_size = self.max_batch_size.min(state.pending.len());
        Some(state.pending.drain(..batch_size).collect())
    }

    fn collect_until_deadline<'a>(
        &self,
        mut state: std::sync::MutexGuard<'a, DecodeFrameBatcherState>,
    ) -> std::sync::MutexGuard<'a, DecodeFrameBatcherState> {
        if self.collection_window.is_zero() || state.pending.len() >= self.max_batch_size {
            return state;
        }
        let deadline = Instant::now() + self.collection_window;
        while !state.stopping && state.pending.len() < self.max_batch_size {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            let (next, timeout) = self
                .ready
                .wait_timeout(state, remaining)
                .expect("decode frame batcher lock poisoned");
            state = next;
            if timeout.timed_out() {
                break;
            }
        }
        state
    }

    fn run_batch(&self, batch: Vec<PendingDecodeFrame>) {
        let batch_size = batch.len();
        let batch_wait_ms = batch
            .iter()
            .map(|pending| pending.enqueued_at.elapsed().as_secs_f64() * 1000.0)
            .fold(0.0, f64::max);
        let lock_started = Instant::now();
        let runtime_result = self
            .runtime
            .lock()
            .map_err(|_| anyhow!("runtime lock poisoned while running decode frame batch"));
        let runtime_lock_wait_ms = elapsed_ms(lock_started);
        let result = runtime_result.and_then(|mut runtime| {
            let hold_started = Instant::now();
            // Keep reconciliation and decode atomic. A caller-side alignment can race the
            // batch worker and otherwise decode a correction on the rejected suffix.
            let alignments = batch
                .iter()
                .map(|pending| {
                    runtime.align_session_to_token_count_if_ahead(
                        &pending.session_id,
                        pending.target_token_count,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let requests = batch
                .iter()
                .map(|pending| RuntimeDecodeFrameBatchRequest {
                    session_id: pending.session_id.as_str(),
                    token_id: pending.token_id,
                    sampling: pending.sampling.as_ref(),
                    input: pending.input.as_ref(),
                })
                .collect::<Vec<_>>();
            let outputs = runtime.decode_frame_batch_sampled(&requests)?;
            Ok((outputs, alignments, elapsed_ms(hold_started)))
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
        batch: Vec<PendingDecodeFrame>,
        batch_size: usize,
        batch_wait_ms: f64,
        runtime_lock_wait_ms: f64,
        result: Result<(
            Vec<skippy_runtime::DecodeFrameBatchOutput>,
            Vec<Option<RuntimeSessionAlignStats>>,
            f64,
        )>,
    ) {
        match result {
            Ok((outputs, alignments, runtime_lock_hold_ms)) => {
                for ((pending, output), session_alignment) in
                    batch.into_iter().zip(outputs).zip(alignments)
                {
                    let _ = pending.reply.send(Ok(DecodeFrameBatchOutcome {
                        predicted: output.predicted_token,
                        output: output.output,
                        batch_size,
                        batch_wait_ms,
                        runtime_lock_wait_ms,
                        runtime_lock_hold_ms,
                        session_alignment,
                    }));
                }
            }
            Err(error) => {
                let error = error.to_string();
                for pending in batch {
                    let _ = pending.reply.send(Err(anyhow!(error.clone())));
                }
            }
        }
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}
