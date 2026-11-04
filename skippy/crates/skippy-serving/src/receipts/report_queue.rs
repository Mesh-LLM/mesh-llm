//! Applies the native recorder's reports to the stage's [`Recorder`] on a
//! worker thread, in the order they were made.
//!
//! The decoding thread only queues a report, so hashing a ubatch's leaves
//! overlaps the GPU computing the next one. The runtime holds the next
//! ubatch until the worker drops the previous one, which bounds the queue.
//! Reads of the recorder (commit, open, recorded range) first flush the
//! native recorder, which on an asynchronous device holds the last ubatches
//! back, and then wait until every report queued before
//! them has been applied.
//!
//! The start of a session's turn is a report like the others, so the
//! recorder sees it between the ubatches computed before the session took
//! its sequence and the session's own.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread::JoinHandle;

use skippy_receipts::recorder::{MemoryOp, Recorder, TurnId};
use skippy_runtime::{ReceiptFlusher, ReceiptMemoryEvent, RecordedUbatch};

use super::native::apply_ubatch;

pub(super) enum Report {
    Ubatch(RecordedUbatch),
    Unreceiptable(String),
    /// A ubatch the native recorder could not receipt.
    UnrecordedUbatch(String),
    Memory(ReceiptMemoryEvent),
    /// The recurrent state the next turn of `seq` starts from.
    State {
        seq: i32,
        state: Option<Vec<u8>>,
    },
    /// A session starts turn `turn` on `seq`.
    Turn {
        seq: i32,
        turn: TurnId,
    },
}

/// Reports queued and applied so far.
#[derive(Default)]
struct Progress {
    counts: Mutex<(u64, u64)>,
    applied: Condvar,
}

impl Progress {
    fn lock(&self) -> std::sync::MutexGuard<'_, (u64, u64)> {
        self.counts.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

pub(super) struct ReportQueue {
    recorder: Arc<Recorder>,
    progress: Arc<Progress>,
    sender: Mutex<Option<Sender<Report>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    native: OnceLock<ReceiptFlusher>,
}

impl ReportQueue {
    pub(super) fn new(recorder: Arc<Recorder>) -> Self {
        Self {
            recorder,
            progress: Arc::default(),
            sender: Mutex::new(None),
            worker: Mutex::new(None),
            native: OnceLock::new(),
        }
    }

    /// The native recorder whose held-back ubatch a read reports first.
    pub(super) fn set_native(&self, flusher: ReceiptFlusher) {
        let _ = self.native.set(flusher);
    }

    pub(super) fn recorder(&self) -> &Recorder {
        &self.recorder
    }

    /// The recorder, for a test's replayer to hold as the native one holds
    /// its stage.
    #[cfg(test)]
    pub(super) fn shared_recorder(&self) -> Arc<Recorder> {
        self.recorder.clone()
    }

    /// Queues the start of a session's turn on `seq`, after every ubatch
    /// computed before it: the ubatches the native recorder still holds
    /// back are reported first. Call it with the stage idle, before the
    /// session's first ubatch. On a stage with hybrid memory, that ubatch
    /// then reports the recurrent state the turn starts from, also when the
    /// sequence was not removed whole since its last turn.
    pub(super) fn begin_turn(&self, seq: i32, turn: TurnId) {
        self.flush_native();
        self.push(Report::Turn { seq, turn });
        self.arm_state(seq);
    }

    fn flush_native(&self) {
        if let Some(native) = self.native.get()
            && let Err(error) = native.flush()
        {
            tracing::warn!("receipts: cannot flush the recorder: {error:#}");
        }
    }

    /// Starts a new turn of `seq` in the native recorder, which on a stage
    /// with hybrid memory reports the state the turn starts from.
    pub(super) fn arm_state(&self, seq: i32) {
        if let Some(native) = self.native.get()
            && let Err(error) = native.arm_state(seq)
        {
            tracing::warn!("receipts: cannot mark the start of a turn: {error:#}");
        }
    }

    /// Queues a report, starting the worker on first use.
    pub(super) fn push(&self, report: Report) {
        let mut sender = self.sender.lock().unwrap_or_else(PoisonError::into_inner);
        let sender = sender.get_or_insert_with(|| self.spawn_worker());
        self.progress.lock().0 += 1;
        if let Err(mpsc::SendError(report)) = sender.send(report) {
            // the worker is gone; apply here so the counts still meet
            apply(&self.recorder, report);
            mark_applied(&self.progress);
        }
    }

    /// Waits until every report made before the call has been applied.
    pub(super) fn wait_applied(&self) {
        self.flush_native();
        let counts = self.progress.lock();
        let target = counts.0;
        drop(
            self.progress
                .applied
                .wait_while(counts, |counts| counts.1 < target)
                .unwrap_or_else(PoisonError::into_inner),
        );
    }

    fn spawn_worker(&self) -> Sender<Report> {
        let (sender, receiver) = mpsc::channel::<Report>();
        let recorder = self.recorder.clone();
        let progress = self.progress.clone();
        let spawned = std::thread::Builder::new()
            .name("receipt-recorder".into())
            .spawn(move || {
                for report in receiver {
                    apply(&recorder, report);
                    mark_applied(&progress);
                }
            });
        match spawned {
            Ok(worker) => {
                *self.worker.lock().unwrap_or_else(PoisonError::into_inner) = Some(worker);
            }
            // dropping the receiver makes every send fail, so push applies inline
            Err(error) => tracing::warn!("receipts: cannot start the recorder thread: {error}"),
        }
        sender
    }
}

impl Drop for ReportQueue {
    /// Ends the worker, so the recorder it holds goes with the queue.
    fn drop(&mut self) {
        let sender = self
            .sender
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        drop(sender);
        let worker = self
            .worker
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(worker) = worker {
            let _ = worker.join();
        }
    }
}

fn mark_applied(progress: &Progress) {
    progress.lock().1 += 1;
    progress.applied.notify_all();
}

fn apply(recorder: &Recorder, report: Report) {
    let applied = catch_unwind(AssertUnwindSafe(|| match report {
        Report::Ubatch(ubatch) => apply_ubatch(recorder, ubatch),
        Report::Unreceiptable(reason) => {
            tracing::warn!("receipts: the recorder lost track of the stage: {reason}");
            recorder.unreceiptable(&reason);
        }
        Report::UnrecordedUbatch(reason) => {
            tracing::debug!("receipts: a ubatch cannot be receipted: {reason}");
            recorder.unrecorded_ubatch(&reason);
        }
        Report::Memory(event) => match MemoryOp::from_code(event.op) {
            Some(op) => recorder.memory_op(op, event.seq, event.seq_dst, event.p0, event.p1),
            None => recorder.unreceiptable(&format!("unknown KV-cache operation {}", event.op)),
        },
        Report::State { seq, state } => recorder.record_state(seq, state.as_deref()),
        Report::Turn { seq, turn } => recorder.begin_turn(seq, turn),
    }));
    if applied.is_err() {
        recorder.unreceiptable("recording a report panicked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waiting_sees_every_report_queued_before_it() {
        let queue = ReportQueue::new(Arc::new(Recorder::new(0, 0..2, "test")));
        for i in 0..100 {
            queue.push(Report::Unreceiptable(format!("report {i}")));
        }
        queue.wait_applied();
        assert_eq!(*queue.progress.lock(), (100, 100));
    }

    // The worker holds the recorder; it ends with the queue, so a stage
    // that is dropped leaves no recorder behind.
    #[test]
    fn the_recorder_goes_with_the_queue() {
        let recorder = Arc::new(Recorder::new(0, 0..2, "test"));
        let weak = Arc::downgrade(&recorder);
        let queue = ReportQueue::new(recorder);
        queue.begin_turn(1, TurnId(4));
        queue.wait_applied();
        drop(queue);
        assert!(weak.upgrade().is_none());
    }
}
