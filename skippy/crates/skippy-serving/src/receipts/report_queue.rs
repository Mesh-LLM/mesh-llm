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

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};

use skippy_receipts::recorder::{MemoryOp, Recorder};
use skippy_runtime::{ReceiptFlusher, ReceiptMemoryEvent, RecordedUbatch};

use super::native::apply_ubatch;

pub(super) enum Report {
    Ubatch(RecordedUbatch),
    Unreceiptable(String),
    Memory(ReceiptMemoryEvent),
    /// The recurrent state the next turn of `seq` starts from.
    State {
        seq: i32,
        state: Option<Vec<u8>>,
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
    native: OnceLock<ReceiptFlusher>,
}

impl ReportQueue {
    pub(super) fn new(recorder: Arc<Recorder>) -> Self {
        Self {
            recorder,
            progress: Arc::default(),
            sender: Mutex::new(None),
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
        if let Some(native) = self.native.get()
            && let Err(error) = native.flush()
        {
            tracing::warn!("receipts: cannot flush the recorder: {error:#}");
        }
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
        if let Err(error) = spawned {
            // dropping the receiver makes every send fail, so push applies inline
            tracing::warn!("receipts: cannot start the recorder thread: {error}");
        }
        sender
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
            tracing::warn!("receipts: a ubatch cannot be receipted: {reason}");
            recorder.unreceiptable(&reason);
        }
        Report::Memory(event) => match MemoryOp::from_code(event.op) {
            Some(op) => recorder.memory_op(op, event.seq, event.seq_dst, event.p0, event.p1),
            None => recorder.unreceiptable(&format!("unknown KV-cache operation {}", event.op)),
        },
        Report::State { seq, state } => recorder.record_state(seq, state.as_deref()),
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
}
