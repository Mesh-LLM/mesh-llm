//! Host-side consumer for the per-call native model-open event queue.
//!
//! The native callback only copies records into a `ModelOpenEventQueue`.
//! This module owns the other half: while a model open blocks, a host thread
//! drains the queue and hands each expanded event to the host's
//! `NativeModelOpenEventReporter`, so formatting, output events, and
//! runtime-event ingress all run off the native thread. After the open
//! returns, one final drain runs and the resulting [`ModelOpenObservation`]
//! is handed to the caller's reconciler together with the authoritative
//! native return. The reconciler reports losses and contradictions; the
//! return value itself passes through untouched.

mod observation;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use skippy_runtime::{ModelOpenEventQueue, NativeEventRecord, OperationId};

pub(crate) use observation::{ModelOpenObservation, ModelOpenReturn};

use super::NativeModelOpenEventReporter;

/// Upper bound on how long a drained record waits before reaching the
/// reporter while an open is in flight.
const DRAIN_INTERVAL: Duration = Duration::from_millis(20);

/// Settles what the host saw of a model open against its native return.
pub(crate) type ModelOpenReconciler =
    Box<dyn FnOnce(&ModelOpenObservation, ModelOpenReturn) + Send>;

/// Host consumers for one native model open: the per-event sink and the
/// post-return reconciler.
pub(crate) struct NativeModelOpenEvents {
    pub(crate) reporter: NativeModelOpenEventReporter,
    pub(crate) reconcile: ModelOpenReconciler,
}

/// Runs `load`, drains its model-open events into `events.reporter`, then
/// hands the observation and the return outcome to `events.reconcile`.
/// The load result is returned unchanged.
pub(crate) fn observe_model_open<T, E>(
    operation_id: OperationId,
    events: Option<NativeModelOpenEvents>,
    load: impl FnOnce(Option<Arc<ModelOpenEventQueue>>) -> Result<T, E>,
) -> Result<T, E> {
    let Some(NativeModelOpenEvents {
        reporter,
        reconcile,
    }) = events
    else {
        return load(None);
    };
    let (result, observation) =
        with_model_open_events(ModelOpenEventQueue::new(operation_id), reporter, load);
    reconcile(&observation, ModelOpenReturn::of(&result));
    result
}

/// Runs `load` against `queue` while a scoped host thread forwards every
/// drained event to `reporter`. Returns `load`'s result and what was
/// observed, including a final drain after `load` returned.
fn with_model_open_events<T>(
    queue: Arc<ModelOpenEventQueue>,
    reporter: NativeModelOpenEventReporter,
    load: impl FnOnce(Option<Arc<ModelOpenEventQueue>>) -> T,
) -> (T, ModelOpenObservation) {
    let finished = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let drainer = scope.spawn(|| drain_until_finished(&queue, &finished, reporter));
        // Signals the drainer even when `load` unwinds; otherwise the scope
        // would wait forever for a drainer that never sees `finished`.
        let stop = StopDrainer {
            finished: &finished,
            drainer: drainer.thread(),
        };
        let result = load(Some(Arc::clone(&queue)));
        drop(stop);
        let observation = drainer.join().unwrap_or_default();
        (result, observation.with_losses_from(&queue))
    })
}

struct StopDrainer<'a> {
    finished: &'a AtomicBool,
    drainer: &'a std::thread::Thread,
}

impl Drop for StopDrainer<'_> {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);
        self.drainer.unpark();
    }
}

fn drain_until_finished(
    queue: &ModelOpenEventQueue,
    finished: &AtomicBool,
    mut reporter: NativeModelOpenEventReporter,
) -> ModelOpenObservation {
    let mut observation = ModelOpenObservation::default();
    let mut records = Vec::new();
    loop {
        let done = finished.load(Ordering::Acquire);
        forward_drained(queue, &mut records, &mut reporter, &mut observation);
        if done {
            return observation;
        }
        std::thread::park_timeout(DRAIN_INTERVAL);
    }
}

fn forward_drained(
    queue: &ModelOpenEventQueue,
    records: &mut Vec<NativeEventRecord>,
    reporter: &mut NativeModelOpenEventReporter,
    observation: &mut ModelOpenObservation,
) {
    queue.drain(records, usize::MAX);
    for record in records.drain(..) {
        observation.record(&record);
        let event = record.to_event();
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reporter(event))).is_err() {
            tracing::warn!("native model-open event reporter panicked; event skipped");
        }
    }
}

#[cfg(test)]
mod tests;
