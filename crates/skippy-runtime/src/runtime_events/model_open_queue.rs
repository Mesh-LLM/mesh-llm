//! Per-call ingress for native model-open events.
//!
//! A `_with_events` model open hands the native runtime a reporter whose
//! callback fires on a native loader or worker thread. That callback does
//! the same three things the process-global reporter does and nothing else:
//! validate the fixed-size header, copy the event into a [`NativeEventRecord`]
//! on the stack, and push it into a bounded lock-free queue. No mutex, no
//! allocation, no formatting, no caller-supplied closure.
//!
//! The caller owns the queue through an `Arc`, drains it on its own thread,
//! and expands records into owned `RuntimeEvent`s there. A full queue drops
//! and counts; a malformed event is refused and counted. Neither ever makes
//! the native thread wait.

use std::ffi::c_void;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_queue::ArrayQueue;
use skippy_ffi::{
    SkippyRuntimeEventReporterV1 as RawRuntimeEventReporter,
    SkippyRuntimeEventV1 as RawRuntimeEvent,
};

use super::{NativeEventRecord, OperationId, RUNTIME_EVENT_V1_ABI_VERSION};

/// Records one model-open queue holds before it starts dropping.
///
/// A model open emits a handful of lifecycle facts plus bounded progress;
/// this covers a full open without a consumer draining mid-call.
pub const MODEL_OPEN_RECORD_CAPACITY: usize = 256;

/// Bounded, lock-free record queue for one model-open operation.
pub struct ModelOpenEventQueue {
    operation_id: OperationId,
    records: ArrayQueue<NativeEventRecord>,
    dropped: AtomicU64,
    rejected: AtomicU64,
}

impl ModelOpenEventQueue {
    /// A queue with [`MODEL_OPEN_RECORD_CAPACITY`] slots.
    #[must_use]
    pub fn new(operation_id: OperationId) -> Arc<Self> {
        Self::with_capacity(operation_id, MODEL_OPEN_RECORD_CAPACITY)
    }

    /// A queue with `capacity` slots, clamped to at least one.
    #[must_use]
    pub fn with_capacity(operation_id: OperationId, capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            operation_id,
            records: ArrayQueue::new(capacity.max(1)),
            dropped: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        })
    }

    /// The operation every record in this queue belongs to.
    #[must_use]
    pub fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    /// Move up to `max` records, oldest first, into `out`. Returns how many
    /// were moved. Consumer-side only.
    pub fn drain(&self, out: &mut Vec<NativeEventRecord>, max: usize) -> usize {
        let mut taken = 0;
        while taken < max {
            let Some(record) = self.records.pop() else {
                break;
            };
            out.push(record);
            taken += 1;
        }
        taken
    }

    /// Records dropped because the queue was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Events refused at the boundary (null, short, wrong ABI, oversized).
    #[must_use]
    pub fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Test seam: deliver `event` through the same code path the native
    /// trampoline runs, so allocation and blocking can be measured from an
    /// integration test.
    ///
    /// # Safety
    ///
    /// Same contract as the native reporter callback: `event` must be null
    /// or point to a valid `SkippyRuntimeEventV1` for the duration of the
    /// call.
    #[doc(hidden)]
    pub unsafe fn deliver_for_test(&self, event: *const RawRuntimeEvent) {
        unsafe { model_open_event_trampoline(event, self as *const Self as *mut c_void) };
    }

    fn ingest(&self, event: *const RawRuntimeEvent) {
        // SAFETY: forwarded unchanged from the native reporter callback,
        // whose ABI contract is `NativeEventRecord::from_raw_ptr`'s.
        match unsafe { NativeEventRecord::from_raw_ptr(event) } {
            Ok(record) => {
                if self.records.push(record).is_err() {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(_) => {
                self.rejected.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Validate, copy, push. `catch_unwind` stays because unwinding across the
/// FFI boundary is undefined behavior.
pub(super) unsafe extern "C" fn model_open_event_trampoline(
    event: *const RawRuntimeEvent,
    user_data: *mut c_void,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() {
            return;
        }
        // SAFETY: `user_data` is `Arc::as_ptr` of a queue kept alive by the
        // `ModelOpenEventReporterRegistration` for the whole native call;
        // only a shared reference is formed, and every field it touches is
        // lock-free and `Sync`.
        let queue = unsafe { &*(user_data as *const ModelOpenEventQueue) };
        queue.ingest(event);
    }));
}

/// Owns the queue handle and the raw reporter that points at it for the
/// duration of one `_with_events` native call.
pub(super) struct ModelOpenEventReporterRegistration {
    _queue: Arc<ModelOpenEventQueue>,
    reporter: RawRuntimeEventReporter,
}

impl ModelOpenEventReporterRegistration {
    pub(super) fn new(queue: &Arc<ModelOpenEventQueue>) -> Self {
        let queue = Arc::clone(queue);
        let reporter = RawRuntimeEventReporter {
            abi_version: RUNTIME_EVENT_V1_ABI_VERSION,
            struct_size: mem::size_of::<RawRuntimeEventReporter>() as u32,
            callback: Some(model_open_event_trampoline),
            user_data: Arc::as_ptr(&queue) as *mut c_void,
        };
        Self {
            _queue: queue,
            reporter,
        }
    }

    pub(super) fn reporter_ptr(&self) -> *const RawRuntimeEventReporter {
        &self.reporter
    }
}

#[cfg(test)]
mod tests;
