//! The process-global runtime-scoped event reporter.
//!
//! The native reporter callback fires on whichever worker thread raised
//! the event -- a decode thread, a loader thread, a device monitor. That
//! thread is doing real work, and every instruction it spends here is
//! stolen from it.
//!
//! So the callback does one thing: copy the event into a
//! [`NativeEventRecord`] on the stack and push it into a bounded lock-free
//! ring. No mutex, no allocation, no notify, no conversion. A consumer
//! calls [`drain_runtime_events`] on its own schedule and pays for all of
//! that there.
//!
//! It used to call an installed `Box<dyn FnMut>` sink inline, under a
//! mutex, and that sink went on to take an identity-registry lock,
//! allocate several owned strings, reserve a slot, and submit -- all on the
//! native worker's thread, with the sink mutex held for the whole of it.
//! Two callbacks from different native threads serialized on that mutex
//! against each other and against every install or clear.
//!
//! ## Loss is bounded and counted
//!
//! The ring holds [`RECORD_RING_CAPACITY`] records. A push into a full
//! ring drops the record and increments [`dropped_runtime_events`], which
//! is the honest behavior for this boundary: the alternative is making a
//! decode thread wait for a consumer, which is the thing this file exists
//! to prevent. Native events are diagnostic observations, and the runtime
//! event contract already requires consumers to tolerate missing progress
//! and diagnostic events.

use std::ffi::c_void;
use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};

use crossbeam_queue::ArrayQueue;
use skippy_ffi::{
    FEATURE_RUNTIME_EVENT_REPORTER, SkippyRuntimeEventReporterV1 as RawReporter,
    SkippyRuntimeEventV1 as RawRuntimeEvent,
};

use crate::capability_probe::probe_capabilities;
use crate::runtime_events::{NativeEventRecord, RUNTIME_EVENT_V1_ABI_VERSION};

/// Records the ring holds before it starts dropping.
///
/// Sized for a burst, not a backlog: the consumer drains on the engine
/// driver's own tick, so this only has to cover what every native thread
/// can produce between two ticks.
pub const RECORD_RING_CAPACITY: usize = 1024;

static RECORDS: OnceLock<ArrayQueue<NativeEventRecord>> = OnceLock::new();
static DROPPED: AtomicU64 = AtomicU64::new(0);
static REPORTER_LIFECYCLE: OnceLock<Mutex<()>> = OnceLock::new();

fn records() -> &'static ArrayQueue<NativeEventRecord> {
    RECORDS.get_or_init(|| ArrayQueue::new(RECORD_RING_CAPACITY))
}

fn lifecycle_slot() -> &'static Mutex<()> {
    REPORTER_LIFECYCLE.get_or_init(|| Mutex::new(()))
}

/// Copy-and-push only. Nothing here locks, allocates, blocks, or converts.
///
/// The `catch_unwind` stays: unwinding across the FFI boundary is undefined
/// behavior, and while nothing inside can panic today, "nothing can panic"
/// is not a property the compiler checks.
unsafe extern "C" fn runtime_reporter_trampoline(
    event: *const RawRuntimeEvent,
    _user_data: *mut c_void,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the reporter ABI contract guarantees `event` is null or
        // points at a `SkippyRuntimeEventV1` valid for this call.
        let Some(record) = (unsafe { NativeEventRecord::from_raw_ptr(event) }) else {
            return;
        };
        if records().push(record).is_err() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }));
}

/// Take up to `max` buffered records, oldest first, appending to `out`.
/// Returns how many were taken.
///
/// Called by the consumer, never by a native thread. Expanding a record
/// into an owned `RuntimeEvent` -- which allocates -- is
/// [`NativeEventRecord::to_event`]'s job and equally the consumer's.
pub fn drain_runtime_events(out: &mut Vec<NativeEventRecord>, max: usize) -> usize {
    let mut taken = 0;
    while taken < max {
        let Some(record) = records().pop() else {
            break;
        };
        out.push(record);
        taken += 1;
    }
    taken
}

/// Records dropped because the ring was full when a native thread pushed.
/// Monotonic for the life of the process.
#[must_use]
pub fn dropped_runtime_events() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// Records currently buffered for the consumer.
#[must_use]
pub fn buffered_runtime_events() -> usize {
    records().len()
}

/// Test seam: deliver `event` through the real trampoline.
///
/// Exists so the two properties that actually matter about this boundary
/// can be asserted from an integration test with its own global allocator:
/// that a callback never waits on a consumer, and that it never allocates.
/// Neither can be observed through the public API, because the trampoline
/// is only ever called by native code.
///
/// # Safety
///
/// Same contract as the native reporter callback: `event` must be null or
/// point to a valid `SkippyRuntimeEventV1` for the duration of the call.
#[doc(hidden)]
pub unsafe fn deliver_runtime_event_for_test(event: *const RawRuntimeEvent) {
    unsafe { runtime_reporter_trampoline(event, std::ptr::null_mut()) };
}

type SetReporterFn = unsafe extern "C" fn(*const RawReporter) -> skippy_ffi::Status;
type ClearReporterFn = unsafe extern "C" fn();

fn set_reporter_fn() -> Option<SetReporterFn> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        skippy_ffi::skippy_set_runtime_event_reporter_fn()
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        static CACHE: OnceLock<Option<SetReporterFn>> = OnceLock::new();
        *CACHE.get_or_init(|| {
            #[cfg(unix)]
            {
                let symbol = unsafe {
                    libc::dlsym(
                        libc::RTLD_DEFAULT,
                        c"skippy_set_runtime_event_reporter".as_ptr(),
                    )
                };
                (!symbol.is_null())
                    .then(|| unsafe { std::mem::transmute::<*mut c_void, SetReporterFn>(symbol) })
            }
            #[cfg(not(unix))]
            {
                None
            }
        })
    }
}

fn clear_reporter_fn() -> Option<ClearReporterFn> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        skippy_ffi::skippy_clear_runtime_event_reporter_fn()
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        static CACHE: OnceLock<Option<ClearReporterFn>> = OnceLock::new();
        *CACHE.get_or_init(|| {
            #[cfg(unix)]
            {
                let symbol = unsafe {
                    libc::dlsym(
                        libc::RTLD_DEFAULT,
                        c"skippy_clear_runtime_event_reporter".as_ptr(),
                    )
                };
                (!symbol.is_null())
                    .then(|| unsafe { std::mem::transmute::<*mut c_void, ClearReporterFn>(symbol) })
            }
            #[cfg(not(unix))]
            {
                None
            }
        })
    }
}

/// Installs the runtime-scoped (process-global) event reporter, gated on the
/// probed `runtime_event_reporter` family. Returns `false` without touching
/// native state when the family is unavailable or a symbol failed to
/// resolve, so a caller can fall back cleanly on an older runtime.
pub fn install_runtime_event_reporter() -> bool {
    if !probe_capabilities().family_confirmed(FEATURE_RUNTIME_EVENT_REPORTER) {
        return false;
    }
    let Some(set_fn) = set_reporter_fn() else {
        return false;
    };

    install_runtime_event_reporter_with_setter(set_fn)
}

fn install_runtime_event_reporter_with_setter(set_fn: SetReporterFn) -> bool {
    // Serialize install against clear. The native setter may wait for
    // in-flight callbacks to return, which is precisely why the trampoline
    // must not take this lock: a callback blocking here while the setter
    // waits for that callback would deadlock.
    let _lifecycle = lifecycle_slot()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    // Make sure the ring exists before any callback can fire, so the
    // trampoline's `get_or_init` is never the first one.
    let _ = records();

    let reporter = RawReporter {
        abi_version: RUNTIME_EVENT_V1_ABI_VERSION,
        struct_size: mem::size_of::<RawReporter>() as u32,
        callback: Some(runtime_reporter_trampoline),
        user_data: std::ptr::null_mut(),
    };
    (unsafe { set_fn(&reporter) }) == skippy_ffi::Status::Ok
}

/// Clears the runtime-scoped event reporter. Blocks (via the native
/// `skippy_clear_runtime_event_reporter` quiescence contract) until every
/// in-flight callback has returned, so no callback is still running when
/// this returns. A no-op when nothing was installed.
///
/// Records already in the ring are deliberately left there: they describe
/// things that genuinely happened, and a consumer draining after teardown
/// should still see them.
pub fn clear_runtime_event_reporter() {
    // Dynamic symbol lookup requires a loaded runtime; the guard keeps this
    // public cleanup function a safe no-op before dynamic startup.
    let clear_fn = if skippy_ffi::native_runtime_loaded() {
        clear_reporter_fn()
    } else {
        None
    };
    clear_runtime_event_reporter_with_clearer(clear_fn);
}

fn clear_runtime_event_reporter_with_clearer(clear_fn: Option<ClearReporterFn>) {
    let _lifecycle = lifecycle_slot()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(clear_fn) = clear_fn {
        unsafe { clear_fn() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_events::INLINE_DETAIL_BYTES;

    use skippy_ffi::{
        SkippyRuntimeEventCategory as RawRuntimeEventCategory,
        SkippyRuntimeEventEmitterKind as RawRuntimeEventEmitterKind,
        SkippyRuntimeEventFailureCode as RawRuntimeEventFailureCode,
        SkippyRuntimeEventKind as RawRuntimeEventKind,
        SkippyRuntimeEventProgressUnit as RawRuntimeEventProgressUnit,
        SkippyRuntimeEventV1 as RawRuntimeEvent,
    };

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn test_guard() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn raw_event() -> RawRuntimeEvent {
        RawRuntimeEvent {
            abi_version: 1,
            struct_size: mem::size_of::<RawRuntimeEvent>() as u32,
            category: RawRuntimeEventCategory::MODEL_OPEN,
            kind: RawRuntimeEventKind::MODEL_OPEN_STARTED,
            emitter: RawRuntimeEventEmitterKind::WORKER_THREAD,
            reserved0: 0,
            sequence: 1,
            timestamp_mono_ns: 1,
            model_id: 1,
            stage_id: 0,
            session_id: 0,
            progress_current: 0,
            progress_total: 0,
            progress_unit: RawRuntimeEventProgressUnit::NONE,
            failure_code: RawRuntimeEventFailureCode::NONE,
            status: skippy_ffi::Status::Ok,
            reserved1: 0,
            detail_ptr: std::ptr::null(),
            detail_len: 0,
            numeric_summary_0: 0,
            numeric_summary_1: 0,
            numeric_summary_2: 0,
            numeric_summary_3: 0,
        }
    }

    unsafe extern "C" fn successful_setter(_reporter: *const RawReporter) -> skippy_ffi::Status {
        skippy_ffi::Status::Ok
    }

    unsafe extern "C" fn failing_setter(_reporter: *const RawReporter) -> skippy_ffi::Status {
        skippy_ffi::Status::Error
    }

    unsafe extern "C" fn observing_clearer() {
        let event = raw_event();
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
    }

    /// Empty the process-global ring so a test measures only its own
    /// pushes. The ring outlives any single test, unlike the per-test
    /// sink it replaced.
    fn drain_all() -> Vec<NativeEventRecord> {
        let mut out = Vec::new();
        drain_runtime_events(&mut out, usize::MAX);
        out
    }

    #[test]
    fn install_returns_false_without_a_confirmed_family() {
        let _test_guard = test_guard();
        // No native runtime is loaded in unit tests, so the family probe
        // always reports unconfirmed; install must refuse cleanly rather
        // than dereference an absent native symbol.
        assert!(!install_runtime_event_reporter());
    }

    #[test]
    fn clear_is_a_safe_no_op_when_nothing_was_installed() {
        let _test_guard = test_guard();
        clear_runtime_event_reporter();
    }

    /// The callback's whole job: the raw event arrives in the ring as a
    /// record, field for field, with no conversion done on the calling
    /// thread.
    #[test]
    fn the_trampoline_copies_a_raw_event_into_the_ring() {
        let _test_guard = test_guard();
        let _ = drain_all();

        let event = raw_event();
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };

        let records = drain_all();
        assert_eq!(records.len(), 1);
        let record = records[0];
        assert_eq!(record.sequence, event.sequence);
        assert_eq!(record.model_id, event.model_id);
        assert_eq!(record.session_id, event.session_id);
        assert_eq!(record.kind, event.kind.0);
        assert_eq!(record.category, event.category.0);
        assert_eq!(record.status, event.status);
        assert!(!record.detail_truncated);
    }

    /// A null pointer, and a `struct_size` that does not cover the base
    /// layout, are both refused without reading any other field.
    #[test]
    fn a_malformed_event_is_refused_rather_than_queued() {
        let _test_guard = test_guard();
        let _ = drain_all();

        unsafe { runtime_reporter_trampoline(std::ptr::null(), std::ptr::null_mut()) };

        let mut undersized = raw_event();
        undersized.struct_size = 8;
        unsafe { runtime_reporter_trampoline(&undersized, std::ptr::null_mut()) };

        assert!(drain_all().is_empty());
    }

    /// Detail longer than the inline budget is truncated and SAID to be
    /// truncated. The previous path copied the whole allocation on the
    /// callback thread and let the consumer bound it silently, so a
    /// consumer could not tell a short detail from a clipped one.
    #[test]
    fn oversized_detail_is_truncated_and_flagged() {
        let _test_guard = test_guard();
        let _ = drain_all();

        let detail = vec![b'x'; INLINE_DETAIL_BYTES * 3];
        let mut event = raw_event();
        event.detail_ptr = detail.as_ptr().cast();
        event.detail_len = detail.len() as u64;
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };

        let records = drain_all();
        assert_eq!(records.len(), 1);
        assert!(records[0].detail_truncated);
        assert_eq!(records[0].detail().len(), INLINE_DETAIL_BYTES);
        assert!(records[0].detail().iter().all(|byte| *byte == b'x'));
    }

    /// Detail that fits is carried whole, with the flag clear.
    #[test]
    fn detail_that_fits_is_carried_whole() {
        let _test_guard = test_guard();
        let _ = drain_all();

        let detail = b"native detail".to_vec();
        let mut event = raw_event();
        event.detail_ptr = detail.as_ptr().cast();
        event.detail_len = detail.len() as u64;
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };

        let records = drain_all();
        assert_eq!(records.len(), 1);
        assert!(!records[0].detail_truncated);
        assert_eq!(records[0].detail(), detail.as_slice());
    }

    /// A full ring drops and counts rather than making the calling thread
    /// wait, and draining restores capacity.
    ///
    /// Waiting is the one thing this boundary must never do: the caller is
    /// a native worker thread in the middle of real work.
    #[test]
    fn a_full_ring_drops_and_counts_instead_of_blocking() {
        let _test_guard = test_guard();
        let _ = drain_all();
        let before = dropped_runtime_events();

        let event = raw_event();
        for _ in 0..RECORD_RING_CAPACITY {
            unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
        }
        assert_eq!(buffered_runtime_events(), RECORD_RING_CAPACITY);
        assert_eq!(
            dropped_runtime_events(),
            before,
            "the ring was not yet full"
        );

        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
        assert_eq!(
            dropped_runtime_events(),
            before + 1,
            "a push into a full ring must be counted, not silently lost"
        );

        assert_eq!(drain_all().len(), RECORD_RING_CAPACITY);
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
        assert_eq!(
            buffered_runtime_events(),
            1,
            "draining must restore capacity rather than leaving the ring wedged"
        );
        let _ = drain_all();
    }

    /// `drain_runtime_events` honors its bound and preserves push order.
    #[test]
    fn draining_is_bounded_and_ordered() {
        let _test_guard = test_guard();
        let _ = drain_all();

        for sequence in 0..5u64 {
            let mut event = raw_event();
            event.sequence = sequence;
            unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
        }

        let mut first = Vec::new();
        assert_eq!(drain_runtime_events(&mut first, 2), 2);
        assert_eq!(
            first
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        let rest = drain_all();
        assert_eq!(
            rest.iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }

    /// Install reports the native setter's own verdict, and does not
    /// swallow a rejection.
    #[test]
    fn install_reports_the_native_setters_verdict() {
        let _test_guard = test_guard();
        assert!(install_runtime_event_reporter_with_setter(
            successful_setter
        ));
        assert!(!install_runtime_event_reporter_with_setter(failing_setter));
    }

    /// A callback firing while the native clear call is still running is
    /// still served: the trampoline takes no lock the clear path holds, so
    /// it can neither deadlock nor lose the record.
    ///
    /// This used to be the delicate case -- the sink had to be kept alive
    /// across a native call that was itself waiting for callbacks to
    /// return. There is no sink to keep alive now.
    #[test]
    fn a_callback_during_native_clear_is_still_recorded() {
        let _test_guard = test_guard();
        let _ = drain_all();
        assert!(install_runtime_event_reporter_with_setter(
            successful_setter
        ));

        clear_runtime_event_reporter_with_clearer(Some(observing_clearer));

        assert_eq!(
            drain_all().len(),
            1,
            "the callback the clearer raised must have reached the ring"
        );
    }

    /// Expanding a record into the owned wire type round-trips every field
    /// the callback copied.
    #[test]
    fn a_record_expands_back_into_the_owned_event() {
        let _test_guard = test_guard();
        let _ = drain_all();

        let detail = b"round trip".to_vec();
        let mut raw = raw_event();
        raw.detail_ptr = detail.as_ptr().cast();
        raw.detail_len = detail.len() as u64;
        unsafe { runtime_reporter_trampoline(&raw, std::ptr::null_mut()) };

        let records = drain_all();
        let event = records[0].to_event();
        assert_eq!(event.sequence, raw.sequence);
        assert_eq!(event.model_id, raw.model_id);
        assert_eq!(event.session_id, raw.session_id);
        assert_eq!(event.detail_bytes, detail);
        assert_eq!(event.abi_version, raw.abi_version);
    }
}
