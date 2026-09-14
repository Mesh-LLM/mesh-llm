//! The two properties the native reporter callback has to have.
//!
//! It fires on whichever native worker thread raised the event -- a decode
//! thread, a loader thread, a device monitor. So:
//!
//! 1. **It never waits on a consumer.** Not when the ring is empty, not
//!    when it is full, not when another native thread is pushing at the
//!    same time. A decode thread stalling to report a diagnostic is the
//!    failure this boundary exists to prevent.
//! 2. **It never allocates.** An allocation can take a global lock inside
//!    the allocator, which is a stall by another name, and it is the exact
//!    cost the old path paid -- it copied every detail byte into a `Vec` on
//!    the callback thread before doing anything else.
//!
//! A separate integration binary because it installs a counting global
//! allocator, which would otherwise instrument every other test in the
//! crate.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use skippy_ffi::{
    SkippyRuntimeEventCategory, SkippyRuntimeEventEmitterKind, SkippyRuntimeEventFailureCode,
    SkippyRuntimeEventKind, SkippyRuntimeEventProgressUnit, SkippyRuntimeEventV1, Status,
};

thread_local! {
    /// Total allocation CALLS on this thread, never decremented. A
    /// net-zero counter cannot distinguish "no allocation" from "an equal
    /// number of same-window allocate/free pairs"; this one can.
    static TOTAL_ALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TOTAL_ALLOC_CALLS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        TOTAL_ALLOC_CALLS.with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn total_alloc_calls() -> u64 {
    TOTAL_ALLOC_CALLS.with(Cell::get)
}

fn raw_event(detail: &[u8]) -> SkippyRuntimeEventV1 {
    SkippyRuntimeEventV1 {
        abi_version: 1,
        struct_size: std::mem::size_of::<SkippyRuntimeEventV1>() as u32,
        category: SkippyRuntimeEventCategory::DEVICE,
        kind: SkippyRuntimeEventKind::DEVICE_READY,
        emitter: SkippyRuntimeEventEmitterKind::WORKER_THREAD,
        reserved0: 0,
        sequence: 7,
        timestamp_mono_ns: 11,
        model_id: 13,
        stage_id: 17,
        session_id: 19,
        progress_current: 1,
        progress_total: 2,
        progress_unit: SkippyRuntimeEventProgressUnit::ITEMS,
        failure_code: SkippyRuntimeEventFailureCode::NONE,
        status: Status::Ok,
        reserved1: 0,
        detail_ptr: if detail.is_empty() {
            std::ptr::null()
        } else {
            detail.as_ptr().cast()
        },
        detail_len: detail.len() as u64,
        numeric_summary_0: 1,
        numeric_summary_1: 2,
        numeric_summary_2: 3,
        numeric_summary_3: 4,
    }
}

/// The record ring is process-global, so these tests cannot run
/// concurrently with each other: one test's drain would consume another's
/// evidence. Every test takes this first.
static RING: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ring_guard() -> std::sync::MutexGuard<'static, ()> {
    RING.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn drain_all() -> usize {
    let mut out = Vec::new();
    skippy_runtime::drain_runtime_events(&mut out, usize::MAX);
    out.len()
}

/// The callback allocates nothing, including with detail long enough that
/// the previous implementation would have heap-allocated for it.
#[test]
fn the_callback_allocates_nothing_even_with_long_detail() {
    let _ring = ring_guard();
    let _ = drain_all();
    // Detail far past the inline budget: the old path copied all of it
    // into a `Vec` on this thread.
    let detail = vec![b'd'; 64 * 1024];
    let event = raw_event(&detail);

    // Warm this thread's own thread-local before measuring.
    unsafe { skippy_runtime::deliver_runtime_event_for_test(&event) };
    let _ = drain_all();

    let before = total_alloc_calls();
    for _ in 0..128 {
        unsafe { skippy_runtime::deliver_runtime_event_for_test(&event) };
    }
    let after = total_alloc_calls();

    assert_eq!(
        after,
        before,
        "the native callback performed {} allocation calls across 128 events; \
         it must copy into a fixed-size record and nothing more",
        after - before
    );
    let _ = drain_all();
}

/// The callback returns immediately with no consumer at all, including
/// long after the ring has filled.
///
/// A ring with nobody draining it is the worst case for this boundary: if
/// backpressure could ever reach the caller, this is where it would.
#[test]
fn the_callback_never_waits_for_a_consumer() {
    const EVENTS: usize = skippy_runtime::RECORD_RING_CAPACITY * 4;
    /// Generous enough not to flake on a loaded machine, tight enough that
    /// any real waiting would blow straight through it.
    const BUDGET: Duration = Duration::from_millis(200);

    let _ring = ring_guard();
    let _ = drain_all();
    let detail = vec![b'd'; 4096];
    let event = raw_event(&detail);

    // No consumer runs for the whole of this loop, so the ring fills after
    // RECORD_RING_CAPACITY events and every later push is a drop.
    let started = Instant::now();
    for _ in 0..EVENTS {
        unsafe { skippy_runtime::deliver_runtime_event_for_test(&event) };
    }
    let elapsed = started.elapsed();

    assert!(
        elapsed < BUDGET,
        "{EVENTS} callbacks with no consumer took {elapsed:?}; the callback \
         must never wait, full ring or not"
    );
    assert_eq!(
        skippy_runtime::buffered_runtime_events(),
        skippy_runtime::RECORD_RING_CAPACITY
    );
    let _ = drain_all();
}

/// Eight native threads calling the callback concurrently never wait on
/// each other. Under the previous implementation they serialized on the
/// sink mutex, and each of them held it across an identity-registry lock,
/// several allocations, a reservation, and a submit.
#[test]
fn concurrent_native_threads_do_not_serialize_on_each_other() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 2_000;
    const BUDGET: Duration = Duration::from_millis(500);

    let _ring = ring_guard();
    let _ = drain_all();
    let slowest = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|threads| {
        for _ in 0..THREADS {
            let slowest = Arc::clone(&slowest);
            threads.spawn(move || {
                let detail = vec![b'd'; 1024];
                let event = raw_event(&detail);
                let started = Instant::now();
                for _ in 0..PER_THREAD {
                    unsafe { skippy_runtime::deliver_runtime_event_for_test(&event) };
                }
                slowest.fetch_max(
                    usize::try_from(started.elapsed().as_micros()).unwrap_or(usize::MAX),
                    Ordering::Relaxed,
                );
            });
        }
    });

    let slowest = Duration::from_micros(slowest.load(Ordering::Relaxed) as u64);
    assert!(
        slowest < BUDGET,
        "the slowest of {THREADS} native threads took {slowest:?} for \
         {PER_THREAD} callbacks; concurrent callbacks must not serialize"
    );
    let _ = drain_all();
}
