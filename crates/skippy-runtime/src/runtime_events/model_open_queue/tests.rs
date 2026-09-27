use std::sync::{Arc, Barrier};
use std::thread;

use skippy_ffi::{
    SkippyRuntimeEventCategory as RawRuntimeEventCategory,
    SkippyRuntimeEventEmitterKind as RawRuntimeEventEmitterKind,
    SkippyRuntimeEventFailureCode as RawRuntimeEventFailureCode,
    SkippyRuntimeEventKind as RawRuntimeEventKind,
    SkippyRuntimeEventProgressUnit as RawRuntimeEventProgressUnit,
    SkippyRuntimeEventV1 as RawRuntimeEvent, Status,
};

use super::{
    MODEL_OPEN_RECORD_CAPACITY, ModelOpenEventQueue, ModelOpenEventReporterRegistration,
    OperationId,
};
use crate::runtime_events::{NativeEventRecord, RUNTIME_EVENT_V1_ABI_VERSION, RuntimeEventKind};

fn raw_event(kind: RawRuntimeEventKind, sequence: u64) -> RawRuntimeEvent {
    RawRuntimeEvent {
        abi_version: 1,
        struct_size: std::mem::size_of::<RawRuntimeEvent>() as u32,
        category: RawRuntimeEventCategory::MODEL_OPEN,
        kind,
        emitter: RawRuntimeEventEmitterKind::WORKER_THREAD,
        reserved0: 0,
        sequence,
        timestamp_mono_ns: sequence,
        model_id: 1,
        stage_id: 0,
        session_id: 0,
        progress_current: 0,
        progress_total: 0,
        progress_unit: RawRuntimeEventProgressUnit::NONE,
        failure_code: RawRuntimeEventFailureCode::NONE,
        status: Status::Ok,
        reserved1: 0,
        detail_ptr: std::ptr::null(),
        detail_len: 0,
        numeric_summary_0: 0,
        numeric_summary_1: 0,
        numeric_summary_2: 0,
        numeric_summary_3: 0,
    }
}

fn progress(sequence: u64) -> RawRuntimeEvent {
    raw_event(RawRuntimeEventKind::MODEL_OPEN_PROGRESS, sequence)
}

fn drain_all(queue: &ModelOpenEventQueue) -> Vec<NativeEventRecord> {
    let mut out = Vec::new();
    queue.drain(&mut out, usize::MAX);
    out
}

#[test]
fn push_drain_preserves_sequence_order() {
    let queue = ModelOpenEventQueue::new(OperationId(1));
    for sequence in 0..5 {
        unsafe { queue.deliver_for_test(&progress(sequence)) };
    }

    let mut first = Vec::new();
    assert_eq!(queue.drain(&mut first, 2), 2);
    let rest = drain_all(&queue);

    let order = first
        .iter()
        .chain(rest.iter())
        .map(|record| record.sequence)
        .collect::<Vec<_>>();
    assert_eq!(order, vec![0, 1, 2, 3, 4]);
}

#[test]
fn full_queue_drops_and_counts() {
    let queue = ModelOpenEventQueue::with_capacity(OperationId(1), 4);
    for sequence in 0..10 {
        unsafe { queue.deliver_for_test(&progress(sequence)) };
    }

    assert_eq!(queue.len(), 4);
    assert_eq!(queue.dropped(), 6);
    assert_eq!(queue.rejected(), 0);
    let kept = drain_all(&queue)
        .iter()
        .map(|record| record.sequence)
        .collect::<Vec<_>>();
    assert_eq!(kept, vec![0, 1, 2, 3], "the oldest records are kept");
}

#[test]
fn rejections_are_counted_not_queued() {
    let queue = ModelOpenEventQueue::new(OperationId(1));
    let mut short = progress(1);
    short.struct_size = 8;
    let mut wrong_abi = progress(2);
    wrong_abi.abi_version = RUNTIME_EVENT_V1_ABI_VERSION + 1;
    let mut oversized = progress(3);
    oversized.detail_len = u64::MAX;

    unsafe {
        queue.deliver_for_test(std::ptr::null());
        queue.deliver_for_test(&short);
        queue.deliver_for_test(&wrong_abi);
        queue.deliver_for_test(&oversized);
    }

    assert!(queue.is_empty());
    assert_eq!(queue.rejected(), 4);
    assert_eq!(queue.dropped(), 0);
}

/// Ported from the removed mutex-ingress concurrency test: callbacks from
/// many native threads land every record, with nothing serializing them.
#[test]
fn concurrent_native_threads_enqueue_all_records() {
    const THREADS: u64 = 8;
    const PER_THREAD: u64 = 25;
    let queue = ModelOpenEventQueue::new(OperationId(7));
    let barrier = Arc::new(Barrier::new(THREADS as usize));

    thread::scope(|scope| {
        for thread_index in 0..THREADS {
            let queue = &queue;
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                barrier.wait();
                for sequence in 0..PER_THREAD {
                    let event = progress(thread_index * PER_THREAD + sequence);
                    unsafe { queue.deliver_for_test(&event) };
                }
            });
        }
    });

    let mut sequences = drain_all(&queue)
        .iter()
        .map(|record| record.sequence)
        .collect::<Vec<_>>();
    sequences.sort_unstable();
    assert_eq!(sequences, (0..THREADS * PER_THREAD).collect::<Vec<_>>());
    assert_eq!(queue.dropped(), 0);
}

#[test]
fn unknown_kind_survives_to_consumer() {
    let queue = ModelOpenEventQueue::new(OperationId(1));
    unsafe { queue.deliver_for_test(&raw_event(RawRuntimeEventKind(9999), 1)) };

    let records = drain_all(&queue);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].kind, 9999);
    assert_eq!(records[0].to_event().kind, RuntimeEventKind::Unknown(9999));
}

#[test]
fn registration_points_the_native_reporter_at_the_queue() {
    let queue = ModelOpenEventQueue::new(OperationId(3));
    let registration = ModelOpenEventReporterRegistration::new(&queue);
    let reporter = unsafe { &*registration.reporter_ptr() };
    let callback = reporter.callback.expect("callback installed");

    unsafe { callback(&progress(5), reporter.user_data) };
    unsafe { callback(&progress(6), std::ptr::null_mut()) };

    assert_eq!(reporter.abi_version, RUNTIME_EVENT_V1_ABI_VERSION);
    assert_eq!(queue.operation_id(), OperationId(3));
    let sequences = drain_all(&queue)
        .iter()
        .map(|record| record.sequence)
        .collect::<Vec<_>>();
    assert_eq!(sequences, vec![5], "a null user_data is ignored");
}

#[test]
fn default_capacity_holds_a_full_open_without_draining() {
    let queue = ModelOpenEventQueue::new(OperationId(1));
    for sequence in 0..MODEL_OPEN_RECORD_CAPACITY as u64 {
        unsafe { queue.deliver_for_test(&progress(sequence)) };
    }
    assert_eq!(queue.len(), MODEL_OPEN_RECORD_CAPACITY);
    assert_eq!(queue.dropped(), 0);
}

#[test]
fn zero_capacity_is_clamped_to_one() {
    let queue = ModelOpenEventQueue::with_capacity(OperationId(1), 0);
    unsafe {
        queue.deliver_for_test(&progress(1));
        queue.deliver_for_test(&progress(2));
    }
    assert_eq!(queue.len(), 1);
    assert_eq!(queue.dropped(), 1);
}
