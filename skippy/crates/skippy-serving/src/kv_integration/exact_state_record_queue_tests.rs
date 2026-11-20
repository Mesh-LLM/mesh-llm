//! Tests for the bounded exact-state record queue and its worker.

use skippy_cache::ExactStatePayload;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    mpsc::sync_channel,
};

use super::{
    BTreeSet, EXACT_STATE_RECORD_CAPACITY, ExactStateExtra, ExactStateRecordAdmission,
    ExactStateRecordWorker, ExactStateWorkerHandles, KvLifecycleEvent, KvLifecycleObserver,
    PendingExactStateRecord, enqueue_exact_state_record, run_exact_state_record_job,
};
use crate::kv_integration::exact_state::{CaptureAdmission, ExactStateAdmissionCredit};

/// A fresh, observable budget: the SAME counters back every credit a test
/// creates, so release paths are asserted on real accounting, not on
/// throwaway stubs.
fn budget() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
    (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)))
}

fn credit(
    total: &Arc<AtomicUsize>,
    best_effort: &Arc<AtomicUsize>,
    class: CaptureAdmission,
) -> ExactStateAdmissionCredit {
    ExactStateAdmissionCredit::acquire(total, best_effort, class)
        .expect("a fresh budget must admit")
}

struct RecordingObserver(Arc<Mutex<Vec<KvLifecycleEvent>>>);

impl KvLifecycleObserver for RecordingObserver {
    fn observe(&self, event: KvLifecycleEvent) {
        self.0.lock().unwrap().push(event);
    }
}

fn pending(page_id: &str, admission_credit: ExactStateAdmissionCredit) -> PendingExactStateRecord {
    PendingExactStateRecord {
        page_id: page_id.to_string(),
        payload: ExactStatePayload::full_state(vec![1]),
        extra: ExactStateExtra::default(),
        namespace: "test".to_string(),
        retention_priority: 0,
        token_ids: vec![1],
        l3_fill_claim: None,
        write_through_l3: true,
        l2_promotion_digest: None,
        l3_cost: None,
        admission_credit,
    }
}

fn pending_with_bytes(page_id: &str, bytes: usize) -> PendingExactStateRecord {
    let (total, best_effort) = budget();
    PendingExactStateRecord {
        payload: ExactStatePayload::full_state(vec![1; bytes]),
        ..pending(
            page_id,
            credit(&total, &best_effort, CaptureAdmission::BestEffort),
        )
    }
}

const CAP: u64 = 1024;

#[test]
fn final_worker_owner_drains_queued_records_before_drop_returns() {
    let (sender, receiver) = sync_channel(2);
    let completed = Arc::new(AtomicUsize::new(0));
    let worker_completed = completed.clone();
    let task = std::thread::spawn(move || {
        while receiver.recv().is_ok() {
            worker_completed.fetch_add(1, Ordering::Release);
        }
    });
    let worker = Arc::new(ExactStateRecordWorker::new(sender, task, None, None));
    let (total, best_effort) = budget();
    worker.with_sender(|sender| {
        sender
            .unwrap()
            .send(pending(
                "latest",
                credit(&total, &best_effort, CaptureAdmission::BestEffort),
            ))
            .unwrap()
    });

    drop(worker);

    assert_eq!(completed.load(Ordering::Acquire), 1);
}

#[test]
fn queue_is_bounded_in_bytes_not_entries() {
    let (sender, _receiver) = sync_channel(EXACT_STATE_RECORD_CAPACITY);
    let inflight = Mutex::new(BTreeSet::from(["a".to_string(), "b".to_string()]));
    let queued = AtomicU64::new(0);
    let dropped = AtomicU64::new(0);
    let pending_count = AtomicUsize::new(0);
    let queue_bytes = AtomicU64::new(0);
    let worker_healthy = AtomicBool::new(true);

    assert_eq!(
        enqueue_exact_state_record(
            &sender,
            &inflight,
            &queued,
            &dropped,
            &pending_count,
            &queue_bytes,
            CAP,
            &worker_healthy,
            pending_with_bytes("a", 700),
        ),
        ExactStateRecordAdmission::Queued
    );
    assert_eq!(queue_bytes.load(Ordering::Relaxed), 700);

    // Plenty of entry slots left; the bytes are what is full.
    assert_eq!(
        enqueue_exact_state_record(
            &sender,
            &inflight,
            &queued,
            &dropped,
            &pending_count,
            &queue_bytes,
            CAP,
            &worker_healthy,
            pending_with_bytes("b", 700),
        ),
        ExactStateRecordAdmission::DroppedFull
    );
    assert_eq!(
        queue_bytes.load(Ordering::Relaxed),
        700,
        "a dropped record left bytes claimed"
    );
    assert!(!inflight.lock().unwrap().contains("b"));
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert_eq!(pending_count.load(Ordering::Relaxed), 1);
}

#[test]
fn one_oversized_record_is_admitted_into_an_empty_queue() {
    let (sender, _receiver) = sync_channel(EXACT_STATE_RECORD_CAPACITY);
    let inflight = Mutex::new(BTreeSet::from(["huge".to_string()]));
    let queued = AtomicU64::new(0);
    let dropped = AtomicU64::new(0);
    let pending_count = AtomicUsize::new(0);
    let queue_bytes = AtomicU64::new(0);
    let worker_healthy = AtomicBool::new(true);

    assert_eq!(
        enqueue_exact_state_record(
            &sender,
            &inflight,
            &queued,
            &dropped,
            &pending_count,
            &queue_bytes,
            CAP,
            &worker_healthy,
            pending_with_bytes("huge", 4096),
        ),
        ExactStateRecordAdmission::Queued,
        "a large model could never record if one export over the cap were refused"
    );
    assert_eq!(queue_bytes.load(Ordering::Relaxed), 4096);
    assert!(queue_bytes.load(Ordering::Acquire) >= CAP);
}

#[test]
fn finishing_a_record_releases_its_bytes() {
    let inflight = Mutex::new(BTreeSet::from(["done".to_string()]));
    let dropped = AtomicU64::new(0);
    let pending_count = AtomicUsize::new(1);
    let queue_bytes = AtomicU64::new(300);
    let worker_healthy = AtomicBool::new(true);
    let worker_panics = AtomicU64::new(0);

    run_exact_state_record_job(
        ExactStateWorkerHandles {
            inflight_records: &inflight,
            dropped: &dropped,
            pending_count: &pending_count,
            queue_bytes: &queue_bytes,
            worker_healthy: &worker_healthy,
            worker_panics: &worker_panics,
        },
        None,
        pending_with_bytes("done", 300),
        |_| Ok(()),
    );

    assert_eq!(queue_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(pending_count.load(Ordering::Relaxed), 0);
    assert!(inflight.lock().unwrap().is_empty());
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
}

#[test]
fn full_queue_drops_optional_record_and_releases_inflight_page() {
    let (total, best_effort) = budget();
    let (sender, receiver) = sync_channel(1);
    // One BestEffort record occupies the channel buffer; its credit stays
    // observable in `total`.
    sender
        .send(pending(
            "queued",
            credit(&total, &best_effort, CaptureAdmission::BestEffort),
        ))
        .unwrap();
    let inflight = Mutex::new(BTreeSet::from(["dropped".to_string()]));
    let queued = AtomicU64::new(0);
    let dropped = AtomicU64::new(0);
    let pending_count = AtomicUsize::new(0);
    let queue_bytes = AtomicU64::new(0);
    let worker_healthy = AtomicBool::new(true);

    assert_eq!(
        enqueue_exact_state_record(
            &sender,
            &inflight,
            &queued,
            &dropped,
            &pending_count,
            &queue_bytes,
            CAP,
            &worker_healthy,
            pending(
                "dropped",
                credit(&total, &best_effort, CaptureAdmission::Continuation)
            ),
        ),
        ExactStateRecordAdmission::DroppedFull
    );
    assert!(!inflight.lock().unwrap().contains("dropped"));
    assert_eq!(queued.load(Ordering::Relaxed), 0);
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert_eq!(pending_count.load(Ordering::Relaxed), 0);
    assert_eq!(
        total.load(Ordering::Acquire),
        1,
        "the Full rejection must release the dropped record's credit; the buffered one stays"
    );
    assert_eq!(
        best_effort.load(Ordering::Acquire),
        1,
        "the buffered BestEffort credit is still outstanding"
    );

    // Receiver drop: discarding the channel drops the buffered record and
    // releases its credit too.
    drop(sender);
    drop(receiver);
    assert_eq!(
        total.load(Ordering::Acquire),
        0,
        "receiver drop released the buffered credit"
    );
    assert_eq!(best_effort.load(Ordering::Acquire), 0);
}

#[test]
fn disconnected_worker_releases_inflight_page() {
    let (total, best_effort) = budget();
    let (sender, receiver) = sync_channel(1);
    drop(receiver);
    let inflight = Mutex::new(BTreeSet::from(["orphaned".to_string()]));
    let queued = AtomicU64::new(0);
    let dropped = AtomicU64::new(0);
    let pending_count = AtomicUsize::new(0);
    let queue_bytes = AtomicU64::new(0);
    let worker_healthy = AtomicBool::new(true);

    assert_eq!(
        enqueue_exact_state_record(
            &sender,
            &inflight,
            &queued,
            &dropped,
            &pending_count,
            &queue_bytes,
            CAP,
            &worker_healthy,
            pending(
                "orphaned",
                credit(&total, &best_effort, CaptureAdmission::Continuation)
            ),
        ),
        ExactStateRecordAdmission::WorkerStopped
    );
    assert!(!inflight.lock().unwrap().contains("orphaned"));
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert_eq!(pending_count.load(Ordering::Relaxed), 0);
    assert_eq!(
        total.load(Ordering::Acquire),
        0,
        "the disconnected channel must release the orphaned credit"
    );
    assert_eq!(best_effort.load(Ordering::Acquire), 0);
}

#[test]
fn worker_keeps_page_inflight_until_record_finishes() {
    let (total, best_effort) = budget();
    let (sender, receiver) = sync_channel(1);
    let inflight = Arc::new(Mutex::new(BTreeSet::from(["page".to_string()])));
    let worker_inflight = inflight.clone();
    let queued = AtomicU64::new(0);
    let dropped = AtomicU64::new(0);
    let pending_count = Arc::new(AtomicUsize::new(0));
    let worker_pending_count = pending_count.clone();
    let queue_bytes = AtomicU64::new(0);
    let worker_healthy = AtomicBool::new(true);

    assert_eq!(
        enqueue_exact_state_record(
            &sender,
            &inflight,
            &queued,
            &dropped,
            &pending_count,
            &queue_bytes,
            CAP,
            &worker_healthy,
            pending(
                "page",
                credit(&total, &best_effort, CaptureAdmission::Continuation)
            ),
        ),
        ExactStateRecordAdmission::Queued
    );
    assert!(inflight.lock().unwrap().contains("page"));
    assert_eq!(
        total.load(Ordering::Acquire),
        1,
        "the credit is outstanding while the record sits in the channel"
    );

    let worker = std::thread::spawn(move || {
        let pending = receiver.recv().unwrap();
        assert!(worker_inflight.lock().unwrap().contains(&pending.page_id));
        worker_inflight.lock().unwrap().remove(&pending.page_id);
        worker_pending_count.fetch_sub(1, Ordering::Relaxed);
        // `pending` (and its credit) drops when this closure returns.
    });
    worker.join().unwrap();

    assert!(!inflight.lock().unwrap().contains("page"));
    assert_eq!(queued.load(Ordering::Relaxed), 1);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    assert_eq!(pending_count.load(Ordering::Relaxed), 0);
    assert_eq!(
        total.load(Ordering::Acquire),
        0,
        "the credit released when the record dropped after processing"
    );
}

#[test]
fn worker_success_notifies_completed_not_mere_admission() {
    let inflight = Mutex::new(BTreeSet::from(["written".to_string()]));
    let dropped = AtomicU64::new(0);
    let pending_count = AtomicUsize::new(1);
    let queue_bytes = AtomicU64::new(1);
    let worker_healthy = AtomicBool::new(true);
    let worker_panics = AtomicU64::new(0);
    let events: Arc<Mutex<Vec<KvLifecycleEvent>>> = Arc::default();
    let observer: Arc<dyn KvLifecycleObserver> = Arc::new(RecordingObserver(events.clone()));
    let (total, best_effort) = budget();

    run_exact_state_record_job(
        ExactStateWorkerHandles {
            inflight_records: &inflight,
            dropped: &dropped,
            pending_count: &pending_count,
            queue_bytes: &queue_bytes,
            worker_healthy: &worker_healthy,
            worker_panics: &worker_panics,
        },
        Some(&observer),
        pending(
            "written",
            credit(&total, &best_effort, CaptureAdmission::Continuation),
        ),
        |_| Ok(()),
    );

    assert_eq!(
        *events.lock().unwrap(),
        vec![KvLifecycleEvent::ExactStateRecordCompleted]
    );
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    assert_eq!(
        total.load(Ordering::Acquire),
        0,
        "successful completion released the real credit"
    );
}

#[test]
fn worker_write_error_notifies_failed_not_completed() {
    let inflight = Mutex::new(BTreeSet::from(["broken".to_string()]));
    let dropped = AtomicU64::new(0);
    let pending_count = AtomicUsize::new(1);
    let queue_bytes = AtomicU64::new(1);
    let worker_healthy = AtomicBool::new(true);
    let worker_panics = AtomicU64::new(0);
    let events: Arc<Mutex<Vec<KvLifecycleEvent>>> = Arc::default();
    let observer: Arc<dyn KvLifecycleObserver> = Arc::new(RecordingObserver(events.clone()));
    let (total, best_effort) = budget();

    run_exact_state_record_job(
        ExactStateWorkerHandles {
            inflight_records: &inflight,
            dropped: &dropped,
            pending_count: &pending_count,
            queue_bytes: &queue_bytes,
            worker_healthy: &worker_healthy,
            worker_panics: &worker_panics,
        },
        Some(&observer),
        pending(
            "broken",
            credit(&total, &best_effort, CaptureAdmission::Continuation),
        ),
        |_| anyhow::bail!("simulated write failure"),
    );

    assert_eq!(
        *events.lock().unwrap(),
        vec![KvLifecycleEvent::ExactStateRecordFailed]
    );
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert_eq!(
        total.load(Ordering::Acquire),
        0,
        "the worker error path released the real credit"
    );
    assert_eq!(best_effort.load(Ordering::Acquire), 0);
}

#[test]
fn worker_panic_fails_closed_and_releases_all_record_bookkeeping() {
    let (sender, _receiver) = sync_channel(1);
    let inflight = Mutex::new(BTreeSet::from(["panicked".to_string()]));
    let queued = AtomicU64::new(0);
    let dropped = AtomicU64::new(0);
    let pending_count = AtomicUsize::new(1);
    let queue_bytes = AtomicU64::new(1);
    let worker_healthy = AtomicBool::new(true);
    let worker_panics = AtomicU64::new(0);
    let (total, best_effort) = budget();

    run_exact_state_record_job(
        ExactStateWorkerHandles {
            inflight_records: &inflight,
            dropped: &dropped,
            pending_count: &pending_count,
            queue_bytes: &queue_bytes,
            worker_healthy: &worker_healthy,
            worker_panics: &worker_panics,
        },
        None,
        pending(
            "panicked",
            credit(&total, &best_effort, CaptureAdmission::Continuation),
        ),
        |_| panic!("injected exact-record worker failure"),
    );

    assert!(!worker_healthy.load(Ordering::Acquire));
    assert_eq!(worker_panics.load(Ordering::Relaxed), 1);
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert_eq!(pending_count.load(Ordering::Acquire), 0);
    assert!(inflight.lock().unwrap().is_empty());
    assert_eq!(
        total.load(Ordering::Acquire),
        0,
        "the unwind dropped the record and released its real credit"
    );
    assert_eq!(best_effort.load(Ordering::Acquire), 0);

    inflight.lock().unwrap().insert("later".to_string());
    assert_eq!(
        enqueue_exact_state_record(
            &sender,
            &inflight,
            &queued,
            &dropped,
            &pending_count,
            &queue_bytes,
            CAP,
            &worker_healthy,
            pending(
                "later",
                credit(&total, &best_effort, CaptureAdmission::Continuation)
            ),
        ),
        ExactStateRecordAdmission::WorkerStopped
    );
    assert!(inflight.lock().unwrap().is_empty());
    assert_eq!(dropped.load(Ordering::Relaxed), 2);
    assert_eq!(pending_count.load(Ordering::Acquire), 0);
    assert_eq!(
        total.load(Ordering::Acquire),
        0,
        "the WorkerStopped follow-up also released its real credit"
    );
}
