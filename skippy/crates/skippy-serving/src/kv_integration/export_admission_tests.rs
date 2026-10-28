use std::sync::atomic::Ordering;

use super::exact_state::{CaptureAdmission, ExactStateAdmissionCredit};
use super::*;

#[test]
fn pre_export_credit_transfers_once_and_releases_on_every_queue_terminal_path() {
    for scenario in 0..4 {
        let (sender, receiver) = std::sync::mpsc::sync_channel(usize::from(scenario != 1));
        let mut receiver = Some(receiver);
        if scenario == 2 {
            receiver.take();
        }
        let inflight = Mutex::new(BTreeSet::from(["page".to_owned()]));
        let total = Arc::new(AtomicUsize::new(0));
        let best = Arc::new(AtomicUsize::new(0));
        let bytes = Arc::new(AtomicU64::new(0));
        let mut credit =
            ExactStateAdmissionCredit::acquire(&total, &best, CaptureAdmission::Continuation)
                .unwrap();
        assert!(credit.reserve_bytes(&bytes, 8, 8));
        let pending = PendingExactStateRecord {
            page_id: "page".to_owned(),
            payload: ExactStatePayload::full_state(vec![1; 8]),
            extra: ExactStateExtra::default(),
            namespace: "test".to_owned(),
            token_ids: vec![1],
            l3_fill_claim: None,
            write_through_l3: false,
            l2_promotion_digest: None,
            l3_cost: None,
            admission: super::RecordAdmission::Credit(credit),
        };
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let count = AtomicUsize::new(0);
        let healthy = AtomicBool::new(scenario != 3);
        let panics = AtomicU64::new(0);
        let admission = enqueue_exact_state_record(
            &sender, &inflight, &queued, &dropped, &count, &bytes, 8, &healthy, pending,
        );
        if scenario == 0 {
            assert_eq!(admission, ExactStateRecordAdmission::Queued);
            assert_eq!(
                bytes.load(Ordering::Acquire),
                8,
                "export is not charged twice"
            );
            run_exact_state_record_job(
                ExactStateWorkerHandles {
                    inflight_records: &inflight,
                    dropped: &dropped,
                    pending_count: &count,
                    queue_bytes: &bytes,
                    worker_healthy: &healthy,
                    worker_panics: &panics,
                },
                None,
                receiver.unwrap().recv().unwrap(),
                |_| Ok(()),
            );
        } else {
            assert_ne!(admission, ExactStateRecordAdmission::Queued);
        }
        assert_eq!(total.load(Ordering::Acquire), 0, "scenario {scenario}");
        assert_eq!(bytes.load(Ordering::Acquire), 0, "scenario {scenario}");
        assert_eq!(count.load(Ordering::Acquire), 0, "scenario {scenario}");
        assert!(inflight.lock().unwrap().is_empty());
    }
}
