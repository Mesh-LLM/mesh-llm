//! Native cold-export credits and identity claims follow the queued payload.
use super::*;
use crate::kv_integration::resident_offload::NativeOffloadClaim;
use std::sync::atomic::Ordering;

fn pending(token: i32, bytes: usize) -> PendingDurableSpill {
    PendingDurableSpill {
        page_id: format!("native-{token}"),
        payload: skippy_cache::ExactStatePayload::kv_recurrent(
            vec![7; bytes / 2],
            vec![9; bytes - bytes / 2],
        ),
        extra: super::super::super::ExactStateExtra::default(),
        namespace: "native-test".into(),
        token_ids: vec![token],
        native_raw: true,
        l3_cost: None,
    }
}

#[test]
fn native_spill_queue_full_and_disconnect_release_bytes_and_claims() {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let sender = DurableSpillSender::new(tx, 64);
    let claims = Arc::new(Mutex::new(BTreeSet::new()));
    for (token, accepted) in [(1, true), (2, false)] {
        let claim = NativeOffloadClaim::acquire(&claims, "native-test", &[token]).unwrap();
        let reservation = sender.try_reserve(16).unwrap();
        assert_eq!(
            sender.send_reserved(pending(token, 16), reservation, claim),
            accepted
        );
        assert_eq!(sender.held_bytes.load(Ordering::Acquire), 16);
        assert_eq!(claims.lock().unwrap().len(), 1);
    }
    let active = rx.recv().unwrap();
    assert!(NativeOffloadClaim::acquire(&claims, "native-test", &[1]).is_none());
    assert_eq!(sender.held_bytes.load(Ordering::Acquire), 16);
    drop(active);
    drop(rx);
    let claim = NativeOffloadClaim::acquire(&claims, "native-test", &[1]).unwrap();
    assert!(!sender.send_reserved(pending(1, 16), sender.try_reserve(16).unwrap(), claim));
    assert_eq!(sender.held_bytes.load(Ordering::Acquire), 0);
    assert!(claims.lock().unwrap().is_empty());
    assert!(sender.try_reserve(64).is_some());
}

#[test]
fn native_spill_underestimated_payload_releases_reservation_and_claim() {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let sender = DurableSpillSender::new(tx, 64);
    let claims = Arc::new(Mutex::new(BTreeSet::new()));
    let claim = NativeOffloadClaim::acquire(&claims, "native-test", &[1]).unwrap();
    assert!(!sender.send_reserved(pending(1, 17), sender.try_reserve(16).unwrap(), claim));
    assert_eq!(sender.held_bytes.load(Ordering::Acquire), 0);
    assert!(claims.lock().unwrap().is_empty());
    assert!(matches!(
        rx.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
}

#[test]
fn native_spill_blocked_and_refused_write_retains_then_releases_credits() {
    let root = tempfile::tempdir().unwrap();
    let tier = Arc::new(L3Tier::open(root.path(), 1, "native-spill-refusal".into(), 4096).unwrap());
    let received = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let pause = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let (sender, worker) = start_durable_spill_worker(
        Some(tier.clone()),
        false,
        64,
        "native-spill-test",
        received.clone(),
        pause.clone(),
    )
    .unwrap();
    let sender = sender.unwrap();
    let claims = Arc::new(Mutex::new(BTreeSet::new()));
    let claim = NativeOffloadClaim::acquire(&claims, "native-test", &[1]).unwrap();
    assert!(sender.send_reserved(pending(1, 64), sender.try_reserve(64).unwrap(), claim));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while received.load(Ordering::Acquire) == 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let writing = received.load(Ordering::Acquire) == 1;
    let held = sender.held_bytes.load(Ordering::Acquire);
    let refused = sender.try_reserve(1).is_none();
    let claimed = NativeOffloadClaim::acquire(&claims, "native-test", &[1]).is_none();
    let held_bytes = sender.held_bytes.clone();
    // Always release and join before asserting, including on a failed admission.
    pause.store(false, Ordering::Release);
    drop(sender);
    worker.unwrap().join().unwrap();
    assert!(writing);
    assert_eq!(held, 64);
    assert!(refused && claimed);
    assert_eq!(held_bytes.load(Ordering::Acquire), 0);
    assert!(claims.lock().unwrap().is_empty());
    assert!(
        tier.locate_longest("native-test", &[1], 1)
            .unwrap()
            .is_none()
    );
}
