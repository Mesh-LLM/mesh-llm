use super::*;
use crate::kv_integration::ExactStateRecordAdmission;
use std::sync::atomic::Ordering;

#[test]
fn l1_is_visible_while_l3_spill_is_blocked() {
    let radix = Mutex::new(UnifiedRadixCache::new());
    let blobs = Mutex::new(CacheBlobStore::new(4));
    let (root, tier) = test_l3("l1-before-l3");
    let budget = StorageBudget::new();
    let spill_gate = (Mutex::new((false, false)), std::sync::Condvar::new());

    std::thread::scope(|scope| {
        let spill_gate_ref = &spill_gate;
        let before_l3_spill = move || {
            let (state, ready) = spill_gate_ref;
            let mut state = state.lock().unwrap();
            state.0 = true;
            ready.notify_one();
            while !state.1 {
                state = ready.wait(state).unwrap();
            }
        };
        let worker_radix = &radix;
        let worker_blobs = &blobs;
        let worker_tier = &tier;
        let worker_budget = &budget;
        let worker = scope.spawn(move || {
            store_exact_radix_record_with_codec(
                worker_radix,
                worker_blobs,
                1,
                limits(0, 0),
                None,
                DurableRecordTarget {
                    l3: Some(worker_tier),
                    cachegen_enabled: false,
                    before_l3_spill: Some(&before_l3_spill),
                },
                pending("first", &[1, 2], b"first-exact-state", worker_budget),
            )
        });

        let (state, ready) = &spill_gate;
        let mut state = state.lock().unwrap();
        while !state.0 {
            state = ready.wait(state).unwrap();
        }
        drop(state);

        let visible_page_id = radix
            .lock()
            .unwrap()
            .lookup_recurrent("model", &[1, 2])
            .map(|lookup| lookup.value.page_id);

        let mut state = spill_gate.0.lock().unwrap();
        state.1 = true;
        ready.notify_one();
        drop(state);
        worker.join().unwrap().unwrap();

        assert_eq!(visible_page_id.as_deref(), Some("first"));
    });

    assert!(
        tier.locate_longest("model", &[1, 2], 2).unwrap().is_some(),
        "released spill must still persist the durable entry"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn l3_refusal_preserves_l1_record() {
    let radix = Mutex::new(UnifiedRadixCache::new());
    let blobs = Mutex::new(CacheBlobStore::new(4));
    let root = std::env::temp_dir()
        .join("skippy-server-l3-tests")
        .join(format!("refusal-preserves-l1-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let tier = L3Tier::open(&root, 4, "blake3:test-tier".to_string(), 4096).unwrap();
    let budget = StorageBudget::new();

    store_exact_radix_record(
        &radix,
        &blobs,
        1,
        limits(0, 0),
        None,
        Some(&tier),
        pending("first", &[1, 2], b"first-exact-state", &budget),
    )
    .unwrap();

    assert_eq!(
        radix
            .lock()
            .unwrap()
            .lookup_recurrent("model", &[1, 2])
            .expect("L3 refusal must leave the L1 entry usable")
            .value
            .page_id,
        "first"
    );
    assert!(
        tier.locate_longest("model", &[1, 2], 2).unwrap().is_none(),
        "refused durable entry must not be published in L3"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn resident_only_exact_record_does_not_spill_to_l3() {
    let radix = Mutex::new(UnifiedRadixCache::new());
    let blobs = Mutex::new(CacheBlobStore::new(4));
    let (root, tier) = test_l3("resident-only");
    let budget = StorageBudget::new();
    let mut record = pending("full-prompt", &[1, 2], b"full-prompt-state", &budget);
    record.write_through_l3 = false;

    store_exact_radix_record(&radix, &blobs, 1, limits(0, 0), None, Some(&tier), record).unwrap();

    assert_eq!(
        radix
            .lock()
            .unwrap()
            .lookup_recurrent("model", &[1, 2])
            .expect("resident-only exact state must remain available in L1")
            .value
            .page_id,
        "full-prompt"
    );
    assert!(
        tier.locate_longest("model", &[1, 2], 2).unwrap().is_none(),
        "resident-only exact state must not create a durable manifest"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn blocked_l3_spill_does_not_head_of_line_block_later_l1_records() {
    struct SpillPauseGuard(Arc<std::sync::atomic::AtomicBool>);

    impl Drop for SpillPauseGuard {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }

    let root = std::env::temp_dir()
        .join("skippy-server-l3-manager-tests")
        .join(format!("l1-while-l3-blocked-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
    let mut config = enabled_auto_config("future/model");
    config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::FullState;
    let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
        &config,
        Some(ModelStateKind::Dense),
        None,
        Some(manager),
        None,
    )
    .unwrap()
    .expect("disk-backed exact cache should be enabled");
    let budget = StorageBudget::new();
    let wait = std::time::Duration::from_secs(30);

    kv.l3_spill_worker_pause.store(true, Ordering::Release);
    let spill_pause_guard = SpillPauseGuard(kv.l3_spill_worker_pause.clone());
    assert_eq!(
        kv.enqueue_exact_state_record(pending("first", &[1, 2], b"first-exact-state", &budget,)),
        ExactStateRecordAdmission::Queued,
    );
    kv.wait_for_exact_state_recording(wait)
        .expect("first L1 record should publish");
    let deadline = std::time::Instant::now() + wait;
    while kv.l3_spill_worker_received.load(Ordering::Acquire) == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "durable worker did not receive the first spill"
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    assert_eq!(
        kv.enqueue_exact_state_record(pending(
            "second",
            &[1, 2, 3],
            b"second-exact-state",
            &budget,
        )),
        ExactStateRecordAdmission::Queued,
    );
    kv.wait_for_exact_state_recording(wait)
        .expect("second L1 record must not wait for the first L3 spill");
    let both_visible = {
        let mut radix = kv.radix.lock().unwrap();
        radix.recurrent_exact("model", &[1, 2]).is_some()
            && radix.recurrent_exact("model", &[1, 2, 3]).is_some()
    };
    drop(spill_pause_guard);

    assert!(
        both_visible,
        "both L1 records must be visible while the first durable spill is blocked"
    );
    drop(kv);
    let _ = std::fs::remove_dir_all(root);
}
