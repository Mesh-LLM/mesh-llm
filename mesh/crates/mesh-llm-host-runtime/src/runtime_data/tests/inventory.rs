use super::*;

const WAITER_POLL_LIMIT: usize = 100;

async fn wait_for_inventory_scan_waiters(collector: &RuntimeDataCollector, expected: usize) {
    for _ in 0..WAITER_POLL_LIMIT {
        if collector.inventory_scan_waiter_count() == expected {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!(
        "inventory scan waiter count did not reach {expected}; got {}",
        collector.inventory_scan_waiter_count()
    );
}

#[tokio::test]
async fn runtime_data_inventory_single_flight_scan_coalesces() {
    let collector = RuntimeDataCollector::new();
    let scan_count = Arc::new(AtomicUsize::new(0));
    let (release_tx, release_rx) = std::sync::mpsc::channel();

    let first = {
        let collector = collector.clone();
        let scan_count = scan_count.clone();
        tokio::spawn(async move {
            collector
                .coalesce_local_inventory_scan_outcome(move || {
                    scan_count.fetch_add(1, Ordering::SeqCst);
                    release_rx
                        .recv()
                        .expect("test should release inventory scan");
                    let mut snapshot = LocalModelInventorySnapshot::default();
                    snapshot.model_names.insert("Qwen3-8B".into());
                    snapshot
                        .size_by_name
                        .insert("Qwen3-8B".into(), 8_000_000_000);
                    Ok(snapshot)
                })
                .await
        })
    };

    wait_for_inventory_scan_waiters(&collector, 1).await;

    let second = {
        let collector = collector.clone();
        tokio::spawn(async move {
            collector
                .coalesce_local_inventory_scan_outcome(
                    || Ok(LocalModelInventorySnapshot::default()),
                )
                .await
        })
    };

    wait_for_inventory_scan_waiters(&collector, 2).await;
    release_tx.send(()).expect("test should release scan");

    let first_outcome = first.await.expect("first inventory scan task should join");
    let second_outcome = second
        .await
        .expect("second inventory scan task should join");
    let first_outcome = first_outcome.expect("first scan should succeed");
    let second_outcome = second_outcome.expect("second scan should share success");

    assert_eq!(scan_count.load(Ordering::SeqCst), 1);
    assert_eq!(first_outcome.snapshot, second_outcome.snapshot);
    assert_eq!(
        first_outcome.disposition,
        InventoryScanDisposition::Executed
    );
    assert_eq!(
        second_outcome.disposition,
        InventoryScanDisposition::Coalesced
    );
    assert_eq!(collector.local_inventory_snapshot(), first_outcome.snapshot);
    assert!(
        collector
            .local_inventory_snapshot()
            .model_names
            .contains("Qwen3-8B")
    );
}

#[tokio::test]
async fn runtime_data_inventory_scan_panic_fans_out_error_and_preserves_snapshot() {
    let collector = RuntimeDataCollector::new();
    let seeded = LocalModelInventorySnapshot {
        model_names: HashSet::from(["last-good".to_string()]),
        ..Default::default()
    };
    collector
        .coalesce_local_inventory_scan({
            let seeded = seeded.clone();
            move || seeded
        })
        .await
        .expect("seed scan should succeed");

    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let first = {
        let collector = collector.clone();
        tokio::spawn(async move {
            collector
                .coalesce_local_inventory_scan_outcome(move || -> super::InventoryScanResult {
                    release_rx
                        .recv()
                        .expect("test should release inventory scan");
                    panic!("scan backend exploded")
                })
                .await
        })
    };
    wait_for_inventory_scan_waiters(&collector, 1).await;
    let second = {
        let collector = collector.clone();
        tokio::spawn(async move {
            collector
                .coalesce_local_inventory_scan_outcome(
                    || Ok(LocalModelInventorySnapshot::default()),
                )
                .await
        })
    };

    wait_for_inventory_scan_waiters(&collector, 2).await;
    release_tx.send(()).expect("test should release scan");

    let first_error = first
        .await
        .expect("first scan task should join")
        .expect_err("panic must surface as a scan failure");
    let second_error = second
        .await
        .expect("second scan task should join")
        .expect_err("coalesced waiter must receive same scan failure");

    assert_eq!(first_error, second_error);
    assert!(matches!(
        first_error,
        InventoryScanError::TaskPanicked(message) if message.contains("scan backend exploded")
    ));
    assert_eq!(collector.local_inventory_snapshot(), seeded);
}
