//! Persistence failure fallback and bounded shutdown coverage for `LoggingService`.

use super::*;

const RECEIVE_TIMEOUT: Duration = Duration::from_secs(1);

async fn recv_with_timeout<T>(receiver: &mut mpsc::UnboundedReceiver<T>, context: &str) -> T {
    tokio::time::timeout(RECEIVE_TIMEOUT, receiver.recv())
        .await
        .expect(context)
        .expect(context)
}

#[tokio::test]
async fn worker_sink_failure_enqueues_one_canonical_fallback_and_suppresses_its_failure() {
    let (sink, mut attempts) = TestSink::failing_with_attempt_notifications();
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    assert!(svc.spawn());

    svc.enqueue_event(
        RequestId::new(),
        ReplayChannel::Requests,
        "{\"event\":\"failing\"}".into(),
    )
    .expect("fail-open enqueue");

    // The original delivery and its canonical System fallback both reach the
    // real sink. The fallback's own failure is intentionally not recursive.
    recv_with_timeout(&mut attempts, "original sink attempt").await;
    recv_with_timeout(&mut attempts, "fallback sink attempt").await;
    assert_eq!(svc.persistence_failures(), 2);

    let fallback_events = canonical_persistence_failure_fallbacks(&svc);
    assert_eq!(fallback_events, 1, "one failed delivery emits one fallback");
    assert_eq!(
        svc.writer_ref()
            .fallback_persistence_suppressions
            .load(AtomicOrdering::Relaxed),
        1,
        "a failed fallback must be suppressed instead of self-logging"
    );
    assert_eq!(
        svc.writer_ref()
            .fallback_guard_suppressions
            .load(AtomicOrdering::Relaxed),
        0,
        "fallback persistence suppression does not mean the guard rejected a fallback"
    );
    assert!(svc.shutdown().await);
}

#[tokio::test]
async fn manual_sink_failure_enqueues_a_canonical_system_fallback_without_blocking() {
    let (sink, mut attempts) = TestSink::failing_with_attempt_notifications();
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    );

    svc.enqueue_event(
        RequestId::new(),
        ReplayChannel::Operations,
        "{\"event\":\"manual-failing\"}".into(),
    )
    .expect("fail-open enqueue");
    assert_eq!(svc.pump_sync().await, 1);
    recv_with_timeout(&mut attempts, "manual sink attempt").await;

    let fallback_events = canonical_persistence_failure_fallbacks(&svc);
    assert_eq!(fallback_events, 1);
    assert_eq!(svc.persistence_failures(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn log_store_sqlite_busy_seam_never_blocks_the_shared_tokio_executor() {
    use crate::logging::LogStoreSink;

    let root = tempfile::tempdir().expect("temporary log-store root");
    let store = Arc::new(
        mesh_llm_log_store::LogStore::open(root.path(), Arc::new(mesh_llm_log_store::RealClock))
            .expect("open log store"),
    );
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
    let hook = Arc::new(move || {
        started_tx.send(()).expect("report blocking worker start");
        release_rx
            .lock()
            .expect("release receiver mutex")
            .recv()
            .expect("release blocking worker");
    });
    let sink = Arc::new(LogStoreSink::with_blocking_hook_for_test(
        Arc::clone(&store),
        hook,
    ));

    // The injected hook runs immediately before the synchronous SQLite call.
    // A direct repository call here would park this current-thread executor
    // before it can observe the hook or yield.
    let write = {
        let sink = Arc::clone(&sink);
        tokio::spawn(async move {
            sink.persist_audit_entry("error".into(), "busy seam".into())
                .await
        })
    };
    tokio::task::spawn_blocking(move || started_rx.recv_timeout(RECEIVE_TIMEOUT))
        .await
        .expect("start observer joins")
        .expect("blocking worker started");
    tokio::task::yield_now().await;
    assert!(
        !write.is_finished(),
        "the SQLite operation should be isolated in the blocking pool"
    );
    release_tx.send(()).expect("release blocking SQLite seam");
    assert!(write.await.expect("write task joins").is_ok());
}

#[tokio::test]
async fn spawn_and_shutdown_recover_a_poisoned_worker_handle_lock() {
    let service = Arc::new(LoggingService::new(
        ServiceConfig::default(),
        Arc::new(TestSink::new()),
        Box::new(TestClock::new()),
    ));
    let poison_target = Arc::clone(&service);

    assert!(
        std::thread::spawn(move || {
            let _worker_handle = poison_target.worker_handle_lock_for_test();
            panic!("poison worker handle lock for recovery coverage");
        })
        .join()
        .is_err()
    );

    assert!(service.spawn());
    assert!(service.shutdown().await);
}

#[tokio::test]
async fn test_shutdown_drains_accepted_entries_joins_and_allows_restart() {
    let (sink, mut started, mut completed, release) = BlockingAuditSink::new();
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    assert!(svc.spawn());

    svc.enqueue_event(request_id, ReplayChannel::System, "{\"event\":1}".into())
        .unwrap();
    recv_with_timeout(&mut started, "first accepted entry began persistence").await;
    svc.enqueue_event(request_id, ReplayChannel::System, "{\"event\":2}".into())
        .unwrap();
    svc.enqueue_event(request_id, ReplayChannel::System, "{\"event\":3}".into())
        .unwrap();
    assert_eq!(svc.persistence_outstanding(), 3);

    // Poll until shutdown has frozen delivery and is waiting on the blocked
    // worker. A zero-duration timeout polls without a wall-clock sleep.
    let shutdown = svc.shutdown();
    tokio::pin!(shutdown);
    assert!(
        tokio::time::timeout(Duration::ZERO, &mut shutdown)
            .await
            .is_err(),
        "blocked worker must keep shutdown pending until released"
    );

    // New request-path events still reach replay but cannot be silently held
    // after the worker's drain boundary.
    svc.enqueue_event(request_id, ReplayChannel::System, "{\"event\":4}".into())
        .unwrap();
    assert_eq!(svc.persistence_queue_drops(), 1);
    assert_eq!(svc.persistence_outstanding(), 3);

    release.notify_one();
    assert!(shutdown.await, "shutdown must join after draining");
    let completed_messages = [
        recv_with_timeout(&mut completed, "first drained entry").await,
        recv_with_timeout(&mut completed, "second drained entry").await,
        recv_with_timeout(&mut completed, "third drained entry").await,
    ];
    assert!(completed_messages[0].contains("\\\"event\\\":1"));
    assert!(completed_messages[1].contains("\\\"event\\\":2"));
    assert!(completed_messages[2].contains("\\\"event\\\":3"));
    assert!(completed.try_recv().is_err());
    assert_eq!(svc.persistence_outstanding(), 0);
    assert_eq!(svc.persistence_shutdown_losses(), 0);
    assert!(!svc.is_spawned());

    // The same service can reinitialize a fresh worker after the old task has
    // been joined; the stopped entry above is not replayed to persistence.
    assert!(svc.spawn());
    svc.enqueue_event(request_id, ReplayChannel::System, "{\"event\":5}".into())
        .unwrap();
    let restarted = recv_with_timeout(&mut completed, "restarted worker entry").await;
    assert!(restarted.contains("\\\"event\\\":5"));
    assert!(svc.shutdown().await);
}

#[tokio::test]
async fn test_stalled_shutdown_is_bounded_aborts_and_accounts_for_owned_entries() {
    let (sink, mut started, mut completed, _release) = BlockingAuditSink::new();
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    )
    .with_shutdown_drain_timeout(Duration::ZERO);
    let request_id = RequestId::new();
    assert!(svc.spawn());

    svc.enqueue_event(request_id, ReplayChannel::Requests, "{\"event\":1}".into())
        .unwrap();
    recv_with_timeout(&mut started, "worker entered the injected stall").await;
    svc.enqueue_event(request_id, ReplayChannel::Requests, "{\"event\":2}".into())
        .unwrap();
    svc.enqueue_event(request_id, ReplayChannel::Requests, "{\"event\":3}".into())
        .unwrap();
    assert_eq!(svc.persistence_outstanding(), 3);

    assert!(svc.shutdown().await);
    assert!(!svc.is_spawned());
    assert_eq!(svc.persistence_outstanding(), 0);
    assert_eq!(svc.persistence_shutdown_losses(), 3);
    assert!(
        completed.try_recv().is_err(),
        "aborted stall must not leak a worker"
    );

    // A clean worker can be started immediately after bounded abort control.
    assert!(svc.spawn());
    svc.enqueue_event(request_id, ReplayChannel::Requests, "{\"event\":4}".into())
        .unwrap();
    let recovered = recv_with_timeout(&mut completed, "restarted worker entry").await;
    assert!(recovered.contains("\\\"event\\\":4"));
    assert_eq!(svc.persistence_shutdown_losses(), 3);
    assert!(svc.shutdown().await);
}
