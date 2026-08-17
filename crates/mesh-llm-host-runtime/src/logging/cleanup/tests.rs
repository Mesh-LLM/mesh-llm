use std::sync::Arc;
use std::time::Duration;

use mesh_llm_log_store::{Clock as StoreClock, FailOpenArtifactCapture, LogStore};

use super::*;
use crate::logging::{LogStoreSink, LoggingDynamicLimits, ServiceConfig, SystemClock};

struct FixedClock(&'static str);

impl StoreClock for FixedClock {
    fn now(&self) -> String {
        self.0.to_string()
    }
}

fn setup(
    now: &'static str,
    retention_ttl_secs: u64,
    _retention_max_rows: u64,
) -> (
    tempfile::TempDir,
    Arc<LogStore>,
    Arc<FailOpenArtifactCapture>,
    Arc<LoggingService>,
) {
    let root = tempfile::tempdir().expect("temporary root");
    let clock: Arc<dyn StoreClock> = Arc::new(FixedClock(now));
    let store = Arc::new(
        LogStore::open(root.path().join("store"), Arc::clone(&clock)).expect("open store"),
    );
    let capture = Arc::new(
        FailOpenArtifactCapture::open(
            root.path().join("artifacts"),
            clock,
            Arc::clone(&store),
            Arc::new(|bytes| bytes.to_vec()),
        )
        .expect("open capture"),
    );
    let service = Arc::new(LoggingService::new_with_dynamic_limits(
        ServiceConfig::default(),
        Arc::new(LogStoreSink::new(Arc::clone(&store))),
        Box::new(SystemClock),
        LoggingDynamicLimits {
            retention_ttl_secs,
            replay_capacity: 8,
        },
    ));
    (root, store, capture, service)
}

fn insert_summary(store: &LogStore, request_id: &str, created_at: &str) {
    store
        .insert_summary(
            request_id, None, None, None, None, created_at, None, None, None,
        )
        .expect("insert summary");
}

fn insert_terminal_summary(store: &LogStore, request_id: &str, occurred_at: &str) {
    insert_summary(store, request_id, occurred_at);
    store
        .write_terminal_event(
            request_id,
            &format!("terminal-{request_id}"),
            r#"{"type":"completed"}"#,
            "completed",
            None,
            occurred_at,
        )
        .expect("write terminal event");
}

#[tokio::test]
async fn ttl_cleanup_uses_live_retention_and_records_a_sanitized_audit() {
    let (_root, store, capture, service) = setup("2026-08-03T12:00:00Z", 172_800, 100);
    insert_terminal_summary(&store, "old", "2026-08-03T10:00:00Z");

    assert_eq!(
        run_cleanup_once(
            Arc::clone(&store),
            Some(capture),
            Arc::clone(&service),
            100,
            3_600
        )
        .await,
        CleanupOutcome::Completed { deleted_count: 0 }
    );
    assert!(store.get_summary("old").expect("get summary").is_some());

    service.apply_dynamic_limits(LoggingDynamicLimits {
        retention_ttl_secs: 3_600,
        replay_capacity: 8,
    });
    assert_eq!(
        run_cleanup_once(Arc::clone(&store), None, Arc::clone(&service), 100, 3_600).await,
        CleanupOutcome::Completed { deleted_count: 2 }
    );
    assert!(store.get_summary("old").expect("get summary").is_none());
    let cleanup_runs: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM cleanup_runs", [], |row| row.get(0))
        .expect("count runs");
    assert_eq!(
        cleanup_runs, 28,
        "two passes each record TTL and cap results for every durable table"
    );

    assert!(service.pump_sync().await >= 2);
    let audit_count: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'logging_cleanup_completed'",
            [],
            |row| row.get(0),
        )
        .expect("read cleanup audits");
    assert_eq!(audit_count, 2);
}

#[tokio::test]
async fn cleanup_failure_is_fail_open_and_records_a_bounded_error_audit() {
    let (_root, store, capture, service) = setup("not-a-timestamp", 3_600, 100);
    assert_eq!(
        run_cleanup_once(store, Some(capture), Arc::clone(&service), 100, 3_600).await,
        CleanupOutcome::Failed
    );
    assert_eq!(service.pump_sync().await, 1);
}

#[tokio::test]
async fn ttl_cleanup_cascades_only_the_artifact_files_selected_by_the_store() {
    let (root, store, capture, service) = setup("2026-08-03T12:00:00Z", 3_600, 100);
    insert_terminal_summary(&store, "old-request", "2026-08-03T10:00:00Z");
    capture
        .write_artifact(
            "old-artifact",
            "old-request",
            "request_body",
            "2026-08-03T10:00:00Z",
            b"already-redacted",
            Some("text/plain"),
            1,
            true,
            false,
            4_096,
            8_192,
        )
        .expect("write artifact");

    assert_eq!(
        run_cleanup_once(store, Some(capture), service, 100, 3_600).await,
        CleanupOutcome::Completed { deleted_count: 3 }
    );
    assert!(
        !root
            .path()
            .join("artifacts/old-request/old-artifact")
            .exists()
    );
}

#[tokio::test(start_paused = true)]
async fn cleanup_worker_shutdown_joins_without_waiting_for_its_cadence() {
    let (_root, store, capture, service) = setup("2026-08-03T12:00:00Z", 3_600, 100);
    let worker = CleanupWorker::start(
        store,
        Some(capture),
        service,
        100,
        3_600,
        Duration::from_secs(3_600),
        Arc::new(Mutex::new(CleanupWorkerStatus::default())),
    );
    assert!(worker.shutdown().await);
    assert!(!worker.shutdown().await);
}

#[tokio::test]
async fn stalled_cleanup_shutdown_is_bounded_truthful_and_cannot_mutate_after_retirement() {
    let (root, store, capture, service) = setup("2026-08-03T12:00:00Z", 3_600, 100);
    insert_terminal_summary(&store, "expired-after-retire", "2026-08-03T10:00:00Z");
    capture
        .write_artifact(
            "retire-artifact",
            "expired-after-retire",
            "request_body",
            "2026-08-03T10:00:00Z",
            b"already-redacted",
            Some("text/plain"),
            1,
            true,
            false,
            4_096,
            8_192,
        )
        .expect("write artifact");
    let status = Arc::new(Mutex::new(CleanupWorkerStatus::default()));
    let stall = Arc::new(CleanupStallHook::new());
    let worker = CleanupWorker::start_with_test_stall(
        Arc::clone(&store),
        Some(capture),
        Arc::clone(&service),
        100,
        3_600,
        Duration::from_secs(3_600),
        Arc::clone(&status),
        Some(Arc::clone(&stall)),
    )
    .with_shutdown_timeout_for_test(Duration::ZERO);

    stall.wait_until_entered().await;
    assert!(
        !worker.shutdown().await,
        "the fixed bound must return even when cleanup is interrupt-resistant"
    );
    assert_eq!(
        *status.lock().expect("cleanup status"),
        CleanupWorkerStatus {
            state: CleanupWorkerState::TimedOut,
            last_outcome: None,
            shutdown_timeouts: 1,
        }
    );
    assert!(
        !worker.shutdown().await,
        "a still-owned worker remains unavailable to replacement"
    );
    assert_eq!(
        status.lock().expect("cleanup status").shutdown_timeouts,
        1,
        "repeated polling must not inflate the bounded timeout accounting"
    );
    assert!(
        store
            .get_summary("expired-after-retire")
            .expect("summary query")
            .is_some(),
        "the stalled pre-mutation phase may not delete durable rows"
    );
    assert!(
        root.path()
            .join("artifacts/expired-after-retire/retire-artifact")
            .exists(),
        "the stalled phase may not delete artifacts"
    );
    assert_eq!(
        service.pump_sync().await,
        0,
        "a retired cleanup may not enqueue a late audit"
    );

    // The retired task can now observe cancellation, exit without crossing
    // its mutation gate, and only then become truthfully stopped.
    stall.release();
    assert!(
        worker.shutdown().await,
        "released task joins after cancellation"
    );
    assert_eq!(
        *status.lock().expect("cleanup status"),
        CleanupWorkerStatus {
            state: CleanupWorkerState::Stopped,
            last_outcome: None,
            shutdown_timeouts: 1,
        }
    );
    assert!(store.get_summary("expired-after-retire").unwrap().is_some());
    assert!(
        root.path()
            .join("artifacts/expired-after-retire/retire-artifact")
            .exists()
    );
    assert_eq!(service.pump_sync().await, 0);
}

#[tokio::test]
async fn shutdown_interrupts_locked_retention_without_late_file_or_audit_mutation() {
    let (root, store, capture, service) = setup("2026-08-03T12:00:00Z", 3_600, 100);
    insert_terminal_summary(&store, "expired-during-stop", "2026-08-03T10:00:00Z");
    capture
        .write_artifact(
            "expired-artifact",
            "expired-during-stop",
            "request_body",
            "2026-08-03T10:00:00Z",
            b"already-redacted",
            Some("text/plain"),
            1,
            true,
            false,
            4_096,
            8_192,
        )
        .expect("write artifact");

    // Open the cleanup-owned connection before locking the primary. This
    // focuses the test on a blocked retention transaction rather than a
    // connection-initialization lock race.
    let cleanup_store = Arc::new(
        store
            .reopen_for_background_worker()
            .expect("open dedicated cleanup store"),
    );
    let cancellation = Arc::new(CleanupCancellation::new(cleanup_store));

    // Hold the primary connection's write lease on a dedicated thread.
    // The async test never holds that synchronous guard across an await.
    let (locked_tx, locked_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let lock_store = Arc::clone(&store);
    let lock_holder = std::thread::spawn(move || {
        let primary = lock_store.conn();
        primary
            .execute_batch("BEGIN IMMEDIATE")
            .expect("hold deterministic sqlite write lock");
        locked_tx.send(()).expect("announce sqlite lock");
        release_rx.recv().expect("release sqlite lock");
        primary
            .execute_batch("ROLLBACK")
            .expect("release sqlite lock");
    });
    locked_rx.recv().expect("wait for sqlite lock");
    let cleanup = tokio::spawn(run_cleanup_once_with_cancellation(
        Arc::clone(&cancellation),
        Some(capture),
        Arc::clone(&service),
        100,
        3_600,
    ));
    // Give the spawned scheduler its deterministic executor turn; no wall
    // clock delay is used. The held write transaction makes its retention
    // path a SQLite busy wait until shutdown interrupts that connection.
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;

    cancellation.cancel();
    let outcome = tokio::time::timeout(Duration::from_secs(1), cleanup)
        .await
        .expect("sqlite interrupt bounds shutdown");
    assert_eq!(
        outcome.expect("cleanup task joins"),
        CleanupOutcome::SkippedUnavailable
    );
    release_tx.send(()).expect("release sqlite lock");
    lock_holder.join().expect("sqlite lock holder joins");

    assert!(
        store
            .get_summary("expired-during-stop")
            .expect("summary query")
            .is_some(),
        "a retired worker may not commit a late retention delete"
    );
    assert!(
        root.path()
            .join("artifacts")
            .join("expired-during-stop")
            .join("expired-artifact")
            .exists()
    );
    assert_eq!(
        service.pump_sync().await,
        0,
        "no cleanup audit after cancellation"
    );
}

#[test]
fn cancellation_between_retention_phases_preserves_maintenance_receipts() {
    struct Active;
    impl mesh_llm_log_store::MaintenanceExecutionControl for Active {
        fn is_cancelled(&self) -> bool {
            false
        }
    }

    let (_root, store, _capture, _service) = setup("2026-08-03T12:00:00Z", 3_600, 100);
    insert_terminal_summary(&store, "old-maintenance-target", "2026-08-03T10:00:00Z");
    let request = mesh_llm_log_store::CleanupPreviewRequest {
        operation_id: mesh_llm_log_store::MaintenanceOperationId::new(uuid::Uuid::from_u128(901)),
        scope: mesh_llm_log_store::CleanupScope::new(
            mesh_llm_log_store::MaintenanceTimestamp::try_from("2026-08-03T11:00:00Z")
                .expect("cutoff"),
            1,
        )
        .expect("scope"),
        reason: mesh_llm_log_store::MaintenanceReason::try_from("retention boundary")
            .expect("reason"),
    };
    store
        .preview_cleanup(&request, &Active)
        .expect("preview receipt");
    let cancellation = CleanupCancellation::new(Arc::clone(&store));
    cancellation.cancel();

    assert!(
        cleanup_maintenance_receipts_if_active(&cancellation, &store, "2999-01-01T00:00:00Z", 0,)
            .is_err()
    );
    assert_eq!(
        store
            .cleanup_receipt(request.operation_id, &request.reason)
            .expect("cancelled worker preserves receipt")
            .state,
        mesh_llm_log_store::MaintenanceReceiptState::Previewed,
        "cancelled worker must not start the second destructive transaction"
    );
}

#[tokio::test]
async fn cleanup_worker_runs_startup_catch_up_before_its_first_cadence() {
    let (_root, store, capture, service) = setup("2026-08-03T12:00:00Z", 3_600, 100);
    insert_terminal_summary(&store, "expired-before-restart", "2026-08-03T10:00:00Z");
    let status = Arc::new(Mutex::new(CleanupWorkerStatus::default()));

    let worker = CleanupWorker::start(
        Arc::clone(&store),
        Some(capture),
        service,
        100,
        3_600,
        Duration::from_secs(24 * 60 * 60),
        Arc::clone(&status),
    );

    assert_eq!(
        worker.wait_for_startup().await,
        CleanupOutcome::Completed { deleted_count: 2 }
    );
    assert!(
        store
            .get_summary("expired-before-restart")
            .expect("get summary")
            .is_none()
    );
    assert!(worker.shutdown().await);
    assert_eq!(
        *status.lock().expect("cleanup status mutex"),
        CleanupWorkerStatus {
            state: CleanupWorkerState::Stopped,
            last_outcome: Some(CleanupOutcome::Completed { deleted_count: 2 }),
            shutdown_timeouts: 0,
        }
    );
}

#[tokio::test]
async fn cleanup_keeps_active_summary_and_its_old_artifact_reference_intact() {
    let (root, store, capture, service) = setup("2026-08-03T12:00:00Z", 3_600, 100);
    insert_summary(&store, "active-request", "2026-08-03T10:00:00Z");
    capture
        .write_artifact(
            "active-artifact",
            "active-request",
            "request_body",
            "2026-08-03T10:00:00Z",
            b"already-redacted",
            Some("text/plain"),
            1,
            true,
            false,
            4_096,
            8_192,
        )
        .expect("write artifact");

    assert_eq!(
        run_cleanup_once(Arc::clone(&store), Some(capture), service, 100, 3_600).await,
        CleanupOutcome::Completed { deleted_count: 0 }
    );
    assert!(store.get_summary("active-request").unwrap().is_some());
    assert!(
        store
            .get_artifact_pointer("active-artifact")
            .unwrap()
            .is_some()
    );
    assert!(
        root.path()
            .join("artifacts/active-request/active-artifact")
            .exists()
    );
}

#[tokio::test]
async fn ttl_cleanup_prunes_standalone_audit_webhook_and_cleanup_receipts() {
    let (_root, store, capture, service) = setup("2026-08-03T12:00:00Z", 3_600, 100);
    let old = "2026-08-03T10:00:00Z";
    let current = "2026-08-03T12:00:00Z";
    store
        .insert_audit_entry("old-audit", None, old, "operator", "old_action", None)
        .unwrap();
    store
        .insert_webhook_delivery("old-webhook", None, old, 1, None)
        .unwrap();
    store
        .insert_cleanup_run("old-run", old, "old", old, 0, None)
        .unwrap();
    store
        .insert_audit_entry(
            "fresh-audit",
            None,
            current,
            "operator",
            "fresh_action",
            None,
        )
        .unwrap();

    assert_eq!(
        run_cleanup_once(Arc::clone(&store), Some(capture), service, 100, 3_600).await,
        CleanupOutcome::Completed { deleted_count: 3 }
    );
    assert_eq!(
        store
            .conn()
            .query_row("SELECT COUNT(*) FROM webhook_deliveries", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        0
    );
    assert_eq!(
        store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM audit_entries WHERE action = 'fresh_action'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
}

#[test]
fn ttl_cutoff_is_stable_and_path_free() {
    assert_eq!(
        ttl_cutoff("2026-08-03T12:00:00.500Z", 3_600).expect("cutoff"),
        "2026-08-03T11:00:00.500000000Z"
    );
    assert!(
        ttl_cutoff("/private/secret", 3_600)
            .expect_err("invalid clock")
            .contains("invalid timestamp")
    );
}

#[tokio::test]
async fn retention_pass_applies_max_rows_after_ttl_and_records_each_policy() {
    let (_root, store, capture, service) = setup("2026-08-03T12:00:00Z", 172_800, 2);
    for (request_id, terminal_at) in [
        ("oldest", "2026-08-03T10:00:00Z"),
        ("middle", "2026-08-03T11:00:00Z"),
        ("newest", "2026-08-03T11:30:00Z"),
    ] {
        insert_summary(&store, request_id, terminal_at);
        store
            .write_terminal_event(
                request_id,
                &format!("event-{request_id}"),
                r#"{"type":"completed"}"#,
                "completed",
                None,
                terminal_at,
            )
            .expect("write terminal");
    }
    insert_summary(&store, "active", "2026-08-03T09:00:00Z");

    assert_eq!(
        run_cleanup_once(
            Arc::clone(&store),
            Some(capture),
            Arc::clone(&service),
            2,
            3_600
        )
        .await,
        CleanupOutcome::Completed { deleted_count: 2 }
    );
    assert!(store.get_summary("oldest").unwrap().is_none());
    assert!(store.get_summary("middle").unwrap().is_some());
    assert!(store.get_summary("newest").unwrap().is_some());
    assert!(store.get_summary("active").unwrap().is_some());
    let policies: Vec<String> = store
        .conn()
        .prepare("SELECT policy_name FROM cleanup_runs ORDER BY rowid ASC")
        .expect("prepare cleanup policies")
        .query_map([], |row| row.get(0))
        .expect("read cleanup policies")
        .collect::<Result<_, _>>()
        .expect("collect cleanup policies");
    assert_eq!(policies.len(), 2, "receipt cap applies after recording");
    assert!(
        policies
            .iter()
            .all(|policy| { policy.starts_with("ttl:") || policy.starts_with("max_rows:") })
    );
    assert_eq!(service.pump_sync().await, 1);
}

#[tokio::test]
async fn invalid_max_row_policy_fails_open_and_records_a_bounded_audit() {
    let (_root, store, capture, service) = setup("2026-08-03T12:00:00Z", 3_600, 1);
    assert_eq!(
        run_cleanup_once(store, Some(capture), Arc::clone(&service), 0, 3_600).await,
        CleanupOutcome::Failed
    );
    assert_eq!(service.pump_sync().await, 1);
}
