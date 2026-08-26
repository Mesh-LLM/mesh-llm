use super::*;

#[tokio::test]
async fn disabled_webhook_config_starts_no_delivery_scheduler() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = LoggingRuntimeState::initialize(&foundation, &Default::default());

    let service = state
        .start_persistence_worker()
        .await
        .expect("logging service starts");

    assert!(!state.has_webhook_delivery_worker_for_test());
    state.shutdown_cleanup_worker().await;
    assert!(service.shutdown().await);
}

#[tokio::test]
async fn enabled_webhook_config_starts_and_retires_one_delivery_scheduler() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let config = mesh_llm_config::LoggingConfig {
        webhook: mesh_llm_config::LoggingWebhookConfig {
            enabled: true,
            url: Some("http://127.0.0.1:9444/webhook".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let state = LoggingRuntimeState::initialize(&foundation, &config);

    state
        .start_persistence_worker()
        .await
        .expect("logging service starts");
    assert!(state.has_webhook_delivery_worker_for_test());

    assert!(state.retire_and_shutdown().await);
    assert!(!state.has_webhook_delivery_worker_for_test());
}

#[tokio::test]
async fn awaited_runtime_startup_cleanup_uses_injected_store_time_before_ready() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let config = mesh_llm_config::LoggingConfig {
        retention_ttl_secs: 3_600,
        cleanup_cadence_secs: 86_400,
        ..Default::default()
    };
    let state = LoggingRuntimeState::initialize_with_store_clock_for_test(
        &foundation,
        &config,
        Arc::new(FixedStoreClock("2026-08-03T12:00:00Z")),
    );
    let store = state.store().expect("metadata store available");
    store
        .insert_summary(
            "expired-before-startup",
            None,
            None,
            None,
            None,
            "2026-08-03T10:00:00Z",
            None,
            None,
            None,
        )
        .expect("insert stale summary");
    store
        .insert_summary(
            "retained-after-startup",
            None,
            None,
            None,
            None,
            "2026-08-03T11:30:01Z",
            None,
            None,
            None,
        )
        .expect("insert retained summary");
    for (request_id, terminal_at) in [
        ("expired-before-startup", "2026-08-03T10:00:00Z"),
        ("retained-after-startup", "2026-08-03T11:30:01Z"),
    ] {
        store
            .write_terminal_event(
                request_id,
                &format!("terminal-{request_id}"),
                r#"{"type":"completed"}"#,
                "completed",
                None,
                terminal_at,
            )
            .expect("write deterministic terminal record");
    }

    let service = state
        .start_persistence_worker()
        .await
        .expect("service becomes ready after cleanup outcome");

    assert!(
        store
            .get_summary("expired-before-startup")
            .expect("load stale summary")
            .is_none(),
        "startup cleanup completed before the ready service was returned"
    );
    assert!(
        store
            .get_summary("retained-after-startup")
            .expect("load retained summary")
            .is_some()
    );
    assert_eq!(state.status().cleanup_last_outcome, Some("completed"));
    assert!(matches!(
        state.status().cleanup_last_deleted_count,
        Some(count) if count >= 1
    ));

    state.shutdown_cleanup_worker().await;
    assert!(service.shutdown().await);
}

#[tokio::test]
async fn concurrent_starts_publish_one_cleanup_scheduler_and_truthful_status() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let config = mesh_llm_config::LoggingConfig {
        cleanup_cadence_secs: 86_400,
        ..Default::default()
    };
    let state = Arc::new(LoggingRuntimeState::initialize(&foundation, &config));
    let barrier = Arc::new(tokio::sync::Barrier::new(3));

    let first_state = Arc::clone(&state);
    let first_barrier = Arc::clone(&barrier);
    let first = tokio::spawn(async move {
        first_barrier.wait().await;
        first_state.start_persistence_worker().await
    });
    let second_state = Arc::clone(&state);
    let second_barrier = Arc::clone(&barrier);
    let second = tokio::spawn(async move {
        second_barrier.wait().await;
        second_state.start_persistence_worker().await
    });

    // Release both calls from the same scheduling boundary. The
    // state-local candidate count is incremented at construction under
    // the activation gate, so it proves no losing task was ever spawned.
    barrier.wait().await;
    let first_service = first
        .await
        .expect("first start task joins")
        .expect("first start returns the ready service");
    let second_service = second
        .await
        .expect("second start task joins")
        .expect("second start returns the ready service");

    assert!(Arc::ptr_eq(&first_service, &second_service));
    assert_eq!(state.cleanup_candidate_count_for_test(), 1);
    assert!(state.has_cleanup_worker_for_test());
    assert_eq!(state.status().cleanup_worker_state, "running");
    assert_eq!(state.status().persistence_worker_state, "running");

    // Shutdown drains the one operational audit produced by the one
    // startup cleanup. A losing scheduler would produce a second audit or
    // leave a second task able to race a later cleanup pass.
    state.shutdown_cleanup_worker().await;
    assert!(first_service.shutdown().await);
    assert!(!state.has_cleanup_worker_for_test());
    assert_eq!(state.status().cleanup_worker_state, "stopped");
    assert_eq!(state.status().persistence_worker_state, "stopped");
    let cleanup_audits: i64 = state
        .store()
        .expect("metadata store available")
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'logging_cleanup_completed'",
            [],
            |row| row.get(0),
        )
        .expect("count startup cleanup audits");
    assert_eq!(cleanup_audits, 1);
}

#[tokio::test]
async fn retirement_after_cleanup_candidate_publication_leaves_no_worker_on_displaced_state() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let config = mesh_llm_config::LoggingConfig {
        cleanup_cadence_secs: 86_400,
        ..Default::default()
    };
    let state = Arc::new(LoggingRuntimeState::initialize(&foundation, &config));
    let hook = state.install_cleanup_publish_hook_for_test();
    let displaced_service = Arc::clone(
        state
            .service
            .as_ref()
            .expect("healthy state owns a persistence service"),
    );

    let starting_state = Arc::clone(&state);
    let start = tokio::spawn(async move { starting_state.start_persistence_worker().await });

    // The candidate is already atomically published, but the starter has
    // not yet observed readiness. Retire the displaced state in this
    // exact window and prove retirement cancels and joins that task.
    hook.candidate_created.wait().await;
    state.retire_and_shutdown().await;
    hook.resume_install.wait().await;

    assert!(start.await.expect("start task joins").is_none());
    assert!(state.is_retired());
    assert!(
        !state.has_cleanup_worker_for_test(),
        "a retired state must not retain a cleanup task handle"
    );
    assert_eq!(state.status().cleanup_worker_state, "stopped");
    assert!(!displaced_service.is_startable());
    assert!(!displaced_service.is_spawned());
    assert_eq!(
        state.status().persistence_worker_state,
        "stopped",
        "replacement must have joined the displaced persistence worker"
    );
}

#[test]
fn openai_lifecycle_observer_snapshot_is_absent_when_disabled_or_retired() {
    assert!(
        LoggingRuntimeState::unavailable(LoggingMetadataState::StorageUnavailable)
            .openai_lifecycle_observer()
            .is_none()
    );

    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = LoggingRuntimeState::initialize(&foundation, &Default::default());
    assert!(state.openai_lifecycle_observer().is_some());

    state.retired.store(true, Ordering::Release);
    assert!(state.openai_lifecycle_observer().is_none());
}
