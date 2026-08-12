use super::*;

struct FixedStoreClock(&'static str);

impl mesh_llm_log_store::Clock for FixedStoreClock {
    fn now(&self) -> String {
        self.0.to_string()
    }
}

fn logging_config(root: &std::path::Path) -> mesh_llm_config::LoggingConfig {
    mesh_llm_config::LoggingConfig {
        application_state_root: Some(root.to_path_buf()),
        retention_ttl_secs: 60 * 60,
        retention_max_rows: 100,
        // A startup pass must perform the cleanup in this test; timer-driven
        // work must not be necessary for readiness.
        cleanup_cadence_secs: 24 * 60 * 60,
        ..Default::default()
    }
}

#[tokio::test]
#[serial_test::serial]
async fn runtime_logging_worker_uses_installed_state_and_stops_cleanly() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        webhook: mesh_llm_config::LoggingWebhookConfig {
            enabled: true,
            url: Some("http://127.0.0.1:9444/webhook".to_string()),
            ..Default::default()
        },
        ..Default::default()
    })
    .await;

    let service = start_run_auto_logging_service()
        .await
        .expect("healthy logging service");
    assert!(service.is_spawned());

    // Starting through the same installed state is idempotent: one service,
    // one worker, including when an embedded entrypoint crosses this boundary.
    let repeated = start_run_auto_logging_service()
        .await
        .expect("installed logging service");
    assert!(Arc::ptr_eq(&service, &repeated));
    assert!(repeated.is_spawned());

    let logging_runtime = crate::logging_runtime_state().expect("installed logging runtime");
    assert!(logging_runtime.has_webhook_delivery_worker_for_test());
    logging_runtime.shutdown_webhook_delivery_worker().await;
    assert!(!logging_runtime.has_webhook_delivery_worker_for_test());
    assert!(service.shutdown().await);
    assert!(!service.is_spawned());
}

#[tokio::test]
#[serial_test::serial]
async fn runtime_logging_worker_is_absent_when_logging_is_disabled() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    })
    .await;

    assert!(start_run_auto_logging_service().await.is_none());
}

/// Exercise the process-installed `LoggingRuntimeState` through the exact
/// `run_auto` persistence start boundary. The startup service is not returned
/// ready until the injected-clock retention pass has completed, and a normal
/// in-process replacement reopens the same durable root without leaving the
/// former worker alive.
#[tokio::test]
#[serial_test::serial]
async fn runtime_logging_startup_cleanup_precedes_readiness_and_reopen_keeps_survivors() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let root = temporary_directory.path().join("logging");
    let config = logging_config(&root);
    let clock: Arc<dyn mesh_llm_log_store::Clock> =
        Arc::new(FixedStoreClock("2026-08-03T12:00:00Z"));

    crate::initialize_logging_foundation_with_store_clock_for_test(&config, Arc::clone(&clock))
        .await;
    let initial_state = crate::logging_runtime_state().expect("installed logging state");
    let initial_store = initial_state.store().expect("healthy metadata store");
    initial_store
        .insert_summary(
            "expired-before-ready",
            None,
            None,
            None,
            None,
            "2026-08-03T10:00:00Z",
            None,
            None,
            None,
        )
        .expect("insert expired summary");
    initial_store
        .write_terminal_event(
            "expired-before-ready",
            "expired-event",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            "2026-08-03T10:00:00Z",
        )
        .expect("make expired summary terminal");
    initial_store
        .insert_summary(
            "retained-across-reopen",
            None,
            None,
            None,
            None,
            "2026-08-03T11:30:00Z",
            None,
            None,
            None,
        )
        .expect("insert retained summary");
    initial_store
        .write_terminal_event(
            "retained-across-reopen",
            "retained-event",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            "2026-08-03T11:30:00Z",
        )
        .expect("make retained summary terminal");

    let service = start_run_auto_logging_service()
        .await
        .expect("startup cleanup completes before run_auto service is ready");
    assert!(service.is_spawned());
    assert!(
        initial_store
            .get_summary("expired-before-ready")
            .expect("read expired summary")
            .is_none(),
        "startup catch-up must delete expired terminal records before readiness"
    );
    assert!(
        initial_store
            .get_summary("retained-across-reopen")
            .expect("read retained summary")
            .is_some()
    );
    assert_eq!(
        initial_state.status().cleanup_last_outcome,
        Some("completed")
    );

    // This is the same replaceable process-local installation boundary used
    // when an embedded host starts after a normal runtime in one process.
    crate::initialize_logging_foundation_with_store_clock_for_test(&config, clock).await;
    assert!(initial_state.is_retired());
    assert!(!service.is_spawned(), "replacement joins the old worker");

    let reopened_state = crate::logging_runtime_state().expect("replacement state");
    let reopened_store = reopened_state.store().expect("reopened metadata store");
    assert!(
        reopened_store
            .get_summary("retained-across-reopen")
            .expect("read reopened summary")
            .is_some(),
        "durable summaries must survive a normal/embedded state replacement"
    );
    let reopened_service = start_run_auto_logging_service()
        .await
        .expect("replacement run_auto service");
    assert!(reopened_service.is_spawned());

    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
    assert!(!reopened_service.is_spawned());
}

#[tokio::test]
#[serial_test::serial]
async fn cli_sourced_audit_records_durably_with_static_code() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let clock: Arc<dyn mesh_llm_log_store::Clock> =
        Arc::new(FixedStoreClock("2026-08-07T12:00:00Z"));
    crate::initialize_logging_foundation_with_store_clock_for_test(
        &mesh_llm_config::LoggingConfig {
            application_state_root: Some(temporary_directory.path().join("logging")),
            ..Default::default()
        },
        clock,
    )
    .await;

    let service = start_run_auto_logging_service()
        .await
        .expect("startable logging service");
    let state = crate::logging_runtime_state().expect("installed logging runtime");

    let record = crate::OperationalAuditRecord::builder("cli", "cli_command_started")
        .severity(crate::OperationalAuditSeverity::Info)
        .build();
    assert!(
        state.write_operational_audit(record),
        "startable logging state must accept the cli record"
    );

    let store = state.store().expect("healthy metadata store");
    let mut durable = Vec::new();
    for _ in 0..50 {
        let page = store
            .list_audit_entries(
                Some(10),
                None,
                mesh_llm_log_store::AuditEntryFilters {
                    source: Some(mesh_llm_log_store::AuditEntrySource::Cli),
                    severity: None,
                    ..mesh_llm_log_store::AuditEntryFilters::default()
                },
            )
            .expect("list cli audit entries");
        durable = page.items;
        if !durable.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    assert_eq!(durable.len(), 1);
    assert_eq!(durable[0].source, "cli");
    assert_eq!(durable[0].code, "cli_command_started");
    assert_eq!(
        durable[0].severity,
        Some(mesh_llm_log_store::AuditEntrySeverity::Info)
    );
    assert!(service.shutdown().await);
}
