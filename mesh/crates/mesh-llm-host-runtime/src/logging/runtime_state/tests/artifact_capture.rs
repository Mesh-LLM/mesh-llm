use super::*;

#[test]
fn production_ingress_wiring_advertises_capture_only_when_storage_is_usable() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = LoggingRuntimeState::initialize(&foundation, &artifact_config());

    let status = state.status();
    assert!(status.artifact_capture_available);
    assert!(status.artifact_capture_ready);
}

#[tokio::test]
async fn ingress_attachment_captures_redacted_request_and_response_bodies() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = Arc::new(LoggingRuntimeState::initialize(
        &foundation,
        &artifact_config(),
    ));
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let mut attachment = state.openai_ingress_attachment(
        request_id,
        RequestSummaryMetadata::from_openai_ingress_path("/v1/chat/completions"),
    );

    attachment.capture_request_body(
        br#"{"authorization":"Bearer test-secret","model":"safe"}"#,
        Some("application/json"),
    );
    attachment.route_observer().capture_response_body(
        br#"{"id":"chatcmpl-safe","api_key":"test-secret"}"#,
        Some("application/json; charset=utf-8"),
    );
    attachment.terminal(TerminalOutcome::CompletedWithStatus(201));

    // Summary, admission, both artifact commands, and terminal state share one
    // serial persistence owner. Ingress itself performs no SQLite/file work.
    assert_eq!(
        state
            .service
            .as_ref()
            .expect("logging service")
            .pump_sync()
            .await,
        5
    );

    let store = state.store().expect("metadata store");
    let request_key = request_id.as_uuid().to_string();
    let artifact_page = store
        .query_artifacts(
            &request_key,
            &mesh_llm_log_store::PageQuery {
                limit: 10,
                cursor: None,
                sort: mesh_llm_log_store::QuerySort::Ascending,
            },
        )
        .expect("query artifacts");
    assert_eq!(artifact_page.items.len(), 2);
    let request_artifact = artifact_page
        .items
        .iter()
        .find(|artifact| artifact.kind == "request")
        .expect("request artifact")
        .artifact_id
        .clone();
    let response_artifact = artifact_page
        .items
        .iter()
        .find(|artifact| artifact.kind == "response")
        .expect("response artifact")
        .artifact_id
        .clone();
    let artifacts = state.query_facade().expect("query facade");
    let request_content = artifacts
        .read_artifact(&request_artifact)
        .expect("request body");
    let response_content = artifacts
        .read_artifact(&response_artifact)
        .expect("response body");
    let request_text = String::from_utf8(request_content.bytes).expect("utf8 request artifact");
    let response_text = String::from_utf8(response_content.bytes).expect("utf8 response artifact");
    assert!(!request_text.contains("test-secret"));
    assert!(!response_text.contains("test-secret"));
    assert!(response_text.contains("chatcmpl-safe"));
    let admitted_at = store
        .get_summary(&request_key)
        .unwrap()
        .expect("summary")
        .created_at;
    assert!(
        artifact_page
            .items
            .iter()
            .all(|artifact| artifact.occurred_at >= admitted_at),
        "artifact metadata must use the lifecycle clock at capture time"
    );
    assert_eq!(
        artifact_page
            .items
            .iter()
            .find(|artifact| artifact.kind == "request")
            .and_then(|artifact| artifact.media_kind.as_deref()),
        Some("application/json"),
        "the parsed OpenAI JSON boundary supplies this closed semantic media kind"
    );
    assert_eq!(
        artifact_page
            .items
            .iter()
            .find(|artifact| artifact.kind == "response")
            .and_then(|artifact| artifact.media_kind.as_deref()),
        Some("application/json")
    );
    let events = store.list_events_for_summary(&request_key).unwrap();
    assert_eq!(events.len(), 2, "one admitted and one terminal event");
    assert_ne!(events[0].event_id, events[1].event_id);
}

#[tokio::test]
async fn streaming_and_oversized_responses_persist_explicit_unavailable_metadata() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = Arc::new(LoggingRuntimeState::initialize(
        &foundation,
        &artifact_config(),
    ));

    let streaming_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let streaming = state.openai_ingress_attachment(
        streaming_id,
        RequestSummaryMetadata::from_openai_ingress_path("/v1/chat/completions"),
    );
    streaming.route_observer().capture_response_unavailable(
        crate::logging::ArtifactUnavailableReason::StreamingResponseNotAssembled,
    );

    let oversized_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let oversized = state.openai_ingress_attachment(
        oversized_id,
        RequestSummaryMetadata::from_openai_ingress_path("/v1/responses"),
    );
    oversized
        .route_observer()
        .capture_response_body(&vec![b'x'; 4_097], Some("not a valid media type"));

    state
        .service
        .as_ref()
        .expect("logging service")
        .pump_sync()
        .await;
    let store = state.store().expect("metadata store");
    for (request_id, reason) in [
        (streaming_id, "streaming_response_not_assembled"),
        (oversized_id, "capture_content_limit_exceeded"),
    ] {
        let page = store
            .query_artifacts(
                &request_id.as_uuid().to_string(),
                &mesh_llm_log_store::PageQuery {
                    limit: 10,
                    cursor: None,
                    sort: mesh_llm_log_store::QuerySort::Ascending,
                },
            )
            .expect("query unavailable artifact");
        assert_eq!(page.items.len(), 1);
        let artifact = &page.items[0];
        assert_eq!(artifact.unavailable_reason.as_deref(), Some(reason));
        assert_eq!(artifact.bytes, 0);
        assert!(artifact.checksum.is_none());
        assert!(!artifact.missing);
        assert!(!artifact.corrupt);
    }
}

#[test]
fn metadata_only_never_opens_or_writes_artifact_capture() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let opened = Arc::new(AtomicBool::new(false));
    let opened_by_callback = Arc::clone(&opened);
    let state = LoggingRuntimeState::initialize_with_capture_opener_for_test(
        &foundation,
        &mesh_llm_config::LoggingConfig::default(),
        move |_, _, _| {
            opened_by_callback.store(true, Ordering::SeqCst);
            Err(LogStoreError::PrivacyNotGuaranteed)
        },
    );

    assert!(!opened.load(Ordering::SeqCst));
    let outcome = state
        .write_artifact(ArtifactCaptureRequest {
            artifact_id: "metadata-only",
            request_id: "request",
            kind: "body",
            occurred_at: "2025-01-01T00:00:00Z",
            content: b"must not be stored",
            media_kind: Some("text/plain"),
            version: 1,
            truncated: false,
        })
        .expect("metadata-only capture is disabled");
    assert!(matches!(outcome, ArtifactCaptureOutcome::Disabled(_)));
    assert_eq!(
        std::fs::read_dir(foundation.artifact_dir())
            .expect("artifact directory")
            .count(),
        0
    );
}

#[test]
fn metadata_only_query_facade_deletes_terminal_metadata_without_artifact_capture() {
    struct NeverCancelled;

    impl MaintenanceExecutionControl for NeverCancelled {
        fn is_cancelled(&self) -> bool {
            false
        }
    }

    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = LoggingRuntimeState::initialize(&foundation, &Default::default());
    let store = state.store().expect("metadata store");
    let request_id = "00000000-0000-4000-8000-000000000143";
    store
        .insert_summary(
            request_id,
            Some("model"),
            Some("route"),
            None,
            None,
            "2025-01-01T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("summary");
    store
        .conn()
        .execute(
            "UPDATE summaries SET state = 'completed' WHERE request_id = ?1",
            [request_id],
        )
        .expect("terminal summary");
    let request = mesh_llm_log_store::DeleteOneRequest::new(
        mesh_llm_log_store::MaintenanceOperationId::new(uuid::Uuid::from_u128(143)),
        request_id,
        mesh_llm_log_store::MaintenanceReason::try_from("operator delete").expect("reason"),
    )
    .expect("delete request");
    let facade = state.query_facade().expect("query facade");

    facade
        .prepare_delete_request(&request, &NeverCancelled)
        .expect("metadata-only preparation");
    let receipt = facade
        .execute_prepared_delete_request(&request, &NeverCancelled)
        .expect("metadata-only execution");

    assert_eq!(
        receipt.state,
        mesh_llm_log_store::MaintenanceReceiptState::Completed
    );
    assert!(store.query_request(request_id).unwrap().is_none());
}

#[test]
fn metadata_only_query_facade_rejects_artifact_pointers_without_preparing_delete() {
    struct NeverCancelled;

    impl MaintenanceExecutionControl for NeverCancelled {
        fn is_cancelled(&self) -> bool {
            false
        }
    }

    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = LoggingRuntimeState::initialize(&foundation, &Default::default());
    let store = state.store().expect("metadata store");
    let request_id = "00000000-0000-4000-8000-000000000144";
    store
        .insert_summary(
            request_id,
            Some("model"),
            Some("route"),
            None,
            None,
            "2025-01-01T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("summary");
    store
        .conn()
        .execute(
            "UPDATE summaries SET state = 'completed' WHERE request_id = ?1",
            [request_id],
        )
        .expect("terminal summary");
    store
        .insert_unavailable_artifact_pointer(mesh_llm_log_store::UnavailableArtifactPointer {
            artifact_id: "00000000-0000-4000-8000-000000000244",
            request_id,
            occurred_at: "2025-01-01T00:00:01Z",
            kind: "response",
            media_kind: Some("text/plain"),
            version: 1,
            reason: "artifact_capture_disabled",
        })
        .expect("artifact pointer");
    let request = mesh_llm_log_store::DeleteOneRequest::new(
        mesh_llm_log_store::MaintenanceOperationId::new(uuid::Uuid::from_u128(144)),
        request_id,
        mesh_llm_log_store::MaintenanceReason::try_from("operator delete").expect("reason"),
    )
    .expect("delete request");

    assert!(matches!(
        state
            .query_facade()
            .expect("query facade")
            .prepare_delete_request(&request, &NeverCancelled),
        Err(LogStoreError::ArtifactDeletionUnavailable)
    ));
    assert!(store.query_request(request_id).unwrap().is_some());
    assert!(store.delete_one_receipt(&request).unwrap().is_none());
}

#[test]
fn privacy_failure_disables_only_artifacts_and_records_one_sanitized_marker() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = LoggingRuntimeState::initialize_with_capture_opener_for_test(
        &foundation,
        &artifact_config(),
        |artifact_root, clock, store| {
            FailOpenArtifactCapture::open_with_privacy(
                artifact_root,
                clock,
                store,
                canonical_artifact_redactor(),
                Arc::new(RejectPrivacy),
            )
        },
    );

    assert_eq!(
        state.health(),
        LoggingRuntimeHealth {
            metadata_available: true,
            artifact_capture_available: false,
            artifact_capture_ready: false,
            artifact_capture_degradation: Some(ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE,),
        }
    );
    let store = state.store().expect("metadata store remains available");
    assert_eq!(marker_audit_count(&store), 1);

    store
        .insert_summary(
            "metadata-request",
            None,
            None,
            None,
            None,
            &store.now(),
            None,
            None,
            None,
        )
        .expect("metadata summary insert");
    store
        .insert_audit_entry(
            "metadata-audit",
            Some("metadata-request"),
            &store.now(),
            "test",
            "metadata_still_available",
            None,
        )
        .expect("metadata audit insert");

    let outcome = state
        .write_artifact(ArtifactCaptureRequest {
            artifact_id: "artifact-after-disable",
            request_id: "metadata-request",
            kind: "request_body",
            occurred_at: &store.now(),
            content: b"redacted",
            media_kind: Some("text/plain"),
            version: 1,
            truncated: false,
        })
        .expect("disabled capture is fail-open");
    assert!(matches!(outcome, ArtifactCaptureOutcome::Disabled(_)));
    let repeated_outcome = state
        .write_artifact(ArtifactCaptureRequest {
            artifact_id: "artifact-after-disable-again",
            request_id: "metadata-request",
            kind: "request_body",
            occurred_at: &store.now(),
            content: b"redacted",
            media_kind: Some("text/plain"),
            version: 1,
            truncated: false,
        })
        .expect("repeated disabled capture is fail-open");
    assert!(matches!(
        repeated_outcome,
        ArtifactCaptureOutcome::Disabled(_)
    ));
    assert_eq!(marker_audit_count(&store), 1);
    assert!(store.get_summary("metadata-request").unwrap().is_some());
    let total_audits: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM audit_entries", [], |row| row.get(0))
        .expect("count all audits");
    assert_eq!(total_audits, 2);
}

#[test]
fn write_time_privacy_failure_publishes_one_marker_and_keeps_metadata_available() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = LoggingRuntimeState::initialize_with_capture_opener_for_test(
        &foundation,
        &artifact_config(),
        |artifact_root, clock, store| {
            FailOpenArtifactCapture::open_with_privacy(
                artifact_root,
                clock,
                store,
                canonical_artifact_redactor(),
                Arc::new(RejectArtifactFiles),
            )
        },
    );
    let store = state.store().expect("metadata store available");
    store
        .insert_summary(
            "write-time-request",
            None,
            None,
            None,
            None,
            &store.now(),
            None,
            None,
            None,
        )
        .expect("metadata summary insert");

    let occurred_at = store.now();
    let write = |artifact_id| ArtifactCaptureRequest {
        artifact_id,
        request_id: "write-time-request",
        kind: "request_body",
        occurred_at: &occurred_at,
        content: b"redacted",
        media_kind: Some("text/plain"),
        version: 1,
        truncated: false,
    };
    assert!(matches!(
        state.write_artifact(write("write-time-artifact")).unwrap(),
        ArtifactCaptureOutcome::Disabled(_)
    ));
    assert!(matches!(
        state
            .write_artifact(write("write-time-artifact-again"))
            .unwrap(),
        ArtifactCaptureOutcome::Disabled(_)
    ));

    assert_eq!(
        state.health().artifact_capture_degradation,
        Some(ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE)
    );
    assert!(state.health().metadata_available);
    assert_eq!(marker_audit_count(&store), 1);
    let marker_actor: String = store
        .conn()
        .query_row(
            "SELECT actor FROM audit_entries WHERE action = ?",
            [ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE],
            |row| row.get(0),
        )
        .expect("query marker audit actor");
    assert_eq!(marker_actor, "logging_service");
}

#[test]
fn raw_artifact_bytes_are_redacted_before_the_capture_boundary() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = LoggingRuntimeState::initialize(&foundation, &artifact_config());
    let store = state.store().expect("metadata store available");
    store
        .insert_summary(
            "raw-artifact-request",
            None,
            None,
            None,
            None,
            &store.now(),
            None,
            None,
            None,
        )
        .expect("metadata summary insert");

    let secret = b"Bearer super-secret-token-0123456789";
    let outcome = state
        .write_artifact(ArtifactCaptureRequest {
            artifact_id: "raw-artifact",
            request_id: "raw-artifact-request",
            kind: "request_body",
            occurred_at: &store.now(),
            content: secret,
            media_kind: Some("text/plain"),
            version: 1,
            truncated: false,
        })
        .expect("artifact write");
    assert!(matches!(
        outcome,
        ArtifactCaptureOutcome::Written(receipt) if receipt.redacted
    ));

    let stored = std::fs::read(
        foundation
            .artifact_dir()
            .join("raw-artifact-request")
            .join("raw-artifact"),
    )
    .expect("read stored artifact");
    assert!(!stored.windows(secret.len()).any(|window| window == secret));
    assert_eq!(stored, b"[REDACTED]");
    assert!(
        store
            .get_artifact_pointer("raw-artifact")
            .expect("artifact pointer")
            .expect("pointer present")
            .redacted
    );
}
