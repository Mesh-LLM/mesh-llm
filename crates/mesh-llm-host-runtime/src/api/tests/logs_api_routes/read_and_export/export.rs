use super::*;

#[tokio::test]
#[serial]
async fn export_is_deterministic_capped_metadata_only_and_audited() {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    })
    .await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let ids = [
        mesh_llm_events::logging::identifiers::RequestId::new(),
        mesh_llm_events::logging::identifiers::RequestId::new(),
        mesh_llm_events::logging::identifiers::RequestId::new(),
    ];
    for (request_id, occurred_at) in ids.iter().zip([
        "2026-08-01T00:00:00Z",
        "2026-08-03T00:00:00Z",
        "2026-08-02T00:00:00Z",
    ]) {
        let request_id = request_id.as_uuid().to_string();
        store
            .insert_summary(
                &request_id,
                Some("safe-model"),
                Some("management"),
                None,
                None,
                occurred_at,
                None,
                None,
                None,
            )
            .unwrap();
    }
    let event = mesh_llm_events::logging::envelope::CanonicalEnvelope::new(
        mesh_llm_events::logging::identifiers::EventId::new(),
        ids[1],
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        1,
        "2026-08-03T00:00:01.000000000Z".to_string(),
        mesh_llm_events::logging::events::LifecycleEvent::Completed {
            status_code: Some(200),
            duration_ms: Some(9),
            usage: Some(mesh_llm_events::logging::events::TokenUsage {
                prompt_tokens: Some(8),
                cached_prompt_tokens: None,
                completion_tokens: Some(3),
                total_tokens: Some(11),
            }),
        },
    );
    store
        .insert_lifecycle_event(
            &ids[1].as_uuid().to_string(),
            &event.event_id.as_uuid().to_string(),
            &serde_json::to_string(&event).unwrap(),
            &event.occurred_at,
        )
        .unwrap();

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let event_response = send_management_request(
        address,
        format!(
            "GET /api/logs/requests/{}/events HTTP/1.1\r\nHost: localhost\r\n\r\n",
            ids[1].as_uuid()
        ),
    )
    .await;
    server.await.unwrap().unwrap();
    assert!(
        event_response.starts_with("HTTP/1.1 200 OK"),
        "{event_response}"
    );
    let event_page = json_body(&event_response);
    assert_eq!(event_page["items"][0]["promptTokens"], 8);
    assert_eq!(event_page["items"][0]["completionTokens"], 3);
    assert_eq!(event_page["items"][0]["totalTokens"], 11);

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let body = r#"{"reason":"operator export"}"#;
    let response = send_management_request(
        address,
        format!(
            "POST /api/logs/requests/export?limit=2 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    server.await.unwrap().unwrap();

    let export = json_body(&response);
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert_eq!(export["items"].as_array().unwrap().len(), 2);
    assert_eq!(
        export["items"][0]["summary"]["requestId"],
        ids[1].as_uuid().to_string()
    );
    assert_eq!(export["items"][0]["events"].as_array().unwrap().len(), 1);
    assert_eq!(export["items"][0]["events"][0]["statusCode"], 200);
    assert_eq!(export["items"][0]["events"][0]["promptTokens"], 8);
    assert_eq!(export["items"][0]["events"][0]["completionTokens"], 3);
    assert_eq!(export["items"][0]["events"][0]["totalTokens"], 11);
    assert_eq!(export["items"][0]["artifacts"].as_array().unwrap().len(), 0);
    assert_eq!(export["artifactContentIncluded"], false);
    assert!(export["nextCursor"].is_string());
    assert!(!response.contains("contentBase64"));

    let (action, detail): (String, String) = store
        .conn()
        .query_row(
            "SELECT action, detail_json FROM audit_entries ORDER BY occurred_at DESC, entry_id DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(action, "log_export");
    assert!(detail.contains("trusted_local_operator"));
    assert!(detail.contains("logs_api"));
    assert!(detail.contains("partial"));

    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

#[tokio::test]
#[serial]
async fn export_never_advances_a_request_cursor_past_partial_child_history() {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    })
    .await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let request_id_text = request_id.as_uuid().to_string();
    store
        .insert_summary(
            &request_id_text,
            Some("safe-model"),
            Some("management"),
            None,
            None,
            "2026-08-03T00:00:00Z",
            None,
            None,
            None,
        )
        .unwrap();
    for sequence in 0..50 {
        let event = mesh_llm_events::logging::envelope::CanonicalEnvelope::new(
            mesh_llm_events::logging::identifiers::EventId::new(),
            request_id,
            mesh_llm_events::logging::replay::ReplayChannel::Requests,
            sequence,
            format!("2026-08-03T00:00:{sequence:02}.000000000Z"),
            mesh_llm_events::logging::events::LifecycleEvent::Admitted {
                model: Some("safe-model".to_string()),
                method: Some("POST".to_string()),
            },
        );
        store
            .insert_lifecycle_event(
                &request_id_text,
                &event.event_id.as_uuid().to_string(),
                &serde_json::to_string(&event).unwrap(),
                &event.occurred_at,
            )
            .unwrap();
    }

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let body = r#"{"reason":"operator export"}"#;
    let response = send_management_request(
        address,
        format!(
            "POST /api/logs/requests/export?limit=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    server.await.unwrap().unwrap();

    let export = json_body(&response);
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert_eq!(export["items"].as_array().unwrap().len(), 1);
    assert_eq!(export["items"][0]["events"].as_array().unwrap().len(), 49);
    assert_eq!(export["items"][0]["childIncomplete"], true);
    assert_eq!(export["retryRequired"], true);
    assert!(export["nextCursor"].is_null());

    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

#[tokio::test]
#[serial]
async fn export_rejects_missing_reason_and_artifact_opt_in_without_capture() {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    })
    .await;
    for body in [
        r#"{}"#,
        r#"{"reason":""}"#,
        r#"{"reason":"copy","includeArtifacts":true}"#,
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!(
                "POST /api/logs/requests/export HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            ),
        )
        .await;
        server.await.unwrap().unwrap();
        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request")
                || response.starts_with("HTTP/1.1 403 Forbidden"),
            "{response}"
        );
        assert!(json_body(&response)["error"]["code"].is_string());
    }
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

#[tokio::test]
#[serial]
async fn export_cursor_and_truncation_follow_actual_page_completeness() {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    })
    .await;
    let store = crate::logging_runtime_state().unwrap().store().unwrap();
    for (index, occurred_at) in [
        "2026-08-01T00:00:00Z",
        "2026-08-02T00:00:00Z",
        "2026-08-03T00:00:00Z",
    ]
    .into_iter()
    .enumerate()
    {
        store
            .insert_summary(
                &format!("00000000-0000-4000-8000-{:012}", 900 + index),
                Some("safe-model"),
                Some("management"),
                None,
                None,
                occurred_at,
                None,
                None,
                None,
            )
            .unwrap();
    }
    let body = r#"{"reason":"operator export"}"#;
    for (limit, expect_len, expect_truncated, expect_cursor) in
        [(3, 3, false, false), (2, 2, true, true)]
    {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!(
                "POST /api/logs/requests/export?limit={limit} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            ),
        )
        .await;
        server.await.unwrap().unwrap();
        let export = json_body(&response);
        assert_eq!(export["items"].as_array().unwrap().len(), expect_len);
        assert_eq!(export["truncated"], expect_truncated);
        assert_eq!(export["nextCursor"].is_string(), expect_cursor);
    }
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

#[tokio::test]
#[serial]
async fn export_includes_redacted_artifact_bytes_only_after_explicit_opt_in() {
    let temporary_directory = tempfile::tempdir().unwrap();
    let mut config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    };
    config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
    crate::initialize_logging_foundation(&config).await;
    let logging = crate::logging_runtime_state().unwrap();
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new()
        .as_uuid()
        .to_string();
    let artifact_id = uuid::Uuid::new_v4().to_string();
    logging
        .store()
        .unwrap()
        .insert_summary(
            &request_id,
            Some("safe-model"),
            Some("management"),
            None,
            None,
            "2026-08-03T00:00:00Z",
            None,
            None,
            None,
        )
        .unwrap();
    logging
        .write_artifact(crate::logging::ArtifactCaptureRequest {
            artifact_id: &artifact_id,
            request_id: &request_id,
            kind: "response",
            occurred_at: "2026-08-03T00:00:01Z",
            content: b"operator-safe export",
            media_kind: Some("text/plain"),
            version: 1,
            truncated: false,
        })
        .unwrap();

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let metadata_body = r#"{"reason":"operator metadata export"}"#;
    let metadata_response = send_management_request(
        address,
        format!(
            "POST /api/logs/requests/export HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{metadata_body}",
            metadata_body.len()
        ),
    )
    .await;
    server.await.unwrap().unwrap();
    let metadata_export = json_body(&metadata_response);
    assert!(metadata_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(metadata_export["artifactContentIncluded"], false);
    assert_eq!(
        metadata_export["items"][0]["artifacts"][0]["artifactId"],
        artifact_id
    );
    assert!(metadata_export["items"][0]["artifacts"][0]["contentBase64"].is_null());

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let body = r#"{"reason":"operator export","includeArtifacts":true}"#;
    let response = send_management_request(
        address,
        format!(
            "POST /api/logs/requests/export HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    server.await.unwrap().unwrap();

    let export = json_body(&response);
    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(export["artifactContentIncluded"], true);
    assert_eq!(
        export["items"][0]["artifacts"][0]["artifactId"],
        artifact_id
    );
    assert!(export["items"][0]["artifacts"][0]["contentBase64"].is_string());

    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}
