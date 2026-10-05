use super::*;

mod export;

#[tokio::test]
#[serial_test::serial]
async fn audit_route_exposes_distinct_positive_sequences_without_private_detail_json() {
    let temporary_directory = install_sse_logging().await;
    let store = crate::logging_runtime_state().unwrap().store().unwrap();
    for (entry_id, occurred_at) in [
        (
            "00000000-0000-4000-8000-000000000091",
            "2026-08-05T00:00:00Z",
        ),
        (
            "00000000-0000-4000-8000-000000000092",
            "2026-08-05T00:00:01Z",
        ),
    ] {
        store
            .insert_audit_entry(
                entry_id,
                None,
                occurred_at,
                "runtime",
                "startup_complete",
                Some(r#"{"severity":"info","secret":"SENTINEL-AUDIT-DETAIL"}"#),
            )
            .unwrap();
    }

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        "GET /api/logs/audit?limit=10 HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    let body = json_body(&response);
    let items = body["items"].as_array().expect("audit items");
    assert_eq!(items.len(), 2);
    let first_sequence = items[0]["sequence"].as_u64().expect("positive sequence");
    let second_sequence = items[1]["sequence"].as_u64().expect("positive sequence");
    assert!(first_sequence > 0);
    assert!(second_sequence > 0);
    assert_ne!(first_sequence, second_sequence);
    assert!(!response.contains("SENTINEL-AUDIT-DETAIL"));
    assert!(items.iter().all(|item| item.get("detailJson").is_none()));
    assert!(items.iter().all(|item| item.get("requestId").is_none()));

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn detail_and_artifact_reads_write_one_metadata_only_success_audit_each() {
    let temporary_directory = tempfile::tempdir().unwrap();
    let mut config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    };
    config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
    crate::initialize_logging_foundation(&config).await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let request_id = "00000000-0000-4000-8000-000000000071";
    let artifact_id = "00000000-0000-4000-8000-000000000072";
    let artifact_body = b"detail-audit-artifact-body";
    seed_terminal_summary(&store, request_id, "2026-08-01T00:00:00Z");
    logging
        .write_artifact(crate::logging::ArtifactCaptureRequest {
            artifact_id,
            request_id,
            kind: "response",
            occurred_at: "2026-08-01T00:00:01Z",
            content: artifact_body,
            media_kind: Some("text/plain"),
            version: 1,
            truncated: false,
        })
        .unwrap();

    for path in [
        format!("/api/logs/requests/{request_id}"),
        format!("/api/logs/artifacts/{artifact_id}"),
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
        )
        .await;
        server.await.unwrap().unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    }

    let audit_count: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM audit_entries", [], |row| row.get(0))
        .unwrap();
    assert_eq!(
        audit_count, 2,
        "one direct audit per read, without recursion"
    );
    for (action, reason) in [
        ("log_request_detail_read", "request detail read"),
        ("log_artifact_read", "artifact read"),
    ] {
        let detail: String = store
            .conn()
            .query_row(
                "SELECT detail_json FROM audit_entries WHERE action = ?1",
                [action],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&detail).unwrap(),
            serde_json::json!({
                "actor": "trusted_local_operator",
                "source": "logs_api",
                "result": "succeeded",
                "reason": reason,
            })
        );
        assert!(!detail.contains(request_id));
        assert!(!detail.contains(artifact_id));
        assert!(!detail.contains(&*String::from_utf8_lossy(artifact_body)));
        assert!(!detail.contains(&*temporary_directory.path().to_string_lossy()));
        assert!(!detail.contains("contentBase64"));
    }

    disable_sse_logging().await;
}

#[tokio::test]
#[serial]
async fn detail_and_artifact_missing_or_unavailable_reads_audit_failures() {
    let temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let missing_request_id = "00000000-0000-4000-8000-000000000073";
    let missing_artifact_id = "00000000-0000-4000-8000-000000000074";
    let unavailable_request_id = "00000000-0000-4000-8000-000000000075";
    let unavailable_artifact_id = "00000000-0000-4000-8000-000000000076";
    seed_terminal_summary(&store, unavailable_request_id, "2026-08-01T00:00:00Z");
    store
        .insert_artifact_pointer(
            unavailable_artifact_id,
            unavailable_request_id,
            "2026-08-01T00:00:01Z",
            "response",
            None,
        )
        .unwrap();
    store
        .update_artifact_pointer_storage(
            unavailable_artifact_id,
            Some("text/plain"),
            "unavailable-checksum",
            1,
            1,
            false,
            false,
        )
        .unwrap();

    for (path, status, code) in [
        (
            format!("/api/logs/requests/{missing_request_id}"),
            "HTTP/1.1 404 Not Found",
            Some("not_found"),
        ),
        (
            format!("/api/logs/artifacts/{missing_artifact_id}"),
            "HTTP/1.1 404 Not Found",
            Some("not_found"),
        ),
        (
            format!("/api/logs/artifacts/{unavailable_artifact_id}"),
            "HTTP/1.1 200 OK",
            None,
        ),
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
        )
        .await;
        server.await.unwrap().unwrap();
        assert!(response.starts_with(status), "{response}");
        if let Some(code) = code {
            assert_eq!(json_body(&response)["error"]["code"], code);
        } else {
            assert_eq!(json_body(&response)["contentState"], "unavailable");
        }
    }

    let detail_failures: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_request_detail_read' AND detail_json LIKE '%\"result\":\"failed\"%'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let artifact_failures: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_artifact_read' AND detail_json LIKE '%\"result\":\"failed\"%'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(detail_failures, 1);
    assert_eq!(artifact_failures, 2);

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn detail_read_serves_normally_when_its_audit_write_fails_without_recursion() {
    let temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let request_id = "00000000-0000-4000-8000-000000000077";
    seed_terminal_summary(&store, request_id, "2026-08-01T00:00:00Z");
    store
        .conn()
        .execute_batch(
            "CREATE TRIGGER reject_detail_read_audit \
             BEFORE INSERT ON audit_entries \
             WHEN NEW.action = 'log_request_detail_read' \
             BEGIN SELECT RAISE(ABORT, 'audit write rejected'); END;",
        )
        .unwrap();

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        format!("GET /api/logs/requests/{request_id} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
    )
    .await;
    server.await.unwrap().unwrap();
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    let audit_count: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM audit_entries", [], |row| row.get(0))
        .unwrap();
    assert_eq!(audit_count, 0, "failed audit must not recursively amplify");

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn cleanup_preview_and_run_share_receipt_and_cascade_only_selected_artifacts() {
    let temporary_directory = tempfile::tempdir().unwrap();
    let mut config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    };
    config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
    crate::initialize_logging_foundation(&config).await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let selected_request = "00000000-0000-4000-8000-000000000011";
    let retained_request = "00000000-0000-4000-8000-000000000012";
    let selected_artifact = "00000000-0000-4000-8000-000000000021";
    let retained_artifact = "00000000-0000-4000-8000-000000000022";
    seed_terminal_summary(&store, selected_request, "2026-08-01T00:00:00Z");
    seed_terminal_summary(&store, retained_request, "2026-08-02T00:00:00Z");
    store
        .conn()
        .execute(
            "UPDATE summaries SET route = 'cleanup-route', model = 'cleanup-model', provider = 'mesh', engine = 'skippy' WHERE request_id = ?1",
            [selected_request],
        )
        .unwrap();
    store
        .conn()
        .execute(
            "UPDATE summaries SET route = 'retained-route', model = 'retained-model', provider = 'other', engine = 'other' WHERE request_id = ?1",
            [retained_request],
        )
        .unwrap();
    for (request_id, artifact_id, occurred_at) in [
        (selected_request, selected_artifact, "2026-08-01T00:00:01Z"),
        (retained_request, retained_artifact, "2026-08-02T00:00:01Z"),
    ] {
        logging
            .write_artifact(crate::logging::ArtifactCaptureRequest {
                artifact_id,
                request_id,
                kind: "response",
                occurred_at,
                content: b"operator-safe cleanup",
                media_kind: Some("text/plain"),
                version: 1,
                truncated: false,
            })
            .unwrap();
    }

    let operation_id = uuid::Uuid::new_v4();
    let preview_body = serde_json::json!({
        "operationId": operation_id,
        "cutoffBefore": "2026-08-03T00:00:00Z",
        "requestLimit": 1,
        "source": "durable",
        "from": "2026-08-01T00:00:00Z",
        "to": "2026-08-02T00:00:00Z",
        "route": "cleanup-route",
        "excludeRoute": "models",
        "model": "cleanup-model",
        "provider": "mesh",
        "engine": "skippy",
        "outcome": "completed",
        "reason": "operator cleanup",
    })
    .to_string();
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let preview_response = send_management_request(
        address,
        cleanup_post("/api/logs/cleanup/preview", &preview_body),
    )
    .await;
    server.await.unwrap().unwrap();
    let preview = json_body(&preview_response);
    assert!(preview_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(preview["state"], "previewed");
    let preview_audit_id = preview["auditId"].as_str().expect("preview audit ID");
    assert_eq!(
        store
            .conn()
            .query_row(
                "SELECT action FROM audit_entries WHERE entry_id = ?1",
                [preview_audit_id],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "log_cleanup_preview"
    );
    assert_eq!(preview["planned"]["requests"], 1);
    assert_eq!(preview["planned"]["artifacts"], 1);
    assert_eq!(preview["hasMore"], false);
    assert_eq!(
        preview["scope"],
        serde_json::json!({
            "source": "durable",
            "cutoffBefore": "2026-08-03T00:00:00.000000000Z",
            "requestLimit": 1,
            "from": "2026-08-01T00:00:00.000000000Z",
            "to": "2026-08-02T00:00:00.000000000Z",
            "route": "cleanup-route",
            "excludeRoute": "models",
            "model": "cleanup-model",
            "provider": "mesh",
            "engine": "skippy",
            "outcome": "completed",
        })
    );
    assert_eq!(
        preview["artifactDeletion"],
        serde_json::json!({ "removed": 0, "failed": 0 })
    );
    assert!(
        preview["selectionFingerprint"]
            .as_str()
            .is_some_and(|value| !value.is_empty())
    );
    assert!(!preview_response.contains(&*temporary_directory.path().to_string_lossy()));

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let preview_replay = send_management_request(
        address,
        cleanup_post("/api/logs/cleanup/preview", &preview_body),
    )
    .await;
    server.await.unwrap().unwrap();
    assert_eq!(json_body(&preview_replay), preview);

    let run_body = serde_json::json!({
        "operationId": operation_id,
        "reason": "operator cleanup",
    })
    .to_string();
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let run_response =
        send_management_request(address, cleanup_post("/api/logs/cleanup/run", &run_body)).await;
    server.await.unwrap().unwrap();
    let run = json_body(&run_response);
    assert!(run_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(run["state"], "completed");
    let run_audit_id = run["auditId"].as_str().expect("run audit ID");
    assert_ne!(run_audit_id, preview_audit_id);
    assert_eq!(
        store
            .conn()
            .query_row(
                "SELECT action FROM audit_entries WHERE entry_id = ?1",
                [run_audit_id],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "log_cleanup_execute"
    );
    assert_eq!(run["operationId"], operation_id.to_string());
    assert_eq!(run["selectionFingerprint"], preview["selectionFingerprint"]);
    assert_eq!(run["planned"], preview["planned"]);
    assert_eq!(run["executed"], preview["planned"]);
    assert_eq!(
        run["artifactDeletion"],
        serde_json::json!({ "removed": 1, "failed": 0 })
    );

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let replay_response =
        send_management_request(address, cleanup_post("/api/logs/cleanup/run", &run_body)).await;
    server.await.unwrap().unwrap();
    assert!(replay_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(json_body(&replay_response), run);
    assert!(store.query_request(selected_request).unwrap().is_none());
    assert!(store.query_artifact(selected_artifact).unwrap().is_none());
    assert!(store.query_request(retained_request).unwrap().is_some());
    assert!(store.query_artifact(retained_artifact).unwrap().is_some());
    let execute_audits: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_cleanup_execute'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(execute_audits, 1);

    disable_sse_logging().await;
}

#[tokio::test]
#[serial]
async fn cleanup_rejects_invalid_scope_and_reason_before_db_and_maps_typed_errors() {
    let temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    for body in [
        r#"{"operationId":"00000000-0000-4000-8000-000000000031","cutoffBefore":"not-a-time","requestLimit":1,"reason":"operator cleanup"}"#,
        r#"{"operationId":"00000000-0000-4000-8000-000000000031","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":101,"reason":"operator cleanup"}"#,
        r#"{"operationId":"00000000-0000-4000-8000-000000000031","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"reason":""}"#,
        r#"{"operationId":"00000000-0000-4000-8000-000000000031","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"model":"/private/model?token=secret","reason":"operator cleanup"}"#,
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response =
            send_management_request(address, cleanup_post("/api/logs/cleanup/preview", body)).await;
        server.await.unwrap().unwrap();
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(json_body(&response)["error"]["code"].is_string());
        assert!(!response.contains("/private/model") && !response.contains("token=secret"));
    }
    let operation_count: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM maintenance_operations", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(operation_count, 0);

    let unknown_run =
        r#"{"operationId":"00000000-0000-4000-8000-000000000032","reason":"operator cleanup"}"#;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response =
        send_management_request(address, cleanup_post("/api/logs/cleanup/run", unknown_run)).await;
    server.await.unwrap().unwrap();
    assert!(response.starts_with("HTTP/1.1 404 Not Found"), "{response}");
    assert_eq!(json_body(&response)["error"]["code"], "not_found");
    let failure_detail: String = store
        .conn()
        .query_row(
            "SELECT detail_json FROM audit_entries WHERE action = 'log_cleanup_run' ORDER BY occurred_at DESC, entry_id DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(failure_detail.contains("failed"));
    assert!(!failure_detail.contains(&*temporary_directory.path().to_string_lossy()));

    let operation_id = "00000000-0000-4000-8000-000000000033";
    let first = format!(
        r#"{{"operationId":"{operation_id}","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"model":"scope-a","reason":"operator cleanup"}}"#
    );
    let changed = format!(
        r#"{{"operationId":"{operation_id}","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"model":"scope-b","reason":"operator cleanup"}}"#
    );
    for (body, expected) in [(&first, "200 OK"), (&changed, "409 Conflict")] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response =
            send_management_request(address, cleanup_post("/api/logs/cleanup/preview", body)).await;
        server.await.unwrap().unwrap();
        assert!(
            response.starts_with(&format!("HTTP/1.1 {expected}")),
            "{response}"
        );
    }

    disable_sse_logging().await;
}

#[tokio::test]
#[serial]
async fn all_registered_log_reads_reach_the_log_dispatcher() {
    let request_id = "00000000-0000-4000-8000-000000000001";
    let paths = vec![
        "/api/logs/requests".to_string(),
        format!("/api/logs/requests/{request_id}"),
        format!("/api/logs/requests/{request_id}/events"),
        format!("/api/logs/requests/{request_id}/artifacts"),
        format!("/api/logs/artifacts/{request_id}"),
        "/api/logs/proxy".to_string(),
    ];
    for path in paths {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
        )
        .await;
        server.await.unwrap().unwrap();
        assert!(
            response.starts_with("HTTP/1.1 200 OK")
                || response.starts_with("HTTP/1.1 404 Not Found")
                || response.starts_with("HTTP/1.1 503 Service Unavailable"),
            "{path}"
        );
        assert!(!response.contains(r#"{"error":"Not found"}"#));
    }
}

#[tokio::test]
#[serial]
async fn successful_log_response_redacts_path_shaped_metadata() {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    })
    .await;
    let logging = crate::logging_runtime_state().unwrap();
    let request_id = "00000000-0000-4000-8000-000000000001";
    logging
        .store()
        .unwrap()
        .insert_summary(
            request_id,
            Some("/Users/operator/private-model.gguf?token=secret"),
            Some("chat"),
            None,
            None,
            "2026-08-01T00:00:00Z",
            None,
            None,
            None,
        )
        .unwrap();

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        "GET /api/logs/requests HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    let body = json_body(&response);
    let item = body["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["requestId"] == request_id))
        .expect("redacted test request is listed");
    assert_eq!(item["model"], "[REDACTED]");
    assert!(!response.contains("/Users/operator") && !response.contains("token=secret"));

    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

#[tokio::test]
#[serial]
async fn malformed_mutating_json_reports_body_not_header_error() {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    })
    .await;
    for (path, body) in [
        ("/api/logs/requests/export", "{"),
        ("/api/logs/cleanup/preview", "{"),
        ("/api/logs/cleanup/run", "{"),
        (
            "/api/logs/webhooks/webhook:00000000-0000-4000-8000-000000000001/retry",
            "{",
        ),
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!(
                "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            ),
        )
        .await;
        server.await.unwrap().unwrap();
        assert_eq!(json_body(&response)["error"]["code"], "invalid_request");
        assert_eq!(
            json_body(&response)["error"]["message"],
            "request body is invalid",
            "{path}: {response}"
        );
    }
    let body = "{";
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        format!(
            "POST /api/logs/requests/00000000-0000-4000-8000-000000000001/delete HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    server.await.unwrap().unwrap();
    assert_eq!(json_body(&response)["error"]["code"], "invalid_request");
    assert_eq!(
        json_body(&response)["error"]["message"],
        "request body is invalid",
        "delete: {response}"
    );
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}
