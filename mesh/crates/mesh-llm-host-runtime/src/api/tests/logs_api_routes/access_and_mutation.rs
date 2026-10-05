use super::*;

#[tokio::test]
async fn log_routes_reject_hostile_host_and_origin_before_dispatch() {
    for header in [
        "Host: hostile.example\r\n",
        "Host: localhost\r\nOrigin: https://hostile.example\r\n",
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!("GET /api/logs/requests HTTP/1.1\r\n{header}\r\n"),
        )
        .await;
        server.await.unwrap().unwrap();
        assert!(response.starts_with("HTTP/1.1 403 Forbidden"));
        assert!(!response.contains("logging service"));
    }
}

#[tokio::test]
async fn cleanup_routes_are_post_only_and_trusted_local_before_dispatch() {
    let operation_id = uuid::Uuid::new_v4();
    let body = serde_json::json!({
        "operationId": operation_id,
        "cutoffBefore": "2026-08-03T00:00:00Z",
        "requestLimit": 1,
        "reason": "operator cleanup",
    })
    .to_string();
    for path in ["/api/logs/cleanup/preview", "/api/logs/cleanup/run"] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
        )
        .await;
        server.await.unwrap().unwrap();
        assert!(
            response.starts_with("HTTP/1.1 405 Method Not Allowed"),
            "{path}"
        );
        assert_eq!(json_body(&response)["error"]["code"], "method_not_allowed");
    }

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        format!(
            "POST /api/logs/cleanup/preview HTTP/1.1\r\nHost: hostile.example\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    server.await.unwrap().unwrap();
    assert!(response.starts_with("HTTP/1.1 403 Forbidden"));
    assert!(!response.contains("logging service"));
}

#[tokio::test]
#[serial]
async fn export_route_requires_post_and_post_reaches_export_validation() {
    let temporary_directory = install_sse_logging().await;

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let get_response = send_management_request(
        address,
        "GET /api/logs/requests/export HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    server.await.unwrap().unwrap();
    assert!(
        get_response.starts_with("HTTP/1.1 405 Method Not Allowed"),
        "{get_response}"
    );
    assert_eq!(
        json_body(&get_response)["error"]["code"],
        "method_not_allowed"
    );
    assert_eq!(
        json_body(&get_response)["error"]["message"],
        "request export requires POST"
    );

    let body = r#"{}"#;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let post_response = send_management_request(
        address,
        format!(
            "POST /api/logs/requests/export HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    server.await.unwrap().unwrap();
    assert!(
        post_response.starts_with("HTTP/1.1 400 Bad Request"),
        "{post_response}"
    );
    assert_ne!(
        json_body(&post_response)["error"]["code"],
        "method_not_allowed"
    );

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
async fn delete_route_rejects_hostile_callers_and_wrong_methods_before_dispatch() {
    let request_id = "00000000-0000-4000-8000-000000000041";
    let body =
        r#"{"operationId":"00000000-0000-4000-8000-000000000042","reason":"operator delete"}"#;
    for header in [
        "Host: hostile.example\r\n",
        "Host: localhost\r\nOrigin: https://hostile.example\r\n",
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!(
                "POST /api/logs/requests/{request_id}/delete HTTP/1.1\r\n{header}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            ),
        )
        .await;
        server.await.unwrap().unwrap();
        assert!(response.starts_with("HTTP/1.1 403 Forbidden"));
        assert!(!response.contains("logging service"));
    }

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        format!("GET /api/logs/requests/{request_id}/delete HTTP/1.1\r\nHost: localhost\r\n\r\n"),
    )
    .await;
    server.await.unwrap().unwrap();
    assert!(response.starts_with("HTTP/1.1 405 Method Not Allowed"));
    assert_eq!(json_body(&response)["error"]["code"], "method_not_allowed");
}

#[tokio::test]
async fn webhook_retry_route_rejects_hostile_callers_and_wrong_methods_before_dispatch() {
    let delivery_id = "webhook:00000000-0000-4000-8000-000000000080";
    let body = r#"{"reason":"operator webhook retry"}"#;
    for header in [
        "Host: hostile.example\r\n",
        "Host: localhost\r\nOrigin: https://hostile.example\r\n",
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(
            address,
            format!(
                "POST /api/logs/webhooks/{delivery_id}/retry HTTP/1.1\r\n{header}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            ),
        )
        .await;
        server.await.unwrap().unwrap();
        assert!(response.starts_with("HTTP/1.1 403 Forbidden"));
        assert!(!response.contains("logging service"));
    }

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        address,
        format!("GET /api/logs/webhooks/{delivery_id}/retry HTTP/1.1\r\nHost: localhost\r\n\r\n"),
    )
    .await;
    server.await.unwrap().unwrap();
    assert!(response.starts_with("HTTP/1.1 405 Method Not Allowed"));
    assert_eq!(json_body(&response)["error"]["code"], "method_not_allowed");
}

#[tokio::test]
#[serial]
async fn webhook_retry_route_rejects_invalid_input_before_mutation_or_audit() {
    let temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let delivery_id = "webhook:00000000-0000-4000-8000-000000000081";
    store
        .insert_webhook_delivery(delivery_id, None, "2026-08-04T00:00:00Z", 1, None)
        .unwrap();

    for request in [
        webhook_retry_post(&"x".repeat(129), r#"{"reason":"operator webhook retry"}"#),
        webhook_retry_post(delivery_id, r#"{}"#),
        webhook_retry_post(delivery_id, r#"{"reason":""}"#),
        webhook_retry_post(
            delivery_id,
            r#"{"reason":"operator webhook retry","extra":true}"#,
        ),
        format!(
            "POST /api/logs/webhooks/{delivery_id}/retry?unexpected=true HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{{\"reason\":\"operator webhook retry\"}}"
        ),
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(address, request).await;
        server.await.unwrap().unwrap();
        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request"),
            "{response}"
        );
    }
    assert_eq!(
        store.webhook_delivery(delivery_id).unwrap().unwrap().state,
        mesh_llm_log_store::WebhookDeliveryState::DeadLetter
    );
    let audit_count: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM audit_entries", [], |row| row.get(0))
        .unwrap();
    assert_eq!(audit_count, 0);

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn webhook_retry_route_is_idempotent_audited_and_private() {
    let temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let delivery_id = "webhook:00000000-0000-4000-8000-000000000082";
    let private_target = "https://hooks.example/private?credential=webhook-secret";
    let private_body = "webhook-private-response-body";
    store
        .insert_webhook_delivery(delivery_id, None, "2026-08-04T00:00:00Z", 1, None)
        .unwrap();
    store
        .conn()
        .execute(
            "UPDATE webhook_deliveries SET target_url = ?1, response_body = ?2, error_msg = ?3 WHERE delivery_id = ?4",
            [
                private_target,
                private_body,
                "credential=error-secret",
                delivery_id,
            ],
        )
        .unwrap();

    let body = r#"{"reason":"operator webhook retry"}"#;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let first_response =
        send_management_request(address, webhook_retry_post(delivery_id, body)).await;
    server.await.unwrap().unwrap();
    assert!(
        first_response.starts_with("HTTP/1.1 200 OK"),
        "{first_response}"
    );
    assert_eq!(
        json_body(&first_response),
        serde_json::json!({ "outcome": "scheduled" })
    );

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let replay_response =
        send_management_request(address, webhook_retry_post(delivery_id, body)).await;
    server.await.unwrap().unwrap();
    assert!(
        replay_response.starts_with("HTTP/1.1 200 OK"),
        "{replay_response}"
    );
    assert_eq!(
        json_body(&replay_response),
        serde_json::json!({ "outcome": "already_scheduled" })
    );

    let delivery = store.webhook_delivery(delivery_id).unwrap().unwrap();
    assert_eq!(
        delivery.state,
        mesh_llm_log_store::WebhookDeliveryState::ManualRetry
    );
    assert_eq!(delivery.attempt_number, 0);
    for value in [&first_response, &replay_response] {
        assert!(!value.contains(delivery_id));
        assert!(!value.contains(private_target));
        assert!(!value.contains(private_body));
        assert!(!value.contains("credential=error-secret"));
    }
    let audit_details = {
        let connection = store.conn();
        let mut statement = connection
            .prepare(
                "SELECT detail_json FROM audit_entries WHERE action = 'log_webhook_manual_retry' ORDER BY occurred_at, entry_id",
            )
            .unwrap();
        statement
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };
    assert_eq!(audit_details.len(), 2);
    for detail in audit_details {
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&detail).unwrap(),
            serde_json::json!({
                "actor": "trusted_local_operator",
                "source": "logs_api",
                "result": "succeeded",
                "reason": "operator webhook retry",
            })
        );
        assert!(!detail.contains(delivery_id));
        assert!(!detail.contains(private_target));
        assert!(!detail.contains(private_body));
        assert!(!detail.contains("credential=error-secret"));
    }

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn webhook_retry_route_maps_typed_failures_and_audit_write_failures() {
    let temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let succeeded_delivery = "webhook:00000000-0000-4000-8000-000000000083";
    let retry_delivery = "webhook:00000000-0000-4000-8000-000000000084";
    let body = r#"{"reason":"operator webhook retry"}"#;
    store
        .insert_webhook_delivery(
            succeeded_delivery,
            None,
            "2026-08-04T00:00:00Z",
            1,
            Some(204),
        )
        .unwrap();
    for delivery_id in [
        "webhook:00000000-0000-4000-8000-000000000085",
        succeeded_delivery,
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response =
            send_management_request(address, webhook_retry_post(delivery_id, body)).await;
        server.await.unwrap().unwrap();
        if delivery_id == succeeded_delivery {
            assert!(response.starts_with("HTTP/1.1 409 Conflict"), "{response}");
            assert_eq!(
                json_body(&response)["error"]["code"],
                "webhook_not_retryable"
            );
        } else {
            assert!(response.starts_with("HTTP/1.1 404 Not Found"), "{response}");
            assert_eq!(json_body(&response)["error"]["code"], "not_found");
        }
    }
    store
        .insert_webhook_delivery(retry_delivery, None, "2026-08-04T00:00:00Z", 1, None)
        .unwrap();
    store
        .conn()
        .execute_batch(
            "CREATE TRIGGER reject_webhook_retry_audit \
             BEFORE INSERT ON audit_entries \
             WHEN NEW.action = 'log_webhook_manual_retry' \
             BEGIN SELECT RAISE(ABORT, 'audit write rejected'); END;",
        )
        .unwrap();
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(address, webhook_retry_post(retry_delivery, body)).await;
    server.await.unwrap().unwrap();
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert_eq!(
        json_body(&response),
        serde_json::json!({ "outcome": "scheduled" })
    );
    assert_eq!(
        store
            .webhook_delivery(retry_delivery)
            .unwrap()
            .unwrap()
            .state,
        mesh_llm_log_store::WebhookDeliveryState::ManualRetry
    );
    let audit_count: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_webhook_manual_retry'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(audit_count, 2, "failed audit must not recursively amplify");

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn delete_route_rejects_invalid_identifiers_and_missing_reason_before_mutation() {
    let temporary_directory = install_sse_logging().await;
    let store = crate::logging_runtime_state().unwrap().store().unwrap();
    for (request_id, body) in [
        (
            "not-a-uuid",
            r#"{"operationId":"00000000-0000-4000-8000-000000000043","reason":"operator delete"}"#,
        ),
        (
            "00000000-0000-4000-8000-000000000044",
            r#"{"operationId":"00000000-0000-4000-8000-000000000045"}"#,
        ),
        (
            "00000000-0000-4000-8000-000000000046",
            r#"{"operationId":"00000000-0000-4000-8000-000000000047","reason":""}"#,
        ),
    ] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(address, delete_post(request_id, body)).await;
        server.await.unwrap().unwrap();
        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request"),
            "{response}"
        );
    }
    let operation_count: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM maintenance_operations", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(operation_count, 0);
    let summary_count: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM summaries", [], |row| row.get(0))
        .unwrap();
    assert_eq!(summary_count, 0);

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn delete_route_cascades_terminal_artifacts_and_replays_the_receipt() {
    let temporary_directory = tempfile::tempdir().unwrap();
    let mut config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    };
    config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
    crate::initialize_logging_foundation(&config).await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let request_id = "00000000-0000-4000-8000-000000000051";
    let artifact_id = "00000000-0000-4000-8000-000000000052";
    seed_terminal_summary(&store, request_id, "2026-08-01T00:00:00Z");
    logging
        .write_artifact(crate::logging::ArtifactCaptureRequest {
            artifact_id,
            request_id,
            kind: "response",
            occurred_at: "2026-08-01T00:00:01Z",
            content: b"operator delete",
            media_kind: Some("text/plain"),
            version: 1,
            truncated: false,
        })
        .unwrap();
    let body =
        r#"{"operationId":"00000000-0000-4000-8000-000000000053","reason":"operator delete"}"#;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let first_response = send_management_request(address, delete_post(request_id, body)).await;
    server.await.unwrap().unwrap();
    assert!(first_response.starts_with("HTTP/1.1 200 OK"));
    let first = json_body(&first_response);
    assert_eq!(first["requestId"], request_id);
    assert_eq!(first["operationId"], "00000000-0000-4000-8000-000000000053");
    let audit_id = first["auditId"].as_str().expect("delete audit ID");
    assert_eq!(
        store
            .conn()
            .query_row(
                "SELECT action FROM audit_entries WHERE entry_id = ?1",
                [audit_id],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "log_delete_request"
    );
    assert_eq!(first["state"], "completed");
    assert_eq!(first["planned"]["requests"], 1);
    assert_eq!(first["executed"], first["planned"]);
    assert_eq!(
        first["artifactDeletion"],
        serde_json::json!({ "removed": 1, "failed": 0 })
    );
    assert!(!first_response.contains(&*temporary_directory.path().to_string_lossy()));

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let replay_response = send_management_request(address, delete_post(request_id, body)).await;
    server.await.unwrap().unwrap();
    assert!(replay_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(json_body(&replay_response), first);
    assert!(store.query_request(request_id).unwrap().is_none());
    assert!(store.query_artifact(artifact_id).unwrap().is_none());
    assert!(
        !temporary_directory
            .path()
            .join("logging")
            .join("artifacts")
            .join(request_id)
            .join(artifact_id)
            .exists()
    );
    let audit_count: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_delete_request'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(audit_count, 1);

    disable_sse_logging().await;
}

#[tokio::test]
#[serial]
async fn delete_route_uses_database_only_receipt_for_default_metadata_only_logging() {
    let temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().expect("logging runtime");
    let store = logging.store().expect("metadata store");
    let request_id = "00000000-0000-4000-8000-000000000054";
    seed_terminal_summary(&store, request_id, "2026-08-01T00:00:00Z");
    let body =
        r#"{"operationId":"00000000-0000-4000-8000-000000000055","reason":"operator delete"}"#;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(address, delete_post(request_id, body)).await;
    server.await.unwrap().unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    let receipt = json_body(&response);
    assert_eq!(receipt["state"], "completed");
    assert_eq!(receipt["planned"]["artifacts"], 0);
    assert_eq!(receipt["executed"], receipt["planned"]);
    assert!(store.query_request(request_id).unwrap().is_none());

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn metadata_only_delete_route_rejects_request_with_artifact_pointers() {
    let temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().expect("logging runtime");
    let store = logging.store().expect("metadata store");
    let request_id = "00000000-0000-4000-8000-000000000056";
    seed_terminal_summary(&store, request_id, "2026-08-01T00:00:00Z");
    store
        .insert_unavailable_artifact_pointer(mesh_llm_log_store::UnavailableArtifactPointer {
            artifact_id: "00000000-0000-4000-8000-000000000057",
            request_id,
            occurred_at: "2026-08-01T00:00:01Z",
            kind: "response",
            media_kind: Some("text/plain"),
            version: 1,
            reason: "artifact_capture_disabled",
        })
        .expect("artifact pointer");
    let body =
        r#"{"operationId":"00000000-0000-4000-8000-000000000058","reason":"operator delete"}"#;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(address, delete_post(request_id, body)).await;
    server.await.unwrap().unwrap();

    assert!(
        response.starts_with("HTTP/1.1 503 Service Unavailable"),
        "{response}"
    );
    assert_eq!(
        json_body(&response)["error"]["code"],
        "artifact_deletion_unavailable"
    );
    assert!(store.query_request(request_id).unwrap().is_some());
    assert_eq!(
        store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM maintenance_operations WHERE operation_id = ?1",
                ["00000000-0000-4000-8000-000000000058"],
                |row| row.get::<_, i64>(0),
            )
            .expect("no accepted receipt"),
        0
    );

    disable_sse_logging().await;
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn delete_route_maps_missing_active_and_unavailable_to_typed_outcomes() {
    let temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let missing_id = "00000000-0000-4000-8000-000000000061";
    let missing_body =
        r#"{"operationId":"00000000-0000-4000-8000-000000000062","reason":"operator delete"}"#;
    for _ in 0..2 {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response =
            send_management_request(address, delete_post(missing_id, missing_body)).await;
        server.await.unwrap().unwrap();
        assert!(response.starts_with("HTTP/1.1 404 Not Found"));
        assert_eq!(json_body(&response)["error"]["code"], "not_found");
    }
    let missing_operation_count: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM maintenance_operations WHERE operation_id = '00000000-0000-4000-8000-000000000062'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(missing_operation_count, 0);

    let active_id = "00000000-0000-4000-8000-000000000063";
    store
        .insert_summary(
            active_id,
            Some("safe-model"),
            Some("management"),
            None,
            None,
            "2026-08-01T00:00:00Z",
            None,
            None,
            None,
        )
        .unwrap();
    let active_body =
        r#"{"operationId":"00000000-0000-4000-8000-000000000064","reason":"operator delete"}"#;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(address, delete_post(active_id, active_body)).await;
    server.await.unwrap().unwrap();
    assert!(response.starts_with("HTTP/1.1 409 Conflict"));
    assert_eq!(json_body(&response)["error"]["code"], "request_active");
    assert!(store.query_request(active_id).unwrap().is_some());
    let active_operation_count: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM maintenance_operations WHERE operation_id = '00000000-0000-4000-8000-000000000064'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(active_operation_count, 0);

    disable_sse_logging().await;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(address, delete_post(active_id, active_body)).await;
    server.await.unwrap().unwrap();
    assert!(response.starts_with("HTTP/1.1 503 Service Unavailable"));
    assert_eq!(json_body(&response)["error"]["code"], "logging_unavailable");
    drop(temporary_directory);
}

#[tokio::test]
#[serial]
async fn delete_route_resumes_durable_previewed_and_partial_receipts() {
    struct NeverCancelled;
    impl mesh_llm_log_store::MaintenanceExecutionControl for NeverCancelled {
        fn is_cancelled(&self) -> bool {
            false
        }
    }

    let temporary_directory = tempfile::tempdir().unwrap();
    let mut config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        ..Default::default()
    };
    config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
    crate::initialize_logging_foundation(&config).await;
    let logging = crate::logging_runtime_state().unwrap();
    let store = logging.store().unwrap();
    let facade = logging.query_facade().expect("query facade");
    let request_id = "00000000-0000-4000-8000-000000000065";
    let operation_id = "00000000-0000-4000-8000-000000000066";
    let reason =
        mesh_llm_log_store::MaintenanceReason::try_from("operator delete").expect("reason");
    let request = mesh_llm_log_store::DeleteOneRequest::new(
        mesh_llm_log_store::MaintenanceOperationId::new(
            uuid::Uuid::parse_str(operation_id).unwrap(),
        ),
        request_id,
        reason,
    )
    .expect("delete request");
    facade
        .delete_request_cascade(&request, &NeverCancelled)
        .expect("seed completed no-op receipt");

    let body = format!(r#"{{"operationId":"{operation_id}","reason":"operator delete"}}"#);
    for state in ["previewed", "partial"] {
        store
            .conn()
            .execute(
                "UPDATE maintenance_operations SET state = ?2, artifact_files_failed = CASE WHEN ?2 = 'partial' THEN 1 ELSE 0 END, completed_at = NULL WHERE operation_id = ?1",
                [operation_id, state],
            )
            .expect("simulate interrupted durable receipt");

        let api = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(api).await;
        let response = send_management_request(address, delete_post(request_id, &body)).await;
        server.await.unwrap().unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert_eq!(json_body(&response)["state"], "completed");
    }

    disable_sse_logging().await;
}
