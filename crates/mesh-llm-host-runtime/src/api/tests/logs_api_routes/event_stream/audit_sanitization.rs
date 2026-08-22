use super::*;

#[tokio::test]
#[serial]
async fn logs_events_audit_sanitizes_raw_sql_actor_and_action() {
    // Given: an audit row inserted outside the typed repository with unsafe scalars.
    let _temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let raw_actor = "Bearer sse-raw-actor-secret";
    let oversized_marker = "SSE-RAW-OVERSIZED-ACTION";
    let raw_action = format!("{}{oversized_marker}", "x".repeat(1_100));
    logging
        .store()
        .expect("installed log store")
        .conn()
        .execute(
            "INSERT INTO audit_entries \
             (entry_id, request_id, occurred_at, actor, action, detail_json) \
             VALUES (?1, NULL, ?2, ?3, ?4, ?5)",
            (
                "00000000-0000-4000-8000-0000000000aa",
                "2026-08-22T12:00:00.000000000Z",
                raw_actor,
                raw_action.as_str(),
                r#"{"severity":"warning"}"#,
            ),
        )
        .expect("insert raw audit row");
    let expected_code = mesh_llm_events::audit::SanitizedAuditScalar::sanitize(&raw_action);
    let expected_code_json = format!(
        "\"code\":{}",
        serde_json::to_string(expected_code.as_str()).expect("serialize expected audit code")
    );

    // When: an audit SSE client reconciles durable rows through the management route.
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream
        .write_all(
            b"GET /api/logs/events?audit=true&cursor=a1:0 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
        )
        .await
        .unwrap();
    let response = read_until_contains(
        &mut stream,
        expected_code_json.as_bytes(),
        Duration::from_secs(2),
    )
    .await;
    let response = String::from_utf8(response).unwrap();

    // Then: only the canonical scalar values cross the SSE boundary.
    assert!(response.contains("HTTP/1.1 200 OK"));
    assert!(response.contains("event: audit_entry"));
    assert!(response.contains("\"source\":\"[REDACTED]\""));
    assert!(response.contains(&expected_code_json));
    assert!(!response.contains(raw_actor));
    assert!(!response.contains("sse-raw-actor-secret"));
    assert!(!response.contains(&raw_action));
    assert!(!response.contains(oversized_marker));

    drop(stream);
    server.abort();
    disable_sse_logging().await;
}
