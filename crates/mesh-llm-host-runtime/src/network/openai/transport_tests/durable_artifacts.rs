use super::{AffinityRouter, RequestId, handle_mesh_request};

#[tokio::test]
#[serial_test::serial]
async fn passive_missing_model_error_persists_the_client_visible_response_artifact() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let root = tempfile::tempdir().expect("temporary logging root");
    let mut config = mesh_llm_config::LoggingConfig {
        enabled: true,
        application_state_root: Some(root.path().to_path_buf()),
        ..Default::default()
    };
    config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
    crate::initialize_logging_foundation(&config).await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind passive test listener");
    let address = listener.local_addr().expect("passive listener address");
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
        .await
        .expect("test node");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept passive client");
        handle_mesh_request(node, stream, true, AffinityRouter::new()).await;
    });

    let request_id = RequestId::new();
    let body = r#"{"model":"not-served"}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nx-request-id: {}\r\nContent-Length: {}\r\n\r\n{body}",
        request_id.as_uuid(),
        body.len(),
    );
    let mut client = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect passive client");
    client
        .write_all(request.as_bytes())
        .await
        .expect("write parsed request");
    let mut wire = Vec::new();
    client
        .read_to_end(&mut wire)
        .await
        .expect("read passive error");
    server.await.expect("passive handler joins");
    assert!(
        String::from_utf8_lossy(&wire).starts_with("HTTP/1.1 429 Too Many Requests"),
        "the passive no-model route returns its normal client-visible error"
    );

    let state = crate::logging_runtime_state().expect("installed logging runtime");
    state.pump_persistence_for_test().await;
    let artifacts = state
        .store()
        .expect("metadata store")
        .query_artifacts(
            &request_id.as_uuid().to_string(),
            &mesh_llm_log_store::PageQuery {
                limit: 10,
                cursor: None,
                sort: mesh_llm_log_store::QuerySort::Ascending,
            },
        )
        .expect("response artifact query");
    let response = artifacts
        .items
        .iter()
        .find(|artifact| artifact.kind == "response")
        .expect("durable error response artifact");
    assert_eq!(response.media_kind.as_deref(), Some("application/json"));
    let body_start = wire
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("HTTP response header terminator")
        + 4;
    let content = state
        .query_facade()
        .expect("artifact reader")
        .read_artifact(&response.artifact_id)
        .expect("response artifact content");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&content.bytes).expect("stored JSON response"),
        serde_json::from_slice::<serde_json::Value>(&wire[body_start..])
            .expect("wire JSON response")
    );
    let response_json = String::from_utf8(content.bytes).expect("JSON response artifact");
    assert!(response_json.contains("not-served"));
    assert!(response_json.contains("rate_limit_exceeded"));
}

#[tokio::test]
#[serial_test::serial]
async fn passive_body_parse_error_persists_a_response_only_after_complete_headers() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let root = tempfile::tempdir().expect("temporary logging root");
    let mut config = mesh_llm_config::LoggingConfig {
        enabled: true,
        application_state_root: Some(root.path().to_path_buf()),
        ..Default::default()
    };
    config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
    crate::initialize_logging_foundation(&config).await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind passive parser test listener");
    let address = listener.local_addr().expect("passive listener address");
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
        .await
        .expect("test node");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept passive client");
        handle_mesh_request(node, stream, true, AffinityRouter::new()).await;
    });

    let request_id = RequestId::new();
    let request = format!(
        "POST /v1/tokenize HTTP/1.1\r\nHost: localhost\r\nx-request-id: {}\r\nContent-Length: 1\r\n\r\n{{",
        request_id.as_uuid(),
    );
    let mut client = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect passive client");
    client
        .write_all(request.as_bytes())
        .await
        .expect("write malformed body");
    let mut wire = Vec::new();
    client
        .read_to_end(&mut wire)
        .await
        .expect("read passive error");
    server.await.expect("passive handler joins");
    assert!(String::from_utf8_lossy(&wire).starts_with("HTTP/1.1 400 Bad Request"));

    let state = crate::logging_runtime_state().expect("installed logging runtime");
    state.pump_persistence_for_test().await;
    let artifacts = state
        .store()
        .expect("metadata store")
        .query_artifacts(
            &request_id.as_uuid().to_string(),
            &mesh_llm_log_store::PageQuery {
                limit: 10,
                cursor: None,
                sort: mesh_llm_log_store::QuerySort::Ascending,
            },
        )
        .expect("response artifact query");
    assert_eq!(
        artifacts.items.len(),
        1,
        "pre-admission parse failures must never fabricate a request artifact"
    );
    let response = &artifacts.items[0];
    assert_eq!(response.kind, "response");
    assert_eq!(response.media_kind.as_deref(), Some("application/json"));
    let body_start = wire
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("HTTP response header terminator")
        + 4;
    let content = state
        .query_facade()
        .expect("artifact reader")
        .read_artifact(&response.artifact_id)
        .expect("response artifact content");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&content.bytes).expect("stored JSON response"),
        serde_json::from_slice::<serde_json::Value>(&wire[body_start..])
            .expect("wire JSON response")
    );
}
