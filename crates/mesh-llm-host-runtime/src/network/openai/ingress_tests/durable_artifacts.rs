use mesh_llm_events::logging::identifiers::RequestId;

use super::{affinity, election, handle_api_proxy_connection, mesh};

#[tokio::test]
#[serial_test::serial]
async fn parsed_missing_model_error_persists_the_client_visible_response_artifact() {
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
        .expect("bind ingress test listener");
    let address = listener.local_addr().expect("ingress listener address");
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept ingress client");
        handle_api_proxy_connection(
            node,
            stream.into(),
            election::ModelTargets::default(),
            affinity::AffinityRouter::new(),
            crate::runtime::IngressType::LocalOpenAi,
        )
        .await;
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
        .expect("connect ingress client");
    let caller_addr = client
        .local_addr()
        .expect("ingress caller address")
        .to_string();
    client
        .write_all(request.as_bytes())
        .await
        .expect("write parsed request");
    let mut wire = Vec::new();
    client
        .read_to_end(&mut wire)
        .await
        .expect("read ingress error");
    server.await.expect("ingress handler joins");
    assert!(
        String::from_utf8_lossy(&wire).starts_with("HTTP/1.1 404 Not Found"),
        "the parsed no-model route returns its normal client-visible error"
    );

    let state = crate::logging_runtime_state().expect("installed logging runtime");
    let active = state
        .service_for_test()
        .expect("logging service")
        .registry_ref()
        .get_recent(&request_id.as_uuid().to_string())
        .expect("active request summary");
    assert_eq!(active.metadata.caller_addr(), Some(caller_addr.as_str()));
    assert_eq!(active.metadata.caller_path_type(), Some("local_http"));
    state.pump_persistence_for_test().await;
    let request_key = request_id.as_uuid().to_string();
    let durable = state
        .store()
        .expect("metadata store")
        .query_request_with_caller(&request_key)
        .expect("durable request query")
        .expect("durable request summary");
    assert_eq!(durable.caller_addr.as_deref(), Some(caller_addr.as_str()));
    assert_eq!(durable.caller_path_type.as_deref(), Some("local_http"));
    assert!(durable.caller_endpoint_id.is_none());
    let artifacts = state
        .store()
        .expect("metadata store")
        .query_artifacts(
            &request_key,
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
    assert_eq!(content.bytes, wire[body_start..]);
    let response_json = String::from_utf8(content.bytes).expect("JSON response artifact");
    assert!(response_json.contains("not-served"));
    assert!(response_json.contains("model_not_found"));
}

#[tokio::test]
#[serial_test::serial]
async fn ingress_body_parse_error_persists_a_response_only_after_complete_headers() {
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
        .expect("bind ingress parser test listener");
    let address = listener.local_addr().expect("ingress listener address");
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept ingress client");
        handle_api_proxy_connection(
            node,
            stream.into(),
            election::ModelTargets::default(),
            affinity::AffinityRouter::new(),
            crate::runtime::IngressType::LocalOpenAi,
        )
        .await;
    });

    let request_id = RequestId::new();
    let request = format!(
        "POST /v1/tokenize HTTP/1.1\r\nHost: localhost\r\nx-request-id: {}\r\nContent-Length: 1\r\n\r\n{{",
        request_id.as_uuid(),
    );
    let mut client = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect ingress client");
    client
        .write_all(request.as_bytes())
        .await
        .expect("write malformed body");
    let mut wire = Vec::new();
    client
        .read_to_end(&mut wire)
        .await
        .expect("read ingress error");
    server.await.expect("ingress handler joins");
    assert!(String::from_utf8_lossy(&wire).starts_with("HTTP/1.1 400 Bad Request"));

    let state = crate::logging_runtime_state().expect("installed logging runtime");
    state.pump_persistence_for_test().await;
    let request_key = request_id.as_uuid().to_string();
    let artifacts = state
        .store()
        .expect("metadata store")
        .query_artifacts(
            &request_key,
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
    assert_eq!(content.bytes, wire[body_start..]);
}
