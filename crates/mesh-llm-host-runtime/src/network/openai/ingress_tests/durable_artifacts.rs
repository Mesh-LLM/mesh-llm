use mesh_llm_events::logging::events::TokenUsage;
use mesh_llm_events::logging::identifiers::RequestId;

use crate::plugin::openai_exchange::{OpenAiExchangePhase, test_support::RecordingChannel};

use super::{
    affinity, election, handle_api_proxy_connection, mesh, proxy, publish_raw_proxy_terminal,
};

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

/// A node with a real GPU/hostname/VRAM survey and a served-model descriptor
/// for `model_name`, so `publish_raw_proxy_terminal`'s served-2xx branch has
/// real hardware and model identity to attach.
async fn node_with_hardware_and_descriptor(model_name: &str) -> mesh::Node {
    let mut node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.gpu_name = Some("Test GPU".to_string());
    node.hostname = Some("test-host".to_string());
    node.is_soc = Some(false);
    node.advertised_memory = mesh::AdvertisedMemory {
        total_bytes: 16_000_000_000,
        ..Default::default()
    };
    node.set_served_model_descriptors(vec![mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model_name.to_string(),
            identity_hash: Some("hash-abc123".to_string()),
            canonical_ref: Some("org/repo@rev1".to_string()),
            revision: Some("rev1".to_string()),
            ..Default::default()
        },
        metadata: Some(mesh::ServedModelMetadata {
            quant: Some("Q4_K_M".to_string()),
            architecture: Some("llama".to_string()),
            native_context_length: Some(4096),
            parameter_size: Some("7B".to_string()),
            layer_count: Some(32),
            ..Default::default()
        }),
        ..Default::default()
    }])
    .await;
    node
}

/// A served 2xx outcome on the host-served path attaches the full serving
/// provenance block (hardware + model identity, both real), the real usage
/// the backend reported, and the real request digest — everything a
/// downstream capsule needs, and nothing fabricated.
#[tokio::test]
async fn publish_raw_proxy_terminal_attaches_full_provenance_and_usage_on_a_served_2xx_outcome() {
    let node = node_with_hardware_and_descriptor("test-model").await;
    let channel = RecordingChannel::default();
    let outcome = proxy::RouteDispatchOutcome::RespondedWithUsage {
        status_code: 200,
        usage: TokenUsage {
            prompt_tokens: Some(10),
            cached_prompt_tokens: Some(2),
            completion_tokens: Some(5),
            total_tokens: Some(15),
        },
        output_digests: Default::default(),
    };

    publish_raw_proxy_terminal(
        &node,
        &channel,
        "exchange-1",
        "test-model",
        &outcome,
        true,
        Some("digest-abc"),
    )
    .await;

    let events = channel.events();
    assert_eq!(events.len(), 1);
    let event = &events[0];
    assert_eq!(event.phase, OpenAiExchangePhase::Terminal);
    assert_eq!(event.status, Some(200));
    assert_eq!(event.request_digest.as_deref(), Some("digest-abc"));

    let provenance = event
        .serving_provenance
        .as_ref()
        .expect("served 2xx outcome carries provenance");
    assert_eq!(provenance.served_by_node_id, node.id().to_string());
    assert_eq!(provenance.hostname.as_deref(), Some("test-host"));
    assert_eq!(provenance.gpu.as_deref(), Some("Test GPU"));
    assert_eq!(provenance.vram_bytes, Some(16_000_000_000));
    assert_eq!(provenance.is_soc, Some(false));
    assert_eq!(provenance.quantization.as_deref(), Some("Q4_K_M"));
    assert_eq!(provenance.architecture.as_deref(), Some("llama"));
    assert_eq!(provenance.context_length, Some(4096));
    assert_eq!(provenance.parameter_size.as_deref(), Some("7B"));
    assert_eq!(provenance.layer_count, Some(32));
    assert_eq!(
        provenance.model_identity_hash.as_deref(),
        Some("hash-abc123")
    );
    assert_eq!(
        provenance.model_canonical_ref.as_deref(),
        Some("org/repo@rev1")
    );
    assert_eq!(provenance.model_revision.as_deref(), Some("rev1"));

    let usage = event.usage.expect("served 2xx outcome carries real usage");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.cached_prompt_tokens, Some(2));
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

/// A degraded 503 served nothing, so there is no hardware or model identity
/// to report — but the client did get a real status, and the terminal event
/// must carry it rather than leaving the outcome unaccounted for.
#[tokio::test]
async fn publish_raw_proxy_terminal_on_a_503_has_no_provenance_but_keeps_the_status() {
    let node = node_with_hardware_and_descriptor("test-model").await;
    let channel = RecordingChannel::default();
    let outcome = proxy::RouteDispatchOutcome::Responded(503);

    publish_raw_proxy_terminal(
        &node,
        &channel,
        "exchange-1",
        "test-model",
        &outcome,
        true,
        None,
    )
    .await;

    let events = channel.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].status, Some(503));
    assert!(events[0].serving_provenance.is_none());
    assert!(events[0].usage.is_none());
}

/// A `Failed` outcome never produced an HTTP response at all, so unlike the
/// 503 case there is no status to report either — never a fabricated one.
#[tokio::test]
async fn publish_raw_proxy_terminal_on_a_failed_outcome_has_no_provenance_and_no_status() {
    let node = node_with_hardware_and_descriptor("test-model").await;
    let channel = RecordingChannel::default();
    let outcome = proxy::RouteDispatchOutcome::Failed("connect_timeout");

    publish_raw_proxy_terminal(
        &node,
        &channel,
        "exchange-1",
        "test-model",
        &outcome,
        true,
        None,
    )
    .await;

    let events = channel.events();
    assert_eq!(events.len(), 1);
    assert!(events[0].status.is_none());
    assert!(events[0].serving_provenance.is_none());
}

/// A `Dropped` outcome (the client disconnected mid-dispatch) is the same
/// shape as `Failed` for this envelope: no status, no provenance.
#[tokio::test]
async fn publish_raw_proxy_terminal_on_a_dropped_outcome_has_no_provenance_and_no_status() {
    let node = node_with_hardware_and_descriptor("test-model").await;
    let channel = RecordingChannel::default();
    let outcome = proxy::RouteDispatchOutcome::Dropped("client_disconnected");

    publish_raw_proxy_terminal(
        &node,
        &channel,
        "exchange-1",
        "test-model",
        &outcome,
        true,
        None,
    )
    .await;

    let events = channel.events();
    assert_eq!(events.len(), 1);
    assert!(events[0].status.is_none());
    assert!(events[0].serving_provenance.is_none());
}

/// The plugin-served path omits the WHOLE serving-provenance block on a 2xx
/// outcome, even when the node happens to have real hardware and a
/// served-model descriptor for the same model name — a plugin endpoint can
/// proxy anywhere, so none of this node's own hardware/identity is honest to
/// report for it.
#[tokio::test]
async fn publish_raw_proxy_terminal_on_the_plugin_served_path_omits_the_whole_block_even_on_2xx() {
    let node = node_with_hardware_and_descriptor("test-model").await;
    let channel = RecordingChannel::default();
    let outcome = proxy::RouteDispatchOutcome::Responded(200);

    publish_raw_proxy_terminal(
        &node,
        &channel,
        "exchange-1",
        "test-model",
        &outcome,
        false, // plugin-served
        None,
    )
    .await;

    let events = channel.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].status, Some(200));
    assert!(events[0].serving_provenance.is_none());
}

/// Unlike `serving_provenance`, the response/tool_calls/reasoning output
/// digests are NOT gated on `served_locally`: a digest of what the host
/// actually returned is defined on every outcome that had a body, plugin-
/// served included. This is the exact property #7b guards -- only
/// `serving_provenance`/`usage` sit behind the `served_locally`/2xx gate;
/// mutating this attachment to gate on `served_locally` must turn this test
/// red.
#[tokio::test]
async fn publish_raw_proxy_terminal_on_the_plugin_served_path_still_attaches_output_digests() {
    let node = node_with_hardware_and_descriptor("test-model").await;
    let channel = RecordingChannel::default();
    let digests = crate::plugin::openai_exchange::ExchangeOutputDigests::from_response_body(
        br#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}"#,
    );
    let outcome = proxy::RouteDispatchOutcome::RespondedWithUsage {
        status_code: 200,
        usage: TokenUsage {
            prompt_tokens: Some(1),
            cached_prompt_tokens: None,
            completion_tokens: Some(1),
            total_tokens: Some(2),
        },
        output_digests: digests,
    };

    publish_raw_proxy_terminal(
        &node,
        &channel,
        "exchange-1",
        "test-model",
        &outcome,
        false, // plugin-served
        None,
    )
    .await;

    let events = channel.events();
    assert_eq!(events.len(), 1);
    assert!(events[0].serving_provenance.is_none());
    assert!(
        events[0].response_digest.is_some(),
        "output digests must attach on the plugin-served path -- not gated on served_locally"
    );
}

/// A served 2xx outcome for a model this node has no served-model descriptor
/// for (a routing-table/descriptor-list staleness window) still reports the
/// real hardware survey, but every model-identity field stays `None` rather
/// than borrowing another model's descriptor. The node DOES have a
/// descriptor registered — just for a different model — so this actually
/// exercises the name match rather than an incidentally-empty list.
#[tokio::test]
async fn publish_raw_proxy_terminal_omits_model_identity_on_a_descriptor_miss() {
    let mut node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.gpu_name = Some("Test GPU".to_string());
    node.hostname = Some("test-host".to_string());
    node.advertised_memory = mesh::AdvertisedMemory {
        total_bytes: 16_000_000_000,
        ..Default::default()
    };
    // A descriptor IS registered, but for a different model than the one
    // being served — must not be borrowed for "test-model".
    node.set_served_model_descriptors(vec![mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: "other-model".to_string(),
            identity_hash: Some("other-hash".to_string()),
            ..Default::default()
        },
        metadata: Some(mesh::ServedModelMetadata {
            quant: Some("Q8_0".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    }])
    .await;
    let channel = RecordingChannel::default();
    let outcome = proxy::RouteDispatchOutcome::Responded(200);

    publish_raw_proxy_terminal(
        &node,
        &channel,
        "exchange-1",
        "test-model",
        &outcome,
        true,
        None,
    )
    .await;

    let events = channel.events();
    let provenance = events[0]
        .serving_provenance
        .as_ref()
        .expect("hardware is still real and reportable on a descriptor miss");
    assert_eq!(provenance.gpu.as_deref(), Some("Test GPU"));
    assert_eq!(provenance.vram_bytes, Some(16_000_000_000));
    assert!(provenance.quantization.is_none());
    assert!(provenance.architecture.is_none());
    assert!(provenance.context_length.is_none());
    assert!(provenance.parameter_size.is_none());
    assert!(provenance.layer_count.is_none());
    assert!(provenance.model_identity_hash.is_none());
    assert!(provenance.model_canonical_ref.is_none());
    assert!(provenance.model_revision.is_none());
}

/// `advertised_memory.total_bytes == 0` means nothing was actually enumerated
/// (a bare CPU host with no accelerator), so `vram_bytes` is omitted rather
/// than reporting a fabricated zero.
#[tokio::test]
async fn publish_raw_proxy_terminal_omits_vram_bytes_when_advertised_total_is_zero() {
    // node_with_hardware_and_descriptor sets a nonzero total; override it.
    let mut node = node_with_hardware_and_descriptor("test-model").await;
    node.advertised_memory = mesh::AdvertisedMemory::default();
    let channel = RecordingChannel::default();
    let outcome = proxy::RouteDispatchOutcome::Responded(200);

    publish_raw_proxy_terminal(
        &node,
        &channel,
        "exchange-1",
        "test-model",
        &outcome,
        true,
        None,
    )
    .await;

    let events = channel.events();
    let provenance = events[0]
        .serving_provenance
        .as_ref()
        .expect("served 2xx outcome still carries provenance");
    assert!(provenance.vram_bytes.is_none());
    // The rest of the hardware/model survey is unaffected by the VRAM figure
    // being unavailable.
    assert_eq!(provenance.gpu.as_deref(), Some("Test GPU"));
    assert_eq!(provenance.quantization.as_deref(), Some("Q4_K_M"));
}
