//! The host-served terminal reports the request's client nonce, so the
//! routing node and the serving node of one exchange report the same nonce.

use tokio::io::AsyncWriteExt;

use crate::network::openai::forwarded_request::prepare_peer_forwarded_request;
use crate::plugin::openai_exchange::{
    ClientNonceSource, OpenAiExchangeDispatchPath, OpenAiExchangeEnvelope,
    test_support::RecordingChannel,
};

use super::{
    RawProxyTerminalFacts, local_route_delivered_by_peer, mesh, proxy,
    publish_local_route_terminal, publish_raw_proxy_terminal, request_client_nonce,
};

const CLIENT_NONCE: &str = "6d7d8d2e-3f4a-4b5c-8d9e-0a1b2c3d4e5f";
const MODEL: &str = "acme/shared-model";

fn client_request(nonce: Option<&str>) -> Vec<u8> {
    let body = format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#);
    let nonce_header = nonce
        .map(|nonce| format!("x-capsule-client-nonce: {nonce}\r\n"))
        .unwrap_or_default();
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n{nonce_header}Content-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

/// Read `raw` through the ingress parser, as a node receives it.
async fn ingress(raw: &[u8]) -> proxy::BufferedHttpRequest {
    let (mut client, mut server) = tokio::io::duplex(64 * 1024);
    client.write_all(raw).await.expect("write request");
    proxy::read_http_request(&mut server)
        .await
        .expect("parse request")
}

/// Node A's `RemoteMesh` terminal and node B's host-served terminal for one
/// exchange the client sent to A and A routed to B.
async fn routed_exchange_terminals(
    client_nonce: Option<&str>,
) -> (OpenAiExchangeEnvelope, OpenAiExchangeEnvelope) {
    let node_a = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("node A");
    let node_b = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("node B");
    let outcome = proxy::RouteDispatchOutcome::Responded(200);

    let request_a = ingress(&client_request(client_nonce)).await;
    let recording_a = RecordingChannel::default();
    publish_local_route_terminal(
        &node_a,
        &recording_a,
        "exchange-a",
        MODEL,
        &outcome,
        Some(local_route_delivered_by_peer(
            &request_a,
            "exchange-a".to_string(),
            MODEL,
            node_b.id().to_string(),
            None,
            None,
        )),
        RawProxyTerminalFacts::default(),
    )
    .await;

    let forwarded = prepare_peer_forwarded_request(&request_a.raw).expect("peer forward");
    let request_b = ingress(&forwarded).await;
    let client_nonce_b = request_client_nonce(&request_b);
    let requested_by = node_a.id().to_string();
    let recording_b = RecordingChannel::default();
    publish_local_route_terminal(
        &node_b,
        &recording_b,
        "exchange-b",
        MODEL,
        &outcome,
        None,
        RawProxyTerminalFacts {
            served_locally: true,
            requested_by_node_id: Some(&requested_by),
            client_nonce: client_nonce_b
                .as_ref()
                .map(|(nonce, source)| (nonce.as_str(), *source)),
            ..Default::default()
        },
    )
    .await;

    let [terminal_a] = recording_a.events().try_into().expect("one A terminal");
    let [terminal_b] = recording_b.events().try_into().expect("one B terminal");
    assert_eq!(
        terminal_a.dispatch_path,
        OpenAiExchangeDispatchPath::RemoteMesh
    );
    assert_eq!(
        terminal_b.dispatch_path,
        OpenAiExchangeDispatchPath::RawProxy
    );
    (terminal_a, terminal_b)
}

#[tokio::test]
async fn served_terminal_carries_the_client_supplied_nonce_of_the_routed_exchange() {
    let (terminal_a, terminal_b) = routed_exchange_terminals(Some(CLIENT_NONCE)).await;

    assert_eq!(terminal_a.nonce.as_deref(), Some(CLIENT_NONCE));
    assert_eq!(terminal_b.nonce.as_deref(), Some(CLIENT_NONCE));
    assert_eq!(
        terminal_a.nonce_source,
        Some(ClientNonceSource::ClientSupplied)
    );
    assert_eq!(
        terminal_b.nonce_source,
        Some(ClientNonceSource::ClientSupplied)
    );
}

/// When the client sent no nonce, A mints one and reports it as its own
/// fallback; B serves that same nonce and, with the origin marker stripped at
/// its ingress, reports it as client-supplied.
#[tokio::test]
async fn served_terminal_carries_the_nonce_the_routing_node_minted() {
    let (terminal_a, terminal_b) = routed_exchange_terminals(None).await;

    let minted = terminal_a.nonce.as_deref().expect("A mints a nonce");
    assert_eq!(terminal_b.nonce.as_deref(), Some(minted));
    assert_eq!(
        terminal_a.nonce_source,
        Some(ClientNonceSource::SidecarGeneratedFallback)
    );
    assert_eq!(
        terminal_b.nonce_source,
        Some(ClientNonceSource::ClientSupplied)
    );
}

/// A caller that passes no client nonce (the plugin-served branch, the paid
/// path) still gets a terminal with neither field.
#[tokio::test]
async fn raw_proxy_terminal_without_a_client_nonce_omits_both_fields() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let recording = RecordingChannel::default();
    publish_raw_proxy_terminal(
        &node,
        &recording,
        "exchange-plugin",
        MODEL,
        &proxy::RouteDispatchOutcome::Responded(200),
        RawProxyTerminalFacts::default(),
    )
    .await;

    let [terminal] = recording.events().try_into().expect("one terminal");
    assert!(terminal.nonce.is_none());
    assert!(terminal.nonce_source.is_none());
}
