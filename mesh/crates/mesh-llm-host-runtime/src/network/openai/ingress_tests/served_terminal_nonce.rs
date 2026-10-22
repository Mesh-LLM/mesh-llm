//! The host-served terminal reports the request's client nonce, so the
//! routing node and the serving node of one exchange report the same nonce.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::logging::OpenAiLifecycleAttachment;
use crate::network::openai::forwarded_request::prepare_peer_forwarded_request;
use crate::plugin::openai_exchange::{
    ClientNonceSource, OpenAiExchangeDispatchPath, OpenAiExchangeEnvelope, OpenAiExchangePhase,
    test_support::RecordingChannel,
};

use super::tests::test_remote_peer;
use super::{
    IngressRouteContext, RawProxyTerminalFacts, affinity, election, mesh, proxy,
    publish_raw_proxy_terminal, route_missing_local_model, route_request,
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

/// A connected loopback socket pair: the route writes its response into the
/// first, and the second stands in for the client.
async fn client_socket_pair() -> (tokio::net::TcpStream, tokio::net::TcpStream) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("local addr");
    let (client, accepted) = tokio::join!(tokio::net::TcpStream::connect(addr), listener.accept());
    (
        accepted.expect("accept server side").0,
        client.expect("connect client side"),
    )
}

/// A local backend that answers one chat completion.
async fn chat_backend() -> u16 {
    let backend = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind backend");
    let port = backend.local_addr().expect("backend addr").port();
    tokio::spawn(async move {
        let (mut stream, _) = backend.accept().await.expect("accept backend");
        proxy::read_http_request(&mut stream)
            .await
            .expect("backend request");
        let body = r#"{"id":"c1","object":"chat.completion","model":"acme/shared-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("backend response");
    });
    port
}

/// Node A routes the client's request to a peer that serves the model, and
/// publishes its `RemoteMesh` terminal. Returns that terminal and the bytes A
/// forwards to the peer.
async fn route_on_a(client_nonce: Option<&str>) -> (OpenAiExchangeEnvelope, Vec<u8>) {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("node A");
    node.insert_test_peer(test_remote_peer(1, MODEL)).await;
    let targets = election::ModelTargets::default();
    let affinity = affinity::AffinityRouter::new();
    let recording = RecordingChannel::default();
    let mut request = ingress(&client_request(client_nonce)).await;
    let ctx = IngressRouteContext {
        node: &node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: None,
        requested_by_node_id: None,
        exchange_channel: Some(&recording),
    };
    let (server_side, _client_side) = client_socket_pair().await;
    let lifecycle = OpenAiLifecycleAttachment::unowned();
    route_missing_local_model(
        server_side.into(),
        &mut request,
        &ctx,
        MODEL,
        None,
        &[],
        None,
        lifecycle.route_observer(),
    )
    .await;

    let terminal = recording.events().pop().expect("A terminal");
    assert_eq!(terminal.phase, OpenAiExchangePhase::Terminal);
    assert_eq!(
        terminal.dispatch_path,
        OpenAiExchangeDispatchPath::RemoteMesh
    );
    // What A sends over the tunnel (`response/routing.rs`).
    let forwarded = prepare_peer_forwarded_request(&request.raw).expect("peer forward");
    (terminal, forwarded)
}

/// Node B reads the tunnelled request and serves it on its own backend,
/// publishing its host-served terminal.
async fn serve_on_b(forwarded: &[u8], requested_by: String) -> OpenAiExchangeEnvelope {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("node B");
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        MODEL.into(),
        vec![election::InferenceTarget::Local(chat_backend().await)],
    );
    let affinity = affinity::AffinityRouter::new();
    let recording = RecordingChannel::default();
    let mut request = ingress(forwarded).await;
    let ctx = IngressRouteContext {
        node: &node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: None,
        requested_by_node_id: Some(requested_by.clone()),
        exchange_channel: Some(&recording),
    };
    let (server_side, mut client_side) = client_socket_pair().await;
    let lifecycle = OpenAiLifecycleAttachment::unowned();
    let outcome = route_request(
        true,
        server_side.into(),
        &mut request,
        &ctx,
        Some(MODEL),
        None,
        lifecycle.route_observer(),
    )
    .await;
    let mut response = String::new();
    client_side
        .read_to_string(&mut response)
        .await
        .expect("read response");
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "{outcome:?}: {response}"
    );

    let terminal = recording.events().pop().expect("B terminal");
    assert_eq!(terminal.phase, OpenAiExchangePhase::Terminal);
    assert_eq!(terminal.dispatch_path, OpenAiExchangeDispatchPath::RawProxy);
    assert_eq!(
        terminal
            .serving_provenance
            .as_ref()
            .and_then(|provenance| provenance.requested_by_node_id.as_deref()),
        Some(requested_by.as_str())
    );
    terminal
}

/// Node A's `RemoteMesh` terminal and node B's host-served terminal for one
/// exchange the client sent to A and A routed to B.
async fn routed_exchange_terminals(
    client_nonce: Option<&str>,
) -> (OpenAiExchangeEnvelope, OpenAiExchangeEnvelope) {
    let (terminal_a, forwarded) = route_on_a(client_nonce).await;
    let terminal_b = serve_on_b(&forwarded, "aa".repeat(32)).await;
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
