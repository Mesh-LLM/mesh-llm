//! Real authenticated QUIC ingress with an installed observer child process.

use super::lifecycle_live_tests::{LiveHost, response_entity};
use crate::mesh::{Node, NodeRole};
use crate::network::{affinity::AffinityRouter, tunnel::Manager};
use sha2::{Digest, Sha256};
use std::time::Duration;
use tokio::io::AsyncReadExt;

async fn completed_exchange_events(host: &LiveHost, count: usize) -> Vec<serde_json::Value> {
    // Client EOF can precede the independent observer's terminal callback.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let events = host.events();
            if events
                .iter()
                .filter(|event| event["phase"] == "exchange_finished")
                .count()
                >= count
            {
                return events;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("observer must publish a terminal callback after QUIC client EOF")
}

async fn quic_request(host: &LiveHost, body: &[u8]) -> Vec<u8> {
    quic_request_with_id(host, body, uuid::Uuid::new_v4()).await
}

async fn quic_request_with_id(host: &LiveHost, body: &[u8], exchange_id: uuid::Uuid) -> Vec<u8> {
    let caller = Box::pin(Node::new_for_tests(NodeRole::Worker))
        .await
        .unwrap();
    let caller_id = caller.id();
    let (legacy_tx, legacy_rx) = tokio::sync::mpsc::channel(1);
    let (http_tx, http_rx) = tokio::sync::mpsc::channel(1);
    let (stage_tx, stage_rx) = tokio::sync::mpsc::channel(1);
    let tunnels = Manager::start(host.node.clone(), legacy_rx, http_rx, stage_rx)
        .await
        .unwrap();
    let (_targets_tx, targets_rx) = tokio::sync::watch::channel(host.targets.clone());
    tunnels.set_http_ingress(targets_rx, AffinityRouter::new());
    let serving_node = host.node.clone();
    let serving = tokio::spawn(async move {
        let connection = serving_node.endpoint.accept().await.unwrap().await.unwrap();
        // The identity comes from QUIC authentication, never an HTTP header.
        let remote = connection.remote_id();
        assert_eq!(remote, caller_id);
        let (send, mut recv) = connection.accept_bi().await.unwrap();
        let mut marker = [0];
        recv.read_exact(&mut marker).await.unwrap();
        assert_eq!(marker[0], crate::protocol::STREAM_TUNNEL_HTTP);
        http_tx.send((remote, send, recv)).await.unwrap();
        connection
    });
    let connection = tokio::time::timeout(
        Duration::from_secs(10),
        caller
            .endpoint
            .connect(host.node.endpoint.addr(), crate::protocol::ALPN_V1),
    )
    .await
    .unwrap()
    .unwrap();
    let (mut send, mut recv) = connection.open_bi().await.unwrap();
    send.write_all(&[crate::protocol::STREAM_TUNNEL_HTTP])
        .await
        .unwrap();
    send.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nx-request-id: {exchange_id}\r\nAuthorization: Bearer must-stay-private\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).as_bytes()).await.unwrap();
    send.write_all(body).await.unwrap();
    let server_connection = serving.await.unwrap();
    let mut response = Vec::new();
    tokio::time::timeout(
        Duration::from_secs(10),
        AsyncReadExt::read_to_end(&mut recv, &mut response),
    )
    .await
    .unwrap()
    .unwrap();
    connection.close(0u32.into(), b"test complete");
    server_connection.close(0u32.into(), b"test complete");
    drop((tunnels, legacy_tx, stage_tx));
    caller.endpoint.close().await;
    response
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_quic_ingress_delivers_real_stream_and_denies_before_backend() {
    let host = LiveHost::start(true, true).await;
    let body = br#"{"model":"allowed-model","stream":true,"messages":[{"role":"user","content":"hello"}]}"#;
    let response = quic_request(&host, body).await;
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "{}",
        String::from_utf8_lossy(&response)
    );
    assert_eq!(host.requests.lock().await.len(), 1);
    let events = completed_exchange_events(&host, 1).await;
    let received = events
        .iter()
        .find(|event| event["phase"] == "request_received")
        .unwrap();
    let finished = events
        .iter()
        .find(|event| event["phase"] == "exchange_finished")
        .unwrap();
    assert_eq!(
        received["request_wire_digest"]["sha256"],
        hex::encode(Sha256::digest(body))
    );
    assert_eq!(
        finished["response_wire_commitment"]["sha256"],
        hex::encode(Sha256::digest(response_entity(&response)))
    );
    assert!(
        events
            .iter()
            .all(|event| event["headers"].get("authorization").is_none())
    );
    let denied = quic_request(
        &host,
        br#"{"model":"blocked-model","messages":[{"role":"user","content":"denied"}]}"#,
    )
    .await;
    assert!(denied.starts_with(b"HTTP/1.1 403"));
    assert_eq!(
        host.requests.lock().await.len(),
        1,
        "denied QUIC request reached backend"
    );
    let events = completed_exchange_events(&host, 2).await;
    assert_eq!(
        events
            .iter()
            .filter(|event| event["phase"] == "exchange_finished")
            .count(),
        2
    );
    host.stop().await;
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_quic_gate_deny_answers_402_before_backend_and_in_the_terminal() {
    // A gate plugin (the shape a payment plugin takes) denies with 402: the
    // peer sees "Payment Required" on the wire, the backend never runs, and
    // the serving host's own terminal event records 402 — not the 403 a
    // plain policy deny carries — so an observer plugin and the ops log tell
    // the same story as the wire.
    let host = LiveHost::start_with_deny_status(true, true, 402).await;
    let denied = quic_request(
        &host,
        br#"{"model":"blocked-model","messages":[{"role":"user","content":"pay me"}]}"#,
    )
    .await;
    assert!(
        denied.starts_with(b"HTTP/1.1 402 Payment Required"),
        "{}",
        String::from_utf8_lossy(&denied)
    );
    assert!(
        host.requests.lock().await.is_empty(),
        "402-gated QUIC request reached backend"
    );
    let events = completed_exchange_events(&host, 1).await;
    let finished = events
        .iter()
        .find(|event| event["phase"] == "exchange_finished")
        .unwrap();
    assert_eq!(finished["status"], 402);
    assert_eq!(finished["execution_outcome"], "policy_denied");
    assert_eq!(finished["admission_denied"], true);
    // An allowed model on the same host is untouched by the gate.
    let allowed = quic_request(
        &host,
        br#"{"model":"allowed-model","messages":[{"role":"user","content":"hello"}]}"#,
    )
    .await;
    assert!(
        allowed.starts_with(b"HTTP/1.1 200"),
        "{}",
        String::from_utf8_lossy(&allowed)
    );
    assert_eq!(host.requests.lock().await.len(), 1);
    host.stop().await;
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_quic_terminal_preserves_request_identity_and_reported_usage() {
    let host = LiveHost::start(true, true).await;
    let id = uuid::Uuid::new_v4();
    let response = quic_request_with_id(
        &host,
        br#"{"model":"allowed-model","messages":[{"role":"user","content":"usage"}]}"#,
        id,
    )
    .await;
    assert!(response.starts_with(b"HTTP/1.1 200"));
    let events = completed_exchange_events(&host, 1).await;
    let received = events
        .iter()
        .find(|event| event["phase"] == "request_received")
        .unwrap();
    let terminal = events
        .iter()
        .find(|event| event["phase"] == "exchange_finished")
        .unwrap();
    assert_eq!(received["observation_point"], "serving_host_ingress");
    assert_eq!(received["exchange_id"], id.to_string());
    assert_eq!(terminal["exchange_id"], received["exchange_id"]);
    assert_eq!(terminal["execution_outcome"], "completed");
    assert_eq!(terminal["usage"]["prompt_tokens"], 2);
    assert_eq!(terminal["usage"]["completion_tokens"], 1);
    assert_eq!(terminal["usage"]["total_tokens"], 3);
    assert_eq!(host.requests.lock().await.len(), 1);
    host.stop().await;
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_quic_backend_abort_does_not_invent_terminal_usage() {
    use crate::inference::election::InferenceTarget;
    use tokio::{io::AsyncWriteExt, net::TcpListener};
    let mut host = LiveHost::start(true, true).await;
    let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
    host.targets.targets.insert(
        "allowed-model".into(),
        vec![InferenceTarget::Local(backend.local_addr().unwrap().port())],
    );
    // Report a usage frame but truncate before [DONE] and before Content-Length.
    // A settlement plugin must distinguish this from a completed response.
    let responder = tokio::spawn(async move {
        let (mut stream, _) = backend.accept().await.unwrap();
        super::lifecycle_live_tests::read_entity_request(&mut stream).await;
        let prefix = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n";
        stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", prefix.len() + 100).as_bytes()).await.unwrap();
        stream.write_all(prefix).await.unwrap();
        stream.shutdown().await.unwrap();
    });
    let response = quic_request(
        &host,
        br#"{"model":"allowed-model","stream":true,"messages":[{"role":"user","content":"abort"}]}"#,
    ).await;
    assert!(response.starts_with(b"HTTP/1.1 200"));
    responder.await.unwrap();
    let events = completed_exchange_events(&host, 1).await;
    let terminal = events
        .iter()
        .find(|event| event["phase"] == "exchange_finished")
        .unwrap();
    assert!(
        terminal["usage"].is_null(),
        "aborted response supplied billable usage: {terminal}"
    );
    assert_ne!(terminal["execution_outcome"], "completed");
    host.stop().await;
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_quic_gateway_and_server_share_exchange_identity() {
    use crate::inference::election::InferenceTarget;
    let seller = LiveHost::start(true, true).await;
    let mut buyer = LiveHost::start(true, true).await;
    let mut peer = super::tests::test_remote_peer(42, "allowed-model");
    peer.id = seller.node.id();
    peer.addr = seller.node.endpoint.addr();
    buyer.node.insert_test_peer(peer).await;
    buyer.targets.targets.insert(
        "allowed-model".into(),
        vec![InferenceTarget::Remote(seller.node.id())],
    );
    let (legacy_tx, legacy_rx) = tokio::sync::mpsc::channel(1);
    let (http_tx, http_rx) = tokio::sync::mpsc::channel(1);
    let (stage_tx, stage_rx) = tokio::sync::mpsc::channel(1);
    let tunnels = Manager::start(seller.node.clone(), legacy_rx, http_rx, stage_rx)
        .await
        .unwrap();
    let (_targets_tx, targets_rx) = tokio::sync::watch::channel(seller.targets.clone());
    tunnels.set_http_ingress(targets_rx, AffinityRouter::new());
    let serving_node = seller.node.clone();
    let buyer_id = buyer.node.id();
    let serving = tokio::spawn(async move {
        let connection = serving_node.endpoint.accept().await.unwrap().await.unwrap();
        let remote = connection.remote_id();
        assert_eq!(remote, buyer_id);
        let (send, mut recv) = connection.accept_bi().await.unwrap();
        let mut marker = [0];
        recv.read_exact(&mut marker).await.unwrap();
        assert_eq!(marker[0], crate::protocol::STREAM_TUNNEL_HTTP);
        http_tx.send((remote, send, recv)).await.unwrap();
        connection
    });
    // Install the authenticated connection as an already-admitted peer, as the
    // production gossip handshake does. This fixture tests exchange transport,
    // not gossip admission.
    let connection = buyer
        .node
        .endpoint
        .connect(seller.node.endpoint.addr(), crate::protocol::ALPN_V1)
        .await
        .unwrap();
    buyer
        .node
        .state
        .lock()
        .await
        .connections
        .insert(seller.node.id(), connection);
    let response = buyer
        .request(
            "/v1/chat/completions",
            br#"{"model":"allowed-model","messages":[{"role":"user","content":"hello"}]}"#,
        )
        .await;
    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "{}",
        String::from_utf8_lossy(&response)
    );
    let buyer_events = completed_exchange_events(&buyer, 1).await;
    let seller_events = completed_exchange_events(&seller, 1).await;
    let buyer_received = buyer_events
        .iter()
        .find(|event| event["phase"] == "request_received")
        .unwrap();
    let seller_received = seller_events
        .iter()
        .find(|event| event["phase"] == "request_received")
        .unwrap();
    assert_eq!(buyer_received["observation_point"], "gateway_ingress");
    assert_eq!(seller_received["observation_point"], "serving_host_ingress");
    assert!(buyer_received["exchange_id"].as_str().is_some());
    // The seller's gate learns WHO is asking at admission time, from the QUIC
    // connection (never a header), and its terminal names the same peer, so a
    // gate plugin can key per-peer state on one identity across both phases.
    // The gateway side names no peer: its client is a local HTTP caller.
    let buyer_hex = serde_json::json!(hex::encode(buyer_id.as_bytes()));
    assert_eq!(seller_received["requested_by_node_id"], buyer_hex);
    assert!(buyer_received.get("requested_by_node_id").is_none());
    let seller_terminal = seller_events
        .iter()
        .find(|event| event["phase"] == "exchange_finished")
        .unwrap();
    assert_eq!(seller_terminal["requested_by_node_id"], buyer_hex);
    for events in [&buyer_events, &seller_events] {
        for event in events.iter().filter(|event| {
            event["phase"] == "request_received" || event["phase"] == "exchange_finished"
        }) {
            assert_eq!(event["exchange_id"], buyer_received["exchange_id"]);
        }
    }
    assert_eq!(buyer.requests.lock().await.len(), 0);
    assert_eq!(seller.requests.lock().await.len(), 1);
    serving.await.unwrap().close(0u32.into(), b"test complete");
    drop((tunnels, legacy_tx, stage_tx));
    buyer.stop().await;
    seller.stop().await;
}
