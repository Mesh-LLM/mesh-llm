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
    send.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer must-stay-private\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).as_bytes()).await.unwrap();
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
