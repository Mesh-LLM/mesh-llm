//! Real authenticated QUIC tunnels into the realtime relay: a peer-assigned
//! session runs on a local model or is refused, never forwarded again.

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh::{self, Node, NodeRole};
use crate::network::{affinity::AffinityRouter, tunnel::Manager};

const MODEL: &str = "whisper-live";

/// A node serving `MODEL` as speech recognition on a fake local backend.
async fn speech_host(port: u16) -> (Node, ModelTargets) {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    node.set_hosted_models(vec![MODEL.into()]).await;
    node.set_served_model_descriptors(vec![mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: MODEL.into(),
            ..Default::default()
        },
        capabilities_known: true,
        capabilities: crate::models::ModelCapabilities {
            audio: crate::models::CapabilityLevel::Supported,
            multimodal: true,
            ..Default::default()
        },
        metadata: Some(mesh::ServedModelMetadata {
            workload_class: Some(mesh::ModelWorkloadClass::SpeechRecognition),
            ..Default::default()
        }),
        ..Default::default()
    }])
    .await;
    let mut targets = ModelTargets::default();
    targets
        .targets
        .insert(MODEL.into(), vec![InferenceTarget::Local(port)]);
    (node, targets)
}

/// Accepts one websocket handshake, answers 101, echoes whatever followed the
/// head in the same connection, and returns the head it received.
async fn fake_speech_backend() -> (u16, tokio::task::JoinHandle<String>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let backend = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut received = Vec::new();
        let mut buf = [0u8; 4096];
        let head_end = loop {
            let n = stream.read(&mut buf).await.unwrap();
            received.extend_from_slice(&buf[..n]);
            if let Some(end) = received.windows(4).position(|w| w == b"\r\n\r\n") {
                break end + 4;
            }
        };
        // The early frame may arrive in the same segment or just after.
        while received.len() < head_end + 7 {
            let n = stream.read(&mut buf).await.unwrap();
            if n == 0 {
                break;
            }
            received.extend_from_slice(&buf[..n]);
        }
        stream
            .write_all(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
            .await
            .unwrap();
        stream.write_all(&received[head_end..]).await.unwrap();
        stream.shutdown().await.unwrap();
        String::from_utf8_lossy(&received[..head_end]).into_owned()
    });
    (port, backend)
}

/// Opens an HTTP tunnel from a fresh peer to `host`, writes `opening` in one
/// write, and returns everything the host sends back.
async fn tunnel_from_peer(host: &Node, targets: ModelTargets, opening: &[u8]) -> Vec<u8> {
    let caller = Box::pin(Node::new_for_tests(NodeRole::Worker))
        .await
        .unwrap();
    let (legacy_tx, legacy_rx) = tokio::sync::mpsc::channel(1);
    let (http_tx, http_rx) = tokio::sync::mpsc::channel(1);
    let (stage_tx, stage_rx) = tokio::sync::mpsc::channel(1);
    let tunnels = Manager::start(host.clone(), legacy_rx, http_rx, stage_rx)
        .await
        .unwrap();
    let (_targets_tx, targets_rx) = tokio::sync::watch::channel(targets);
    tunnels.set_http_ingress(targets_rx, AffinityRouter::new());
    let serving_node = host.clone();
    let serving = tokio::spawn(async move {
        let connection = serving_node.endpoint.accept().await.unwrap().await.unwrap();
        let remote = connection.remote_id();
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
            .connect(host.endpoint.addr(), crate::protocol::ALPN_V1),
    )
    .await
    .unwrap()
    .unwrap();
    let (mut send, mut recv) = connection.open_bi().await.unwrap();
    let mut first = vec![crate::protocol::STREAM_TUNNEL_HTTP];
    first.extend_from_slice(opening);
    send.write_all(&first).await.unwrap();
    let server_connection = serving.await.unwrap();
    let mut response = Vec::new();
    let _ = tokio::time::timeout(
        Duration::from_secs(10),
        AsyncReadExt::read_to_end(&mut recv, &mut response),
    )
    .await;
    connection.close(0u32.into(), b"test complete");
    server_connection.close(0u32.into(), b"test complete");
    drop((tunnels, legacy_tx, stage_tx));
    caller.endpoint.close().await;
    response
}

fn opening(model: &str) -> Vec<u8> {
    let mut bytes = format!(
        "GET /v1/realtime?model={model} HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
    )
    .into_bytes();
    // A first frame sent together with the handshake.
    bytes.extend_from_slice(b"\x82\x05early");
    bytes
}

#[tokio::test]
async fn a_peer_assigned_session_runs_on_the_local_model_with_its_early_frames() {
    let (port, backend) = fake_speech_backend().await;
    let (host, targets) = speech_host(port).await;
    let response = tunnel_from_peer(&host, targets, &opening(MODEL)).await;
    let response = String::from_utf8_lossy(&response).into_owned();
    assert!(response.starts_with("HTTP/1.1 101"), "{response}");
    assert!(response.ends_with("early"), "{response}");
    let head = backend.await.unwrap();
    assert!(head.contains("Connection: Upgrade"), "{head}");
    assert!(head.to_lowercase().contains("upgrade: websocket"), "{head}");
}

#[tokio::test]
async fn a_public_alias_is_forwarded_as_the_resolved_model_through_quic() {
    const PUBLIC: &str = "speech/whisper:live";
    let (port, backend) = fake_speech_backend().await;
    let (host, targets) = speech_host(port).await;
    let mut descriptors = host.all_served_model_descriptors().await;
    descriptors[0].identity.source_kind = mesh::ModelSourceKind::Catalog;
    descriptors[0].identity.canonical_ref = Some(PUBLIC.into());
    host.set_served_model_descriptors(descriptors).await;

    // Local ingress resolves the public alias before assigning the session
    // to a peer, which receives the internal model in its request target.
    let resolved = super::resolve_model(&host, &targets, PUBLIC, false).await;
    assert_eq!(resolved, MODEL);
    let opening = super::upgrade_request(
        &opening("speech%2Fwhisper%3Alive&intent=transcription&x=a%20b"),
        &resolved,
    );
    let response = tunnel_from_peer(&host, targets, &opening).await;
    assert!(response.starts_with(b"HTTP/1.1 101"));
    assert!(response.ends_with(b"early"));
    let head = backend.await.unwrap();
    assert!(
        head.starts_with(
            "GET /v1/realtime?intent=transcription&x=a%20b&model=whisper-live HTTP/1.1\r\n"
        ),
        "{head}"
    );
    assert!(!head.contains("speech%2Fwhisper"), "{head}");
}

#[tokio::test]
async fn a_peer_assigned_session_for_a_model_not_served_here_is_refused_not_forwarded() {
    let (port, _backend) = fake_speech_backend().await;
    let (host, targets) = speech_host(port).await;
    let response = tunnel_from_peer(&host, targets, &opening("someone-elses-model")).await;
    let response = String::from_utf8_lossy(&response).into_owned();
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(response.contains("not served by this host"), "{response}");
}
