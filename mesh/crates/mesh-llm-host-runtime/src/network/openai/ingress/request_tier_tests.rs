//! The tier a request is served at, as this node's local server sees it:
//! the ingress classifies the request and names it on the loopback hop.

use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// A local server that records the head of the one request it gets.
async fn recording_backend() -> (u16, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let mut buffer = [0; 4096];
        let end = loop {
            let n = stream.read(&mut buffer).await.unwrap();
            assert!(n > 0, "request ended before its head");
            bytes.extend_from_slice(&buffer[..n]);
            if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                break end;
            }
        };
        let body = r#"{"id":"test","model":"m","choices":[]}"#;
        let wire = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(wire.as_bytes()).await.unwrap();
        String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase()
    });
    (port, task)
}

/// Send one request through the ingress to a local server; the head that
/// server received.
async fn head_at_local_server(
    node: &mesh::Node,
    ingress_type: crate::runtime::IngressType,
    requested_by: Option<iroh::EndpointId>,
    extra_headers: &str,
) -> String {
    let (port, backend) = recording_backend().await;
    let mut targets = election::ModelTargets::default();
    targets
        .targets
        .insert("m".into(), vec![election::InferenceTarget::Local(port)]);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let ingress = {
        let node = node.clone();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle_api_proxy_connection(
                node,
                stream.into(),
                targets,
                affinity::AffinityRouter::new(),
                ingress_type,
                requested_by,
            )
            .await;
        })
    };
    let mut client = TcpStream::connect(address).await.unwrap();
    let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{extra_headers}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    client.write_all(request.as_bytes()).await.unwrap();
    let mut response = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        client.read_to_string(&mut response),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    ingress.await.unwrap();
    backend.await.unwrap()
}

fn tier_lines(head: &str) -> Vec<&str> {
    head.lines()
        .filter(|line| line.starts_with("x-skippy-request-tier:"))
        .collect()
}

#[tokio::test]
async fn a_local_client_is_served_at_local_unless_it_asks_for_less() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let local = crate::runtime::IngressType::LocalOpenAi;

    let head = head_at_local_server(&node, local, None, "").await;
    assert!(tier_lines(&head).is_empty(), "{head}");

    // A client cannot speak for the node to its server; it can only lower
    // its own tier through the mesh header.
    let head = head_at_local_server(
        &node,
        local,
        None,
        "x-skippy-request-tier: local\r\nx-skippy-cache-partition: private\r\nx-mesh-request-tier: trusted\r\n",
    )
    .await;
    assert_eq!(
        tier_lines(&head),
        ["x-skippy-request-tier: trusted"],
        "{head}"
    );
    assert!(!head.contains("x-skippy-cache-partition"), "{head}");
    node.endpoint.close().await;
}

#[tokio::test]
async fn a_peer_is_served_no_higher_than_this_node_classifies_it() {
    let mut node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let remote = crate::runtime::IngressType::RemoteQuicHttp;
    let stranger = iroh::SecretKey::from_bytes(&[91; 32]).public();
    let listed = iroh::SecretKey::from_bytes(&[92; 32]).public();
    node.private_nodes = crate::mesh::private_nodes::PrivateNodes::of([listed]);

    // An unknown peer's claim to be local does not raise it.
    let head = head_at_local_server(
        &node,
        remote,
        Some(stranger),
        "x-mesh-request-tier: local\r\n",
    )
    .await;
    assert_eq!(tier_lines(&head), ["x-skippy-request-tier: mesh"], "{head}");

    // A private node is this node's own; it still names its client's tier.
    let head = head_at_local_server(&node, remote, Some(listed), "").await;
    assert!(tier_lines(&head).is_empty(), "{head}");
    let head = head_at_local_server(
        &node,
        remote,
        Some(listed),
        "x-mesh-request-tier: trusted\r\n",
    )
    .await;
    assert_eq!(
        tier_lines(&head),
        ["x-skippy-request-tier: trusted"],
        "{head}"
    );
    node.endpoint.close().await;
}
