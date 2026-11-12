//! Private routing at the API ingress fails closed: a node with no verified
//! owner refuses a private request instead of serving it publicly.

use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn roundtrip(node: &mesh::Node, extra_headers: &str, model: &str) -> String {
    roundtrip_path(node, "/v1/chat/completions", extra_headers, model).await
}

async fn roundtrip_path(node: &mesh::Node, path: &str, extra_headers: &str, model: &str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let ingress = {
        let node = node.clone();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle_api_proxy_connection(
                node,
                stream.into(),
                election::ModelTargets::default(),
                affinity::AffinityRouter::new(),
                crate::runtime::IngressType::LocalOpenAi,
                None,
            )
            .await;
        })
    };
    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    let body = format!(r#"{{"model":"{model}","messages":[]}}"#);
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\n{extra_headers}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    client.write_all(request.as_bytes()).await.unwrap();
    let mut response = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client.read_to_string(&mut response),
    )
    .await
    .unwrap()
    .unwrap();
    ingress.await.unwrap();
    response
}

#[tokio::test]
async fn an_unsigned_node_refuses_private_requests() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    for (headers, model) in [
        ("", "qwen@private"),
        ("x-mesh-scope: private\r\n", "qwen"),
        ("X-Mesh-Scope: PRIVATE\r\n", "qwen"),
        ("", "qwen@incognito"),
        ("x-mesh-scope: incognito\r\n", "qwen"),
        ("", "auto@private"),
        ("", "mesh@incognito"),
    ] {
        let response = roundtrip(&node, headers, model).await;
        assert!(response.starts_with("HTTP/1.1 403"), "{response}");
        assert!(response.contains("--owner-key"), "{response}");
    }
    node.endpoint.close().await;
}

#[tokio::test]
async fn a_malformed_scope_header_is_rejected_not_ignored() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    for headers in [
        "x-mesh-scope: public\r\n",
        "x-mesh-scope: private\r\nx-mesh-scope: private\r\n",
    ] {
        let response = roundtrip(&node, headers, "qwen").await;
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    }
    // A bare suffix is a selector with no model, not a model name.
    for model in ["@private", "@incognito"] {
        let response = roundtrip(&node, "", model).await;
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    }
    node.endpoint.close().await;
}

#[tokio::test]
async fn the_private_base_url_is_refused_on_an_unsigned_node() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let response = roundtrip_path(&node, "/private/v1/chat/completions", "", "qwen").await;
    assert!(response.starts_with("HTTP/1.1 403"), "{response}");
    node.endpoint.close().await;
}

fn owned_peer(seed: u32, model: &str, owner: &str) -> mesh::PeerInfo {
    let mut peer = super::tests::test_remote_peer(seed, model);
    peer.owner_summary = crate::crypto::OwnershipSummary {
        owner_id: Some(owner.to_string()),
        verified: true,
        status: crate::crypto::OwnershipStatus::Verified,
        expires_at_unix_ms: Some(u64::MAX),
        ..Default::default()
    };
    peer
}

/// A signed node with one owned peer and one stranger: through the whole
/// ingress, a private request never selects the stranger.
#[tokio::test]
async fn a_signed_node_keeps_private_requests_off_a_strangers_node() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    *node.owner_summary.lock().await = crate::crypto::OwnershipSummary {
        owner_id: Some("me".to_string()),
        verified: true,
        status: crate::crypto::OwnershipStatus::Verified,
        expires_at_unix_ms: Some(u64::MAX),
        ..Default::default()
    };
    node.insert_test_peer(owned_peer(71, "owned-model", "me"))
        .await;
    node.insert_test_peer(owned_peer(72, "stranger-model", "them"))
        .await;

    // Both peers are routable for an ordinary request.
    assert!(!node.hosts_for_model("stranger-model").await.is_empty());
    assert!(!node.hosts_for_model("owned-model").await.is_empty());

    // The owned peer is selected (the test peer cannot be reached, so the
    // attempt fails, but not with "no host").
    let owned = roundtrip(&node, "", "owned-model@private").await;
    assert!(!owned.starts_with("HTTP/1.1 404"), "{owned}");
    assert!(!owned.starts_with("HTTP/1.1 403"), "{owned}");

    for (headers, model) in [
        ("", "stranger-model@private"),
        ("x-mesh-scope: incognito\r\n", "stranger-model"),
    ] {
        let response = roundtrip(&node, headers, model).await;
        assert!(response.starts_with("HTTP/1.1 404"), "{response}");
    }
    node.endpoint.close().await;
}
