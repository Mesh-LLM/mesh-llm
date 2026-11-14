//! Private routing at the API ingress fails closed: a private request only
//! reaches this node and the nodes it lists, never falls back to the rest.

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

/// With no private nodes listed, no peer serves a private request: every
/// selector gets "no host" rather than a public one.
#[tokio::test]
async fn by_default_a_private_request_reaches_no_peer() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    node.insert_test_peer(listed_peer(70, "qwen")).await;
    assert!(!node.hosts_for_model("qwen").await.is_empty());
    for (path, headers, model) in [
        ("/v1/chat/completions", "", "qwen@private"),
        ("/v1/chat/completions", "x-mesh-scope: private\r\n", "qwen"),
        ("/v1/chat/completions", "X-Mesh-Scope: PRIVATE\r\n", "qwen"),
        ("/v1/chat/completions", "", "qwen@incognito"),
        (
            "/v1/chat/completions",
            "x-mesh-scope: incognito\r\n",
            "qwen",
        ),
        ("/private/v1/chat/completions", "", "qwen"),
    ] {
        let response = roundtrip_path(&node, path, headers, model).await;
        assert!(response.starts_with("HTTP/1.1 404"), "{response}");
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

fn listed_peer(seed: u32, model: &str) -> mesh::PeerInfo {
    let mut peer = super::tests::test_remote_peer(seed, model);
    peer.private_routing_supported = true;
    peer
}

/// A node that lists one peer: through the whole ingress, a private request
/// never selects the unlisted one.
#[tokio::test]
async fn a_private_request_keeps_to_the_listed_nodes() {
    let mut node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let listed = listed_peer(71, "listed-model");
    node.private_nodes = crate::mesh::private_nodes::PrivateNodes::of([listed.id]);
    node.insert_test_peer(listed).await;
    node.insert_test_peer(listed_peer(72, "stranger-model"))
        .await;

    // Both peers are routable for an ordinary request.
    assert!(!node.hosts_for_model("stranger-model").await.is_empty());
    assert!(!node.hosts_for_model("listed-model").await.is_empty());

    // The listed peer is selected (the test peer cannot be reached, so the
    // attempt fails, but not with "no host").
    let listed = roundtrip(&node, "", "listed-model@private").await;
    assert!(!listed.starts_with("HTTP/1.1 404"), "{listed}");

    for (headers, model) in [
        ("", "stranger-model@private"),
        ("x-mesh-scope: incognito\r\n", "stranger-model"),
    ] {
        let response = roundtrip(&node, headers, model).await;
        assert!(response.starts_with("HTTP/1.1 404"), "{response}");
    }
    node.endpoint.close().await;
}
