use super::*;

#[tokio::test]
/// Keep one mesh's HTTP ingress out of other catalogs, blob storage and host management.
async fn mesh_http_cannot_reach_other_catalogs_blobs_or_host_management() {
    let fixture = Fixture::new().await;
    fixture.seeds[1]
        .set_serving_models(vec!["beta-private-model".into()])
        .await;
    fixture.seeds[1].regossip().await;
    let models = fixture
        .raw_request(
            0,
            "alpha",
            "GET /v1/models HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n",
        )
        .await;
    assert!(models.starts_with("HTTP/1.1 200"), "{models}");
    assert!(models.contains("shared-test"));
    assert!(!models.contains("beta-private-model"));
    for path in [
        "/api/shared-serving",
        "/v1/files/private",
        "/api/blobs/private",
    ] {
        let request = format!("GET {path} HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n");
        let response = fixture.raw_request(0, "alpha", &request).await;
        assert!(!response.starts_with("HTTP/1.1 2"), "{path}: {response}");
    }
    let body = r#"{"model":"beta-private-model","prompt":"must stay in alpha","max_tokens":1}"#;
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let response = fixture.raw_request(0, "alpha", &request).await;
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(fixture.backend.scopes.lock().unwrap().is_empty());
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));
    fixture.close().await;
}

#[tokio::test]
/// Slow HTTP bodies must hit ingress limits and release all permits when detached.
async fn slow_request_ingress_is_bounded_and_detach_releases_its_tasks() {
    let fixture = Fixture::new().await;
    let status = fixture.host.status().await;
    let endpoint = status.memberships[0]
        .endpoint_id
        .as_ref()
        .unwrap()
        .parse()
        .unwrap();
    let mut slow = Vec::new();
    for _ in 0..4 {
        let (mut send, recv) = fixture.seeds[0].open_http_tunnel(endpoint).await.unwrap();
        send.write_all(b"POST ").await.unwrap();
        slow.push((send, recv));
    }
    tokio::time::timeout(Duration::from_secs(3), async {
        while fixture.host.ingress_slots.available_permits() != 3 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let (mut send, mut recv) = fixture.seeds[0].open_http_tunnel(endpoint).await.unwrap();
    send.write_all(b"POST ").await.unwrap();
    let rejected = tokio::time::timeout(Duration::from_secs(3), recv.read_to_end(4096))
        .await
        .unwrap()
        .unwrap();
    assert!(rejected.starts_with(b"HTTP/1.1 503"));
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));
    fixture.host.leave("alpha").await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while fixture.host.ingress_slots.available_permits() != 7 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    drop((slow, send));
    fixture.close().await;
}

#[tokio::test]
/// Reject HTTP before gossip admission and close stage transport on full-model attachments.
async fn attachment_rejects_unadmitted_http_and_stage_transport() {
    let fixture = Fixture::new().await;
    let outsider = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .clear_ip_transports()
        .bind_addr("127.0.0.1:0".parse::<std::net::SocketAddr>().unwrap())
        .unwrap()
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind()
        .await
        .unwrap();
    let record = fixture.host.record("alpha").unwrap();
    let attachment = record.attachment.lock().await;
    let node = &attachment.as_ref().unwrap().node;
    // Exercise admission over a deterministic loopback path, not the host's
    // VPN/LAN candidates (this fixture's peer is bound only to loopback).
    let port = node
        .endpoint
        .addr()
        .ip_addrs()
        .find(|addr| addr.is_ipv4())
        .unwrap()
        .port();
    let address = iroh::EndpointAddr::new(node.id())
        .with_ip_addr(std::net::SocketAddr::from(([127, 0, 0, 1], port)));
    let connection = outsider
        .connect(address.clone(), crate::protocol::ALPN)
        .await
        .unwrap();
    let (mut send, mut recv) = connection.open_bi().await.unwrap();
    send.write_all(&[crate::protocol::STREAM_TUNNEL_HTTP])
        .await
        .unwrap();
    send.write_all(b"GET /v1/models HTTP/1.1\r\nHost: local\r\n\r\n")
        .await
        .unwrap();
    send.finish().unwrap();
    let rejected = tokio::time::timeout(Duration::from_secs(3), recv.read_to_end(4096))
        .await
        .unwrap();
    assert!(rejected.is_err() || rejected.unwrap().is_empty());
    assert!(fixture.backend.scopes.lock().unwrap().is_empty());
    let stage = outsider
        .connect(address, skippy_protocol::STAGE_ALPN_V2)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), stage.closed())
        .await
        .unwrap();
    assert!(node.control_listener.lock().await.is_none());
    drop(attachment);
    assert!(fixture.request(1, "beta").await.starts_with("HTTP/1.1 200"));
    outsider.close().await;
    fixture.close().await;
}
