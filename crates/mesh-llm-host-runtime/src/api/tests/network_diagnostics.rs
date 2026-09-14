use super::*;

#[tokio::test]
async fn network_diagnostics_route_reports_advertisement_and_peer_paths() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /api/diagnostics/network HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    handle.abort();

    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let body = json_body(&response);
    assert!(body["node_id"].as_str().is_some(), "{body}");

    let advertisement = &body["advertisement"];
    // The test node binds an ephemeral endpoint without relay discovery, so
    // whatever it advertises can never carry an externally observed tuple:
    // every public candidate must be flagged unverified.
    assert!(advertisement["raw_stun_enabled"].as_bool().is_some());
    let candidates = advertisement["public_candidates"].as_array().unwrap();
    for candidate in candidates {
        assert!(candidate["addr"].as_str().is_some());
        assert_eq!(candidate["externally_verified"], false);
    }
    assert!(advertisement["verdict"].as_str().is_some());
    assert!(advertisement["summary"].as_str().is_some());
    assert!(body["peers"].as_array().is_some());
}
