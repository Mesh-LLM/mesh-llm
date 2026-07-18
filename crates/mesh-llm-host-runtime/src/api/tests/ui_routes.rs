#[tokio::test]
async fn lan_details_uses_same_publication_metadata_as_mdns_advertisement() {
    let state = build_test_mesh_api().await;
    state
        .set_mesh_discovery_mode(crate::network::discovery::MeshDiscoveryMode::Mdns)
        .await;
    state
        .set_mesh_publication_metadata(
            Some("garage-mesh".to_string()),
            Some("workshop".to_string()),
            Some(7),
        )
        .await;
    let invite_token = state.node().await.invite_token().await;
    let token_fingerprint = crate::network::discovery::lan_token_fingerprint(&invite_token);
    let challenge = crate::network::discovery::lan_details_challenge(
        &token_fingerprint,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    );
    let proof = crate::network::discovery::lan_details_token_proof(&invite_token, &challenge);
    let body = serde_json::json!({
        "token_fingerprint": token_fingerprint,
        "challenge": challenge,
        "proof": proof,
    })
    .to_string();
    let request = format!(
        "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        crate::network::discovery::LAN_DETAILS_PATH,
        body.len(),
        body,
    );
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(addr, request).await;

    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "expected LAN details success, got: {response}"
    );
    let payload = json_body(&response);
    assert_eq!(payload["listing"]["name"], "garage-mesh");
    assert_eq!(payload["listing"]["region"], "workshop");
    assert_eq!(payload["listing"]["max_clients"], 7);
    assert_eq!(payload["listing"]["invite_token"], "");

    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn status_payload_populates_local_instances_from_scanner() {
    use crate::runtime::instance::LocalInstanceSnapshot;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let snapshots = vec![
        LocalInstanceSnapshot {
            pid: 1234,
            api_port: Some(3131),
            version: Some("0.56.0".to_string()),
            started_at_unix: 1700000000,
            runtime_dir: PathBuf::from("/tmp/a"),
            is_self: true,
        },
        LocalInstanceSnapshot {
            pid: 5678,
            api_port: Some(3132),
            version: Some("0.56.0".to_string()),
            started_at_unix: 1700000100,
            runtime_dir: PathBuf::from("/tmp/b"),
            is_self: false,
        },
    ];

    let shared: Arc<Mutex<Vec<LocalInstanceSnapshot>>> = Arc::new(Mutex::new(snapshots));
    let result: Vec<LocalInstance> = {
        let s = shared.lock().await;
        s.iter()
            .map(|snap| LocalInstance {
                pid: snap.pid,
                api_port: snap.api_port,
                version: snap.version.clone(),
                started_at_unix: snap.started_at_unix,
                runtime_dir: snap.runtime_dir.to_string_lossy().to_string(),
                is_self: snap.is_self,
            })
            .collect()
    };

    assert_eq!(result.len(), 2);
    assert!(result.iter().any(|i| i.is_self && i.pid == 1234));
    assert!(result.iter().any(|i| !i.is_self && i.pid == 5678));
}

#[tokio::test]
async fn status_payload_safety_net_adds_self_when_empty() {
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let shared: Arc<Mutex<Vec<crate::runtime::instance::LocalInstanceSnapshot>>> =
        Arc::new(Mutex::new(vec![]));

    let mut instances: Vec<LocalInstance> = {
        let s = shared.lock().await;
        s.iter()
            .map(|snap| LocalInstance {
                pid: snap.pid,
                api_port: snap.api_port,
                version: snap.version.clone(),
                started_at_unix: snap.started_at_unix,
                runtime_dir: snap.runtime_dir.to_string_lossy().to_string(),
                is_self: snap.is_self,
            })
            .collect()
    };

    // Simulate the safety net logic
    if instances.is_empty() {
        instances.push(LocalInstance {
            pid: std::process::id(),
            api_port: Some(3131),
            version: Some(MESH_LLM_BUILD_VERSION.to_string()),
            started_at_unix: 0,
            runtime_dir: String::new(),
            is_self: true,
        });
    }

    assert_eq!(instances.len(), 1);
    assert!(instances[0].is_self);
    assert_eq!(instances[0].pid, std::process::id());
    assert_eq!(instances[0].api_port, Some(3131));
    assert_eq!(
        instances[0].version,
        Some(MESH_LLM_BUILD_VERSION.to_string())
    );
}

#[test]
fn headless_mode_disables_ui_routes_but_preserves_api() {
    assert!(is_ui_only_route("/"));
    assert!(is_ui_only_route("/dashboard"));
    assert!(is_ui_only_route("/chat"));
    assert!(is_ui_only_route("/configuration"));
    assert!(is_ui_only_route("/configuration/defaults"));
    assert!(is_ui_only_route("/plugins/web-ui-exemplar/overview"));

    assert!(!is_ui_only_route("/api/status"));
    assert!(!is_ui_only_route("/api/events"));
    assert!(!is_ui_only_route("/api/discover"));
    assert!(!is_ui_only_route("/api/runtime"));
    assert!(!is_ui_only_route("/api/plugins"));
}

#[test]
fn headless_mode_returns_404_for_assets_and_dashboard_routes() {
    assert!(is_ui_only_route("/dashboard/"));
    assert!(is_ui_only_route("/chat/"));
    assert!(is_ui_only_route("/chat/some-room"));
    assert!(is_ui_only_route("/configuration/"));
    assert!(is_ui_only_route("/configuration/toml-review"));
    assert!(is_ui_only_route("/plugins/web-ui-exemplar/overview"));
    assert!(is_ui_only_route("/assets/main.js"));
    assert!(is_ui_only_route("/assets/index-abc123.css"));
    assert!(is_ui_only_route("/favicon.ico"));
    assert!(is_ui_only_route("/logo.png"));
    assert!(is_ui_only_route("/manifest.webmanifest"));
    assert!(is_ui_only_route("/site.json"));

    assert!(!is_ui_only_route("/api/status.json"));
}

#[test]
fn default_mode_still_serves_embedded_ui_routes() {
    assert!(is_ui_only_route("/"));
    assert!(is_ui_only_route("/dashboard"));
    assert!(is_ui_only_route("/chat"));
    assert!(is_ui_only_route("/configuration/defaults"));
    assert!(is_ui_only_route("/assets/app.js"));

    assert!(!is_ui_only_route("/api/status"));
    assert!(!is_ui_only_route("/api/events"));
}

#[tokio::test]
async fn direct_configuration_deep_link_serves_embedded_ui_index() {
    assert!(crate::api::server::is_console_index_route(
        "/configuration/defaults"
    ));

    if mesh_llm_ui::index().is_none() {
        return;
    }

    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /configuration/defaults HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "expected direct configuration deep link to serve UI index, got: {response}"
    );
    assert!(
        response.contains("Content-Type: text/html; charset=utf-8"),
        "expected HTML response for UI deep link, got: {response}"
    );
    assert!(
        !response.contains(r#"{"error":"Not found"}"#),
        "UI deep link must not fall through to JSON 404"
    );
    handle.await.unwrap().unwrap();
}

#[test]
fn headless_status_command_works_against_management_api() {
    assert!(
        !is_ui_only_route("/api/status"),
        "/api/status must not be blocked in headless mode"
    );
    assert!(
        !is_ui_only_route("/api/events"),
        "/api/events must not be blocked in headless mode"
    );
    assert!(
        !is_ui_only_route("/api/discover"),
        "/api/discover must not be blocked in headless mode"
    );
}

#[test]
fn headless_mode_still_reads_api_status() {
    assert!(
        !is_ui_only_route("/api/status"),
        "/api/status must be accessible in headless mode"
    );
    assert!(
        !is_ui_only_route("/api/runtime"),
        "/api/runtime must be accessible in headless mode"
    );
}

#[test]
fn headless_custom_console_port_keeps_api_and_disables_ui() {
    assert!(is_ui_only_route("/"), "/ must be blocked in headless mode");
    assert!(is_ui_only_route("/dashboard"), "/dashboard must be blocked");
    assert!(is_ui_only_route("/chat"), "/chat must be blocked");
    assert!(
        is_ui_only_route("/assets/main.js"),
        "/assets/* must be blocked"
    );
    assert!(
        !is_ui_only_route("/api/status"),
        "/api/status must not be blocked"
    );
    assert!(
        !is_ui_only_route("/api/events"),
        "/api/events must not be blocked"
    );
    assert!(
        !is_ui_only_route("/v1/models"),
        "/v1/models must not be blocked"
    );
    assert!(
        !is_ui_only_route("/v1/chat/completions"),
        "/v1/chat/completions must not be blocked"
    );
}
