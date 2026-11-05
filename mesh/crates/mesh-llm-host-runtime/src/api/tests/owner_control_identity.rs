//! Issue #2209: loopback reachability is not owner identity.
//!
//! The owner-control management routes currently gate only on
//! `ensure_loopback_control_caller`: any local OS account that can reach the
//! management listener passes, and the host then signs owner-control traffic
//! with the server-side owner keypair on the caller's behalf
//! (`connect_owner_control_client`). The snapshot payload also serializes the
//! full owner-control config, so a caller that never proved ownership reads
//! secret configuration such as telemetry authorization headers.
//!
//! These tests encode the post-fix contract: a loopback caller that has not
//! presented owner capability is rejected from owner-control mutations, and an
//! unauthenticated snapshot never carries secret config values.

use super::*;

#[tokio::test]
async fn loopback_token_endpoint_does_not_disclose_the_credential() {
    let state = build_test_mesh_api().await;
    seed_owner_control_token(&state).await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/runtime/control/owner-control-token HTTP/1.1\r\nHost: localhost\r\n\r\n"
            .to_owned(),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 403"), "response: {response}");
    assert!(!response.contains(TEST_OWNER_CONTROL_TOKEN));
    handle.abort();
}

struct OwnerControlSnapshotTestServer {
    endpoint_token: String,
    task: tokio::task::JoinHandle<()>,
}

/// An owner-control target that answers `get-config` with a snapshot whose
/// full `config_toml` carries the given secret-bearing configuration.
async fn spawn_owner_control_snapshot_test_server(
    config_toml: String,
) -> OwnerControlSnapshotTestServer {
    let endpoint = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(iroh::SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V2.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap();
    let endpoint_token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&endpoint.addr()).unwrap());
    let task = tokio::spawn(async move {
        let Some(incoming) = endpoint.accept().await else {
            return;
        };
        let mut accepting = incoming.accept().unwrap();
        let _ = accepting.alpn().await.unwrap();
        let conn = accepting.await.unwrap();
        let (mut send, mut recv) = conn.accept_bi().await.unwrap();
        let handshake = mesh_llm_protocol::read_len_prefixed(&mut recv)
            .await
            .unwrap();
        let _ = decode_owner_control_envelope(&handshake).unwrap();
        let request = mesh_llm_protocol::read_len_prefixed(&mut recv)
            .await
            .unwrap();
        let envelope = decode_owner_control_envelope(&request).unwrap();
        let request_id = envelope.request.as_ref().unwrap().request_id;
        let response = OwnerControlEnvelope {
            r#gen: mesh_llm_protocol::NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(OwnerControlResponse {
                request_id,
                get_config: Some(OwnerControlGetConfigResponse {
                    snapshot: Some(OwnerControlConfigSnapshot {
                        node_id: vec![7; 32],
                        revision: 42,
                        config_hash: vec![9; 32],
                        config: Some(NodeConfigSnapshot {
                            version: 1,
                            gpu: None,
                            models: Vec::new(),
                            plugins: Vec::new(),
                            config_toml: Some(config_toml),
                            mesh_requirements: None,
                            lifecycle_log_parser: None,
                        }),
                        hostname: Some("control-target".to_string()),
                    }),
                }),
                watch_config: None,
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            error: None,
        };
        write_len_prefixed(&mut send, &response.encode_to_vec())
            .await
            .unwrap();
        let _ = send.finish();
        tokio::time::sleep(Duration::from_millis(100)).await;
    });
    OwnerControlSnapshotTestServer {
        endpoint_token,
        task,
    }
}

fn seeded_owner_keypair(temp: &tempfile::TempDir) -> std::path::PathBuf {
    let keystore_path = temp.path().join("owner-keystore.json");
    save_keystore(&keystore_path, &OwnerKeypair::generate(), None, true).unwrap();
    keystore_path
}

/// A plain loopback connection carries no owner capability, so the host must
/// not act as owner on the caller's behalf: `apply-config` must be rejected
/// and the mutation must never reach the target node.
#[tokio::test]
#[serial]
async fn owner_control_apply_config_rejects_loopback_caller_without_owner_capability() {
    let temp = tempfile::tempdir().unwrap();
    let _home_guard = HomeEnvGuard::set(temp.path());
    let keystore_path = seeded_owner_keypair(&temp);

    // The target would accept this mutation if the host forwarded it.
    let OwnerControlApplyTestServer {
        endpoint_token,
        task: control_task,
        received_apply,
    } = spawn_owner_control_apply_test_server(OwnerControlApplyTestResponse::Success(
        OwnerControlApplyConfigResponse {
            success: true,
            current_revision: 43,
            config_hash: vec![0xab; 32],
            error: None,
            apply_mode: ConfigApplyMode::Staged as i32,
            diagnostics: Vec::new(),
        },
    ))
    .await;

    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({
        "endpoint": endpoint_token,
        "expected_revision": 42,
        "config": full_mesh_config_fixture(),
    })
    .to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/apply-config", &request_body),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 401") || response.starts_with("HTTP/1.1 403"),
        "apply-config accepted a loopback caller that presented no owner capability: {response}"
    );

    // The mutation must not have been forwarded to the target node.
    if let Ok(apply) = received_apply
        .expect("apply server should expose its receiver")
        .try_recv()
    {
        panic!(
            "apply-config forwarded a revision-{} mutation for a caller without owner capability",
            apply.expected_revision
        );
    }

    handle.await.unwrap().unwrap();
    control_task.abort();
}

/// The same identity gap exposes the read side: `get-config` must not serve
/// the owner-control snapshot to a caller that has not presented owner
/// capability.
#[tokio::test]
#[serial]
async fn owner_control_get_config_rejects_loopback_caller_without_owner_capability() {
    let temp = tempfile::tempdir().unwrap();
    let _home_guard = HomeEnvGuard::set(temp.path());
    let keystore_path = seeded_owner_keypair(&temp);

    let server =
        spawn_owner_control_snapshot_test_server("[telemetry]\nenabled = true\n".to_string()).await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({ "endpoint": server.endpoint_token }).to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/get-config", &request_body),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 401") || response.starts_with("HTTP/1.1 403"),
        "get-config served an owner-control snapshot to a loopback caller with no owner capability: {response}"
    );

    handle.await.unwrap().unwrap();
    server.task.abort();
}

/// The get-config snapshot must never carry secret configuration values:
/// an unauthenticated caller gets no snapshot at all, and an authenticated
/// owner receives the snapshot with telemetry header values replaced by the
/// redaction sentinel.
#[tokio::test]
#[serial]
async fn owner_control_snapshot_redacts_secret_config_values() {
    const SECRET: &str = "sk-owner-secret-0f3b9c4d2e";
    let temp = tempfile::tempdir().unwrap();
    let _home_guard = HomeEnvGuard::set(temp.path());
    let keystore_path = seeded_owner_keypair(&temp);

    let config_toml = format!(
        r#"version = 1

[telemetry]
enabled = true
service_name = "mesh-llm"
endpoint = "http://127.0.0.1:4317"

[telemetry.headers]
authorization = "Bearer {SECRET}"
"#
    );
    let server = spawn_owner_control_snapshot_test_server(config_toml).await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({ "endpoint": server.endpoint_token }).to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/get-config", &request_body),
    )
    .await;
    handle.await.unwrap().unwrap();
    server.task.abort();

    // Guard against a harness fault passing the test vacuously: the request
    // must have reached the snapshot handler (post-fix, an unauthenticated
    // caller may instead be rejected outright, which is also acceptable).
    let reached_snapshot_handler = response.starts_with("HTTP/1.1 200")
        || response.starts_with("HTTP/1.1 401")
        || response.starts_with("HTTP/1.1 403");
    assert!(
        reached_snapshot_handler,
        "get-config did not reach the snapshot handler: {response}"
    );

    assert!(
        !response.contains(SECRET),
        "owner-control snapshot leaked the secret telemetry authorization header to a caller without owner capability: {response}"
    );
}

/// The authenticated owner receives the snapshot, with telemetry header
/// values replaced by the redaction sentinel instead of the real credential.
#[tokio::test]
#[serial]
async fn owner_control_snapshot_serves_redacted_snapshot_to_owner() {
    const SECRET: &str = "sk-owner-secret-0f3b9c4d2e";
    let temp = tempfile::tempdir().unwrap();
    let _home_guard = HomeEnvGuard::set(temp.path());
    let keystore_path = seeded_owner_keypair(&temp);

    let config_toml = format!(
        r#"version = 1

[telemetry]
enabled = true
service_name = "mesh-llm"
endpoint = "http://127.0.0.1:4317"

[telemetry.headers]
authorization = "Bearer {SECRET}"
"#
    );
    let server = spawn_owner_control_snapshot_test_server(config_toml).await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    seed_owner_control_token(&state).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({ "endpoint": server.endpoint_token }).to_string();
    let response = send_management_request(
        addr,
        management_post_request_with_owner_control(
            "/api/runtime/control/get-config",
            &request_body,
        ),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 200"),
        "get-config rejected the authenticated owner: {response}"
    );
    assert!(
        response.contains(crate::protocol::config_redaction::REDACTED_TELEMETRY_HEADER_VALUE),
        "owner-control snapshot did not carry the redaction sentinel for the telemetry header: {response}"
    );
    assert!(
        !response.contains(SECRET),
        "owner-control snapshot leaked the secret telemetry authorization header to the owner: {response}"
    );

    handle.await.unwrap().unwrap();
    server.task.abort();
}
