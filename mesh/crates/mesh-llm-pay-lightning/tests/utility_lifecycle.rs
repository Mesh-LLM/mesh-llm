//! Installed utility lifecycle over host RPC; never accesses a real wallet.
#![cfg(all(unix, feature = "offline-mock"))]
use mesh_llm_plugin::{LocalStream, PROTOCOL_VERSION, proto, read_envelope, write_envelope};
use mesh_llm_wallet::provisioning::WalletPin;
use std::{path::Path, time::Duration};
use tokio::{
    process::{Child, Command},
    time::timeout,
};

fn configure(root: &Path) -> std::path::PathBuf {
    let config = root.join("config.json");
    std::fs::write(
        &config,
        serde_json::to_vec(&serde_json::json!({
            "data_dir": root.join("ledger")
        }))
        .unwrap(),
    )
    .unwrap();
    config
}

async fn launch(root: &Path, config: &Path) -> (Child, LocalStream) {
    let _ = std::fs::remove_file(root.join("wallet.pid"));
    let socket = root.join("host.sock");
    let _ = std::fs::remove_file(&socket);
    let listener = tokio::net::UnixListener::bind(&socket).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_pay-lightning"))
        .env("MESH_LLM_PLUGIN_SETTINGS_JSON", serde_json::json!({"config_path":config,"wallet_binding":"test-wallet","allow_wallet_provisioning":true}).to_string())
        .env("MESH_LLM_PLUGIN_NAME", "pay-lightning")
        .env("MESH_LLM_PLUGIN_ENDPOINT", socket)
        .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
        .env("HOME", root)
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept())
        .await
        .unwrap()
        .unwrap();
    let mut stream = LocalStream::Unix(stream);
    write_envelope(
        &mut stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "pay-lightning".into(),
            request_id: 1,
            payload: Some(proto::envelope::Payload::InitializeRequest(
                proto::InitializeRequest {
                    host_protocol_version: PROTOCOL_VERSION,
                    host_capabilities: vec![
                        mesh_llm_plugin::host_capabilities::OPENAI_EXCHANGE.into(),
                        mesh_llm_plugin::operation_binding::CAPABILITY.into(),
                    ],
                    ..Default::default()
                },
            )),
        },
    )
    .await
    .unwrap();
    let response = timeout(Duration::from_secs(10), read_envelope(&mut stream))
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        response.payload,
        Some(proto::envelope::Payload::InitializeResponse(_))
    ));
    // Exercise the installed executable's actual host RPC path. The wallet
    // remains host-owned: terminating this utility must not launch/kill one.
    timeout(Duration::from_secs(10), async {
        for (attempt, expected) in ["", "", "wallet_balance"].into_iter().enumerate() {
            let envelope = read_envelope(&mut stream).await.unwrap();
            let Some(proto::envelope::Payload::RpcRequest(rpc)) = envelope.payload else {
                panic!("expected bound wallet RPC");
            };
            assert_eq!(rpc.method, mesh_llm_plugin::operation_binding::METHOD);
            let request: mesh_llm_plugin::operation_binding::BoundOperationRequest =
                serde_json::from_str(&rpc.params_json).unwrap();
            assert_eq!(request.binding, "test-wallet");
            assert_eq!(request.operation, expected);
            assert_eq!(request.handle.is_none(), expected.is_empty());
            let output = if expected.is_empty() {
                serde_json::json!({"identity":{"wallet_id":"stable-funded-id","provider":"offline","network":"mainnet"},"created":false})
            } else {
                serde_json::json!({"spendable_msat":1000})
            };
            let response = if attempt == 0 { mesh_llm_plugin::operation_binding::BoundOperationResponse::NotDispatched { reason: "provider not initialized".into() } } else { mesh_llm_plugin::operation_binding::BoundOperationResponse::Responded {
                handle: "test-handle".into(), output_json: output.to_string(), is_error: false,
            } };
            write_envelope(&mut stream, &proto::Envelope {
                protocol_version: PROTOCOL_VERSION, plugin_id: "pay-lightning".into(),
                request_id: envelope.request_id,
                payload: Some(proto::envelope::Payload::RpcResponse(proto::RpcResponse {
                    result_json: serde_json::to_string(&response).unwrap(),
                })),
            }).await.unwrap();
        }
    }).await.unwrap();
    assert!(WalletPin::load(&root.join("ledger")).unwrap().is_some());
    (child, stream)
}

#[tokio::test]
async fn utility_sigkill_reopens_host_binding_with_same_identity() {
    let root = tempfile::Builder::new()
        .prefix("ln-life-")
        .tempdir_in("/tmp")
        .unwrap();
    let config = configure(root.path());
    let (mut utility, stream) = launch(root.path(), &config).await;
    let pin = WalletPin::load(&root.path().join("ledger"))
        .unwrap()
        .unwrap();
    utility.kill().await.unwrap();
    utility.wait().await.unwrap();
    drop(stream);
    assert!(!root.path().join("wallet.pid").exists());
    let (mut utility, stream) = launch(root.path(), &config).await;
    assert_eq!(
        WalletPin::load(&root.path().join("ledger")).unwrap(),
        Some(pin)
    );
    utility.kill().await.unwrap();
    utility.wait().await.unwrap();
    drop(stream);
    assert!(!root.path().join("wallet.pid").exists());
    assert!(!root.path().join(".mesh-llm").exists());
}
