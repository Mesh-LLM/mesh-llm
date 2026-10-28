//! Real utility and synthetic wallet child lifecycle; never accesses a real wallet.
#![cfg(all(unix, feature = "offline-mock"))]
use mesh_llm_plugin::{LocalStream, PROTOCOL_VERSION, proto, read_envelope, write_envelope};
use mesh_llm_wallet::provisioning::WalletPin;
use std::{path::Path, time::Duration};
use tokio::{
    process::{Child, Command},
    time::timeout,
};

fn configure(root: &Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let wallet = root.join("wallet.sh");
    std::fs::write(&wallet, format!(
        "#!/bin/sh\necho $$ > '{}'\nexport MESH_LLM_OFFLINE_WALLET_CHILD=1\nexport MESH_LLM_OFFLINE_NETWORK_DIR='{}'\nexec '{}'\n",
        root.join("wallet.pid").display(), root.join("network").display(),
        env!("CARGO_BIN_EXE_pay-lightning-mock"),
    )).unwrap();
    std::fs::set_permissions(&wallet, std::fs::Permissions::from_mode(0o700)).unwrap();
    let config = root.join("config.json");
    std::fs::write(
        &config,
        serde_json::to_vec(&serde_json::json!({
            "payment": {"data_dir":root.join("ledger")},
            "wallet_plugin":"offline-wallet", "wallet_executable":wallet,
            "wallet_directory":root.join("wallet"), "allow_wallet_provisioning":true,
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
        .env("MESH_LLM_PLUGIN_CONFIG", config)
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
    timeout(Duration::from_secs(10), async {
        while !root.join("wallet.pid").exists()
            || WalletPin::load(&root.join("ledger")).unwrap().is_none()
        {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    (child, stream)
}

async fn wallet_exits(root: &Path) {
    let pid = std::fs::read_to_string(root.join("wallet.pid")).unwrap();
    timeout(Duration::from_secs(10), async {
        loop {
            let status = Command::new("ps")
                .args(["-p", pid.trim(), "-o", "stat="])
                .output()
                .await
                .unwrap();
            let state = String::from_utf8_lossy(&status.stdout);
            if !status.status.success() || state.trim().starts_with('Z') {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("wallet child survived utility exit");
}

#[tokio::test]
async fn utility_sigkill_closes_wallet_ipc_and_restarts_same_identity() {
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
    wallet_exits(root.path()).await;
    let (mut utility, stream) = launch(root.path(), &config).await;
    assert_eq!(
        WalletPin::load(&root.path().join("ledger")).unwrap(),
        Some(pin)
    );
    utility.kill().await.unwrap();
    utility.wait().await.unwrap();
    drop(stream);
    wallet_exits(root.path()).await;
    assert!(!root.path().join(".mesh-llm").exists());
}
