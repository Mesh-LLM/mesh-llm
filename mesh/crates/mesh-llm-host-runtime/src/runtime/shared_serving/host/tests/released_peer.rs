//! Optional released-binary compatibility proof for the composed-product test.

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

pub(super) async fn exercise(binary: &Path, directory: &Path) -> Result<()> {
    ensure!(binary.is_file(), "released client binary must exist");
    let state = directory.join("released-client");
    std::fs::create_dir(&state)?;
    let config = state.join("config.toml");
    std::fs::write(
        &config,
        "[logging]\nenabled = false\n[analytics]\nenabled = false\n",
    )?;
    let api = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let console = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = api.local_addr()?.port();
    let console_port = console.local_addr()?.port();
    drop((api, console));
    let mut process = tokio::process::Command::new(binary)
        .arg("--config")
        .arg(&config)
        .args(["client", "--join-file"])
        .arg(directory.join("alpha.token"))
        .args([
            "--port",
            &port.to_string(),
            "--console",
            &console_port.to_string(),
            "--headless",
            "--no-default-plugins",
            "--disable-iroh-relays",
            "--mesh-discovery-mode",
            "mdns",
            "--bind-ip",
            "127.0.0.1",
            "--trust-policy",
            "off",
            "--owner-key",
        ])
        // An explicit absent owner file avoids inspecting the operator's
        // keystore or asking their OS keychain for credentials.
        .arg(state.join("absent-owner.json"))
        .env("MESH_LLM_ANALYTICS", "0")
        .env("MESH_LLM_NO_DEFAULT_PLUGINS", "1")
        .env("MESH_LLM_DATA_DIR", state.join("data"))
        .env("MESH_LLM_RUNTIME_ROOT", state.join("runtime"))
        .env("HF_HUB_CACHE", state.join("hf-cache"))
        .env("HF_XET_CACHE", state.join("xet-cache"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()?;
    let result = request_from_released_client(port, &mut process).await;
    process.kill().await.ok();
    process.wait().await.ok();
    result
}

async fn request_from_released_client(
    port: u16,
    process: &mut tokio::process::Child,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(90))
        .build()?;
    let base = format!("http://127.0.0.1:{port}/v1");
    tokio::time::timeout(Duration::from_secs(90), async {
        loop {
            if let Some(exit) = process.try_wait()? {
                anyhow::bail!("released client exited before discovery: {exit}");
            }
            if let Ok(response) = client.get(format!("{base}/models")).send().await
                && response.status().is_success()
            {
                let models: Value = response.json().await?;
                if models["data"]
                    .as_array()
                    .is_some_and(|models| models.iter().any(|model| model["id"] == "shared-test"))
                {
                    break Ok::<_, anyhow::Error>(());
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    })
    .await
    .context("released client did not discover the shared model")??;
    let response = client
        .post(format!("{base}/completions"))
        .json(&json!({
            "model": "shared-test", "prompt": "The capital of France is",
            "max_tokens": 8, "temperature": 0,
        }))
        .send()
        .await?;
    let status = response.status();
    let body: Value = response.json().await?;
    ensure!(
        status.is_success(),
        "released client inference failed: {status}: {body}"
    );
    ensure!(
        body["choices"][0]["text"]
            .as_str()
            .is_some_and(|text| !text.trim().is_empty()),
        "released client did not receive native output"
    );
    ensure!(
        body["usage"]["completion_tokens"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "released client response has no native completion accounting"
    );
    Ok(())
}
