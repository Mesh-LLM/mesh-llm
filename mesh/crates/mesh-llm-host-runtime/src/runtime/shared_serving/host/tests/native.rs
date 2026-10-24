//! Opt-in composed-product smoke. Requires a just-built release product and
//! a local causal GGUF; it never downloads artifacts or contacts a public mesh.

use super::*;
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::path::Path;
use std::process::Stdio;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires SHARED_SERVING_BINARY, SHARED_SERVING_MODEL and a matching packaged native runtime"]
async fn composed_product_serves_two_meshes_from_one_native_model() -> Result<()> {
    let binary =
        std::env::var("SHARED_SERVING_BINARY").context("set the just-built release binary")?;
    let model = std::env::var("SHARED_SERVING_MODEL").context("set a local causal GGUF")?;
    let gpu_layers = std::env::var("SHARED_SERVING_GPU_LAYERS").unwrap_or_else(|_| "0".into());
    ensure!(
        gpu_layers.parse::<i32>().is_ok_and(|layers| layers >= -1),
        "SHARED_SERVING_GPU_LAYERS must be -1 or a non-negative integer"
    );
    ensure!(
        Path::new(&binary).is_file() && Path::new(&model).is_file(),
        "smoke inputs must exist"
    );
    let fixture = Fixture::new().await;
    // Reuse only the isolated seed meshes and their token files. The recording
    // attachments are shut down before the real product enters those meshes.
    fixture.host.shutdown().await?;
    let state = fixture.directory.path().join("composed-host");
    // Global CLI initialization may create data/cache parents before dispatch.
    // Establish the private host boundary before pointing those paths inside it.
    private_directory(&state)?;
    let config = fixture.directory.path().join("composed.toml");
    std::fs::write(
        &config,
        "[logging]\nenabled = false\n[analytics]\nenabled = false\n",
    )?;
    let port_reservation = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = port_reservation.local_addr()?.port();
    drop(port_reservation);
    let alpha = format!(
        "alpha={}",
        fixture.directory.path().join("alpha.token").display()
    );
    let beta = format!(
        "beta={}",
        fixture.directory.path().join("beta.token").display()
    );
    let mut process = tokio::process::Command::new(binary)
        .arg("--config")
        .arg(&config)
        .args([
            "serve-shared",
            "--model-id",
            "shared-test",
            "--model-path",
            &model,
            "--membership",
            &alpha,
            "--membership",
            &beta,
            "--context-size",
            "512",
            "--concurrency",
            "1",
            "--queue-capacity",
            "4",
            "--membership-queue-capacity",
            "2",
            "--gpu-layers",
            &gpu_layers,
            "--lan-only",
            "--bind-ip",
            "127.0.0.1",
            "--management-port",
            &port.to_string(),
            "--state-dir",
        ])
        .arg(&state)
        .env("MESH_LLM_ANALYTICS", "0")
        .env("MESH_LLM_DATA_DIR", state.join("data"))
        .env("MESH_LLM_RUNTIME_ROOT", state.join("runtime"))
        .env(
            "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
            state.join("native-cache"),
        )
        .env("HF_HUB_CACHE", state.join("hf-cache"))
        .env("HF_XET_CACHE", state.join("xet-cache"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()?;
    let base = format!("http://127.0.0.1:{port}/api/shared-serving");
    let client = reqwest::Client::builder()
        // Reconnect owns a bounded 30-second mesh admission attempt. Do not
        // cancel it with a shorter test-client deadline while QUIC recovers.
        .timeout(Duration::from_secs(45))
        .build()?;
    let result = exercise_product(&fixture, &client, &base, &state, &mut process).await;
    #[cfg(unix)]
    let result = match result {
        Ok(()) => graceful_stop(&mut process).await,
        Err(error) => Err(error),
    };
    process.kill().await.ok();
    process.wait().await.ok();
    fixture.close().await;
    result
}

#[cfg(unix)]
async fn graceful_stop(process: &mut tokio::process::Child) -> Result<()> {
    let id = process
        .id()
        .context("shared server exited before shutdown")?;
    let id = libc::pid_t::try_from(id)?;
    // SAFETY: this is the positive PID of the live child owned by this test.
    ensure!(
        unsafe { libc::kill(id, libc::SIGTERM) } == 0,
        "cannot signal the test server"
    );
    let status = tokio::time::timeout(Duration::from_secs(40), process.wait()).await??;
    ensure!(
        status.success(),
        "shared host did not shut down cleanly: {status}"
    );
    Ok(())
}

async fn exercise_product(
    fixture: &Fixture,
    client: &reqwest::Client,
    base: &str,
    state: &Path,
    process: &mut tokio::process::Child,
) -> Result<()> {
    let (token, status) = tokio::time::timeout(Duration::from_secs(180), async {
        loop {
            if let Some(exit) = process.try_wait()? {
                anyhow::bail!("composed shared server exited during startup: {exit}");
            }
            if let Ok(token) = std::fs::read_to_string(state.join("management.token"))
                && let Ok(response) = client.get(base).bearer_auth(&token).send().await
                && response.status().is_success()
            {
                let status: Value = response.json().await?;
                ensure!(
                    !status["memberships"].as_array().is_some_and(|members| {
                        members.iter().any(|member| member["phase"] == "failed")
                    }),
                    "composed membership startup failed: {status}"
                );
                if status["memberships"].as_array().is_some_and(|members| {
                    members.len() == 2
                        && members.iter().all(|member| member["phase"] == "connected")
                }) {
                    break Ok::<_, anyhow::Error>((token, status));
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .context("composed shared server did not become ready")??;
    ensure!(
        status["loaded_models"] == 1 && status["shared_concurrency"] == 1,
        "model owner or execution budget was multiplied"
    );
    let runtime_id = status["runtime_id"]
        .as_str()
        .context("native runtime identity")?
        .to_owned();
    ensure!(
        status["native_lane_count"] == 1,
        "native execution lanes were multiplied"
    );
    let members = status["memberships"]
        .as_array()
        .context("membership status")?;
    ensure!(
        members[0]["mesh_id"] != members[1]["mesh_id"],
        "mesh identities collapsed"
    );
    let endpoints = [
        members[0]["endpoint_id"]
            .as_str()
            .context("alpha identity")?
            .parse()?,
        members[1]["endpoint_id"]
            .as_str()
            .context("beta identity")?
            .parse()?,
    ];
    let (a, b) = tokio::join!(
        native_request(&fixture.seeds[0], endpoints[0]),
        native_request(&fixture.seeds[1], endpoints[1])
    );
    a?;
    b?;
    if let Some(binary) = std::env::var_os("SHARED_SERVING_RELEASED_PEER") {
        super::released_peer::exercise(Path::new(&binary), fixture.directory.path()).await?;
    }
    for action in ["pause", "leave"] {
        let response = client
            .post(format!("{base}/memberships/alpha/{action}"))
            .bearer_auth(&token)
            .send()
            .await?;
        ensure!(response.status() == 204, "membership {action} failed");
        native_request(&fixture.seeds[1], endpoints[1]).await?;
    }
    let response = client
        .post(format!("{base}/memberships/alpha/reconnect"))
        .bearer_auth(&token)
        .send()
        .await?;
    ensure!(response.status() == 204, "membership reconnect failed");
    native_request(&fixture.seeds[0], endpoints[0]).await?;
    native_request(&fixture.seeds[1], endpoints[1]).await?;
    let status: Value = client
        .get(base)
        .bearer_auth(&token)
        .send()
        .await?
        .json()
        .await?;
    ensure!(
        status["loaded_models"] == 1 && status["runtime_id"] == runtime_id,
        "attachment lifecycle replaced or dropped the shared native runtime"
    );
    ensure!(
        status["active_requests"] == 0 && status["queued_requests"] == 0,
        "requests did not drain"
    );
    ensure!(
        fixture.backend.scopes.lock().unwrap().is_empty(),
        "request reached the test backend, not the native product"
    );
    Ok(())
}

async fn native_request(seed: &Node, endpoint: iroh::EndpointId) -> Result<()> {
    let body = r#"{"model":"shared-test","prompt":"The capital of France is","max_tokens":8,"temperature":0,"prompt_cache_key":"same-key","user":"same-user"}"#;
    let raw = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let (mut send, mut recv) = seed.open_http_tunnel(endpoint).await?;
    send.write_all(raw.as_bytes()).await?;
    send.finish()?;
    let response = tokio::time::timeout(Duration::from_secs(90), recv.read_to_end(65536)).await??;
    let response = String::from_utf8(response)?;
    ensure!(
        response.starts_with("HTTP/1.1 200"),
        "native inference failed: {response}"
    );
    let payload = response
        .split_once("\r\n\r\n")
        .context("native HTTP body")?
        .1;
    let parsed: Value = serde_json::from_str(payload).context("native JSON response")?;
    ensure!(
        parsed["choices"][0]["text"]
            .as_str()
            .is_some_and(|text| !text.trim().is_empty()),
        "native model produced no text"
    );
    ensure!(
        parsed["usage"]["completion_tokens"]
            .as_u64()
            .is_some_and(|tokens| tokens > 0),
        "missing native token accounting"
    );
    Ok(())
}
