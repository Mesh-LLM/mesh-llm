//! HTTP transport for owner-control runtime commands.

use anyhow::{Context, Result};

pub(super) async fn post_runtime_payload(
    port: u16,
    path: &str,
    body: &serde_json::Value,
) -> Result<serde_json::Value> {
    let token = mesh_llm_commands::owner_control::owner_control_token(port).await?;
    post_with_token(port, path, body, &token).await
}

pub(super) async fn post_with_token(
    port: u16,
    path: &str,
    body: &serde_json::Value,
    token: &str,
) -> Result<serde_json::Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let url = format!("http://127.0.0.1:{port}{path}");
    let response = client
        .post(&url)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
        .json(body)
        .send()
        .await
        .with_context(|| {
            format!("Can't connect to mesh-llm console on port {port}. Is it running?")
        })?;
    let status = response.status();
    let body = response
        .json::<serde_json::Value>()
        .await
        .unwrap_or(serde_json::Value::Null);
    if status.is_success() {
        Ok(body)
    } else {
        let reason = body
            .get("error")
            .and_then(|value| value.get("message").or(Some(value)))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown error");
        anyhow::bail!("{reason}");
    }
}
