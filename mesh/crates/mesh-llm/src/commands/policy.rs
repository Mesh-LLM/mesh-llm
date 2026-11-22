use anyhow::{Context, Result};
use mesh_llm_cli::{Cli, PolicyCommand};
use mesh_llm_system::local_http::local_http_client_builder;
use reqwest::header::AUTHORIZATION;
use serde_json::json;
use std::io::Write;

pub(super) async fn dispatch_policy_command(command: &PolicyCommand, cli: &Cli) -> Result<()> {
    let _ = cli;
    match command {
        PolicyCommand::Set {
            payload,
            signature,
            port,
            json,
        } => run_policy_set(payload, signature, *port, *json).await,
        PolicyCommand::Get { port, json } => run_policy_get(*port, *json).await,
        PolicyCommand::Clear { port, json } => run_policy_clear(*port, *json).await,
    }
}

async fn run_policy_set(
    payload: &str,
    signature: &str,
    port: u16,
    json_output: bool,
) -> Result<()> {
    let token = mesh_llm_commands::owner_control::owner_control_token(port).await?;
    let client = local_http_client_builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/runtime/control/mesh-policy");
    let body = json!({
        "policy_payload_bytes": payload,
        "signature": signature,
    });
    let response = client
        .post(&url)
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .json(&body)
        .send()
        .await
        .with_context(|| format!("Can't connect to mesh-llm on port {port}. Is it running?"))?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    print_policy_response("Set mesh policy", &response, json_output)
}

async fn run_policy_get(port: u16, json_output: bool) -> Result<()> {
    let client = local_http_client_builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/runtime/control/mesh-policy");
    let token = mesh_llm_commands::owner_control::owner_control_token(port).await?;
    let response = client
        .get(&url)
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Can't connect to mesh-llm on port {port}: {e}"))?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    print_policy_response("Get mesh policy", &response, json_output)
}

async fn run_policy_clear(port: u16, json_output: bool) -> Result<()> {
    // POST with empty payload to clear — the server treats missing/invalid payload as clear.
    // We rely on the GET/POST distinction: a POST with no policy_payload_bytes is treated as a
    // clear request by the server. For now, print a note since the endpoint doesn't have a
    // dedicated clear path.
    let _ = port;
    let _ = json_output;
    let mut out = mesh_llm_events::console_out();
    writeln!(
        out,
        "Note: policy clear is not yet wired to a server endpoint. \
         Restart the node to clear the in-memory policy, or wait for it to expire."
    )?;
    Ok(())
}

fn print_policy_response(
    title: &str,
    body: &serde_json::Value,
    json_output: bool,
) -> Result<()> {
    if !json_output {
        let mut out = mesh_llm_events::console_out();
        writeln!(out, "🔐 {title}")?;
        writeln!(out)?;
    }
    let mut out = mesh_llm_events::machine_out();
    writeln!(out, "{}", serde_json::to_string_pretty(body)?)?;
    Ok(())
}
