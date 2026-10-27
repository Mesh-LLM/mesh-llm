use std::io::{self, IsTerminal, Write};

use anyhow::{Context, Result, bail};
use mesh_llm_cli::KvCacheCommand;
use serde_json::{Value, json};

const KV_CACHE_HTTP_TIMEOUT_SECS: u64 = 60;

pub async fn dispatch_kv_cache_command(command: &KvCacheCommand) -> Result<()> {
    match command {
        KvCacheCommand::Status {
            endpoints,
            port,
            json,
        } => {
            let value = if endpoints.is_empty() {
                request(
                    *port,
                    reqwest::Method::GET,
                    "/api/runtime/kv-cache",
                    None,
                    false,
                )
                .await?
            } else {
                remote_request(*port, endpoints, "status", None, None).await?
            };
            print_response(&value, *json)
        }
        KvCacheCommand::Prune {
            target,
            model_identity,
            yes,
            endpoints,
            port,
            json: json_output,
        } => {
            confirm_destructive("prune inactive disk prompt-cache entries", *yes)?;
            let target_bytes = target
                .as_deref()
                .map(mesh_llm_config::parse_iec_size)
                .transpose()
                .context("invalid --target")?;
            let body = json!({
                "target_bytes": target_bytes,
                "model_identity": model_identity,
            });
            let value = if endpoints.is_empty() {
                request(
                    *port,
                    reqwest::Method::POST,
                    "/api/runtime/kv-cache/prune",
                    Some(body),
                    true,
                )
                .await?
            } else {
                remote_request(
                    *port,
                    endpoints,
                    "prune",
                    target_bytes,
                    model_identity.clone(),
                )
                .await?
            };
            print_response(&value, *json_output)
        }
        KvCacheCommand::Clear {
            model_identity,
            yes,
            endpoints,
            port,
            json: json_output,
        } => {
            confirm_destructive("clear inactive disk prompt-cache entries", *yes)?;
            let value = if endpoints.is_empty() {
                request(
                    *port,
                    reqwest::Method::DELETE,
                    "/api/runtime/kv-cache",
                    Some(json!({ "model_identity": model_identity })),
                    true,
                )
                .await?
            } else {
                remote_request(*port, endpoints, "clear", None, model_identity.clone()).await?
            };
            print_response(&value, *json_output)
        }
    }
}

async fn remote_request(
    port: u16,
    endpoints: &[String],
    operation: &str,
    target_bytes: Option<u64>,
    model_identity: Option<String>,
) -> Result<Value> {
    request(
        port,
        reqwest::Method::POST,
        "/api/runtime/control/kv-cache",
        Some(json!({
            "endpoints": endpoints,
            "operation": operation,
            "target_bytes": target_bytes,
            "model_identity": model_identity,
        })),
        true,
    )
    .await
}

/// Issues a loopback management request. Owner-control routes additionally
/// require the host's owner-control credential, fetched once from the
/// loopback token endpoint when `owner_control` is set.
async fn request(
    port: u16,
    method: reqwest::Method,
    path: &str,
    body: Option<Value>,
    owner_control: bool,
) -> Result<Value> {
    let client = reqwest::Client::builder()
        // The runtime bounds a remote batch at 45 seconds and returns one
        // result per requested endpoint, including explicit timeout receipts.
        // Leave transport and JSON-decoding margin outside that server bound.
        .timeout(std::time::Duration::from_secs(KV_CACHE_HTTP_TIMEOUT_SECS))
        .build()?;
    let url = format!("http://127.0.0.1:{port}{path}");
    let mut request = client.request(method, &url);
    if owner_control {
        let token = super::owner_control::owner_control_token(port).await?;
        request = request.header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"));
    }
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("request disk prompt-cache endpoint {url}"))?;
    let status = response.status();
    let value = response
        .json::<Value>()
        .await
        .with_context(|| format!("decode disk prompt-cache response from {url}"))?;
    if !status.is_success() {
        bail!(
            "disk prompt-cache request failed ({status}): {}",
            value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        );
    }
    Ok(value)
}

fn confirm_destructive(action: &str, yes: bool) -> Result<()> {
    if yes {
        return Ok(());
    }
    if !io::stdin().is_terminal() {
        bail!("{action} requires --yes when stdin is not interactive");
    }
    let mut err = mesh_llm_events::console_err();
    let _ = write!(err, "{action}? [y/N] ");
    let _ = err.flush();
    let mut response = String::new();
    io::stdin().read_line(&mut response)?;
    if !matches!(response.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        bail!("operation cancelled");
    }
    Ok(())
}

fn print_response(value: &Value, json_output: bool) -> Result<()> {
    if json_output {
        let mut machine = mesh_llm_events::machine_out();
        writeln!(machine, "{}", serde_json::to_string(value)?)?;
        return Ok(());
    }
    let mut err = mesh_llm_events::console_err();
    if let Some(results) = value.get("results").and_then(Value::as_array) {
        for result in results {
            let node = result
                .get("target_node_id")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            if let Some(error) = result.get("error").filter(|value| !value.is_null()) {
                writeln!(
                    err,
                    "Node {node}: error: {}",
                    error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                )?;
            } else if let Some(freed) = result.get("freed_bytes").and_then(Value::as_u64) {
                writeln!(err, "Node {node}: freed {freed} bytes")?;
            } else {
                let state = result
                    .get("status")
                    .and_then(|status| status.get("effective"))
                    .and_then(|effective| effective.get("state"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                writeln!(err, "Node {node}: {state}")?;
            }
        }
        return Ok(());
    }
    if let Some(freed) = value.get("freed_bytes").and_then(Value::as_u64) {
        writeln!(err, "Freed {freed} bytes")?;
        return Ok(());
    }
    let configured = &value["configured"];
    let effective = &value["effective"];
    writeln!(
        err,
        "Disk prompt cache: {} ({})",
        effective["state"].as_str().unwrap_or("unknown"),
        configured["mode"].as_str().unwrap_or("unknown")
    )?;
    writeln!(
        err,
        "Root: {}",
        configured["directory"].as_str().unwrap_or("unknown")
    )?;
    if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
        writeln!(
            err,
            "Used: {} / {} bytes",
            usage["used_bytes"].as_u64().unwrap_or(0),
            usage["budget_bytes"].as_u64().unwrap_or(0)
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owner_control::tests::serve_in_order;

    fn authorization(head: &[String]) -> Option<String> {
        head.iter().find_map(|line| {
            line.split_once(':')
                .filter(|(name, _)| name.trim().eq_ignore_ascii_case("authorization"))
                .map(|(_, value)| value.trim().to_string())
        })
    }

    #[tokio::test]
    async fn prune_presents_owner_control_credential() {
        let stub = serve_in_order(vec![
            (200, r#"{"token":"kv-secret"}"#.to_string()),
            (200, r#"{"activity":{}}"#.to_string()),
        ]);
        let command = KvCacheCommand::Prune {
            target: None,
            model_identity: None,
            yes: true,
            endpoints: Vec::new(),
            port: stub.port,
            json: true,
        };
        dispatch_kv_cache_command(&command)
            .await
            .expect("prune with the fetched credential");
        let heads = stub.request_heads();
        assert_eq!(
            heads.len(),
            2,
            "one token fetch plus the prune request: {heads:?}"
        );
        assert!(heads[0][0].starts_with("GET /api/runtime/control/owner-control-token"));
        assert!(heads[1][0].starts_with("POST /api/runtime/kv-cache/prune"));
        assert_eq!(
            authorization(&heads[1]).as_deref(),
            Some("Bearer kv-secret")
        );
    }

    #[tokio::test]
    async fn clear_presents_owner_control_credential() {
        let stub = serve_in_order(vec![
            (200, r#"{"token":"kv-secret"}"#.to_string()),
            (200, r#"{"activity":{}}"#.to_string()),
        ]);
        let command = KvCacheCommand::Clear {
            model_identity: None,
            yes: true,
            endpoints: Vec::new(),
            port: stub.port,
            json: true,
        };
        dispatch_kv_cache_command(&command)
            .await
            .expect("clear with the fetched credential");
        let heads = stub.request_heads();
        assert_eq!(
            heads.len(),
            2,
            "one token fetch plus the clear request: {heads:?}"
        );
        assert!(heads[1][0].starts_with("DELETE /api/runtime/kv-cache"));
        assert_eq!(
            authorization(&heads[1]).as_deref(),
            Some("Bearer kv-secret")
        );
    }

    #[tokio::test]
    async fn status_does_not_present_owner_control_credential() {
        let stub = serve_in_order(vec![(200, r#"{"activity":{}}"#.to_string())]);
        let command = KvCacheCommand::Status {
            endpoints: Vec::new(),
            port: stub.port,
            json: true,
        };
        dispatch_kv_cache_command(&command)
            .await
            .expect("loopback status read");
        let heads = stub.request_heads();
        assert_eq!(
            heads.len(),
            1,
            "status must not fetch the credential: {heads:?}"
        );
        assert!(heads[0][0].starts_with("GET /api/runtime/kv-cache"));
        assert!(
            authorization(&heads[0]).is_none(),
            "no credential on the status read"
        );
    }
}
