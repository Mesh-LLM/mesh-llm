use mesh_llm_payments::control::ControlCommand;
use mesh_llm_payments::plugin_server::{CAPABILITY, ops};
use tokio::net::TcpStream;

use super::super::{
    MeshApi,
    http::{respond_error, respond_json},
};

/// The API boundary classifies every wallet route as trusted-local before
/// dispatch. Do not mount this handler on the peer inference transport.
pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    let mut body: serde_json::Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return respond_error(stream, 400, "invalid wallet command").await,
    };
    if let Some(expected) = body.as_object_mut().and_then(|v| v.remove("expected_pid"))
        && expected.as_u64() != Some(u64::from(std::process::id()))
    {
        return respond_error(
            stream,
            409,
            "wallet process changed; refresh before retrying",
        )
        .await;
    }
    let node = state.inner.lock().await.node.clone();
    if let Some(expected) = body
        .as_object_mut()
        .and_then(|v| v.remove("expected_directory"))
    {
        let actual = node.config_state.lock().await.payment_directory();
        let matches = expected.as_str().is_some_and(|expected| {
            let expected = std::path::Path::new(expected);
            match (
                expected.parent().and_then(|p| p.canonicalize().ok()),
                actual.parent().and_then(|p| p.canonicalize().ok()),
            ) {
                (Some(a), Some(b)) => a == b && expected.file_name() == actual.file_name(),
                _ => false,
            }
        });
        if !matches {
            return respond_error(
                stream,
                409,
                "wallet data directory does not match this runtime",
            )
            .await;
        }
    }
    let command: ControlCommand = match serde_json::from_value(body) {
        Ok(command) => command,
        Err(_) => return respond_error(stream, 400, "invalid wallet command").await,
    };
    let Some(plugins) = node.plugin_manager().await else {
        return respond_error(stream, 503, "payments are not available yet").await;
    };
    // The engine spawns durable settlement itself, so it continues even if
    // this HTTP connection disappears.
    let input = serde_json::to_string(&command)?;
    let result = invoke_control(&plugins, &input).await;
    match result {
        Ok(value) => respond_json(stream, 200, &value).await,
        Err(error) => respond_error(stream, 400, &error.to_string()).await,
    }
}

/// Runs `control` on whichever provider serves `payments.v1`. No timeout: a
/// funding or send command may legitimately outlast the default RPC deadline.
async fn invoke_control(
    plugins: &crate::plugin::PluginManager,
    input: &str,
) -> anyhow::Result<serde_json::Value> {
    let provider = plugins
        .available_provider_for_capability(CAPABILITY)
        .await?
        .ok_or_else(|| anyhow::anyhow!("no provider for '{CAPABILITY}'"))?;
    let result = plugins
        .invoke_operation_without_timeout(&provider.plugin_name, ops::CONTROL, input)
        .await?;
    control_output(result)
}

/// Decodes a `payments.v1` `control` result; an operation error carries
/// `{"message": ...}`.
fn control_output(result: crate::plugin::ToolCallResult) -> anyhow::Result<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(&result.content_json)?;
    if result.is_error {
        let message = value
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("payments operation failed");
        anyhow::bail!("{message}");
    }
    Ok(value)
}
