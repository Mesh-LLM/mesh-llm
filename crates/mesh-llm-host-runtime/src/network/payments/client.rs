//! Host-side client of the `payments.v1` capability. Every per-request payment
//! decision the host makes goes through here, to whichever provider serves
//! the capability; the host never holds the engine for these calls.

use anyhow::{Result, anyhow, bail};
use mesh_llm_payments_types::contract::{CAPABILITY, OpError};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::mesh::Node;
use crate::plugin::PluginManager;

/// Invokes `operation` on the `payments.v1` provider. No timeout: settlement
/// and wallet operations may legitimately outlast the default RPC deadline,
/// and the provider owns their durability.
pub(crate) async fn call<Req: Serialize, Res: DeserializeOwned>(
    plugins: &PluginManager,
    operation: &str,
    request: &Req,
) -> Result<Res> {
    let provider = plugins
        .available_provider_for_capability(CAPABILITY)
        .await?
        .ok_or_else(|| anyhow!("no provider for '{CAPABILITY}'"))?;
    let input = serde_json::to_string(request)?;
    let result = plugins
        .invoke_operation_without_timeout(&provider.plugin_name, operation, &input)
        .await?;
    if result.is_error {
        let message = serde_json::from_str::<OpError>(&result.content_json)
            .map(|error| error.message)
            .unwrap_or_else(|_| format!("payments operation '{operation}' failed"));
        bail!("{message}");
    }
    Ok(serde_json::from_str(&result.content_json)?)
}

/// [`call`] on the node's plugin manager; fails while plugins are starting.
pub(crate) async fn call_node<Req: Serialize, Res: DeserializeOwned>(
    node: &Node,
    operation: &str,
    request: &Req,
) -> Result<Res> {
    let plugins = node
        .plugin_manager()
        .await
        .ok_or_else(|| anyhow!("payments are not available yet"))?;
    call(&plugins, operation, request).await
}
