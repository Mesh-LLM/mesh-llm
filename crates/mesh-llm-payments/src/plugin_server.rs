//! Serve the payments engine as the `payments.v1` plugin capability.
//!
//! The host registers [`payments_plugin`] as an in-process builtin and reaches
//! the engine through `invoke_operation_by_capability("payments.v1", ...)`, so
//! an external provider can replace it by capability, never by name.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use mesh_llm_plugin::{
    InternalRpcPlugin, InternalRpcPluginBuilder, OperationRouter, PluginMetadata, PluginResult,
    capability, operation_with_schema, plugin_server_info,
};
use rmcp::model::CallToolResult;
use serde::Serialize;

use crate::control::ControlCommand;
use crate::service::PaymentService;

/// Capability name the host resolves the payments engine by.
pub const CAPABILITY: &str = "payments.v1";

/// Operation names in the `payments.v1` contract.
pub mod ops {
    /// Run one local operator [`crate::control::ControlCommand`].
    pub const CONTROL: &str = "control";
}

/// Opens (or returns the already open) engine. The host owns where the ledger
/// lives and which wallet it talks to, so it supplies this.
pub type ServiceSource = Arc<
    dyn Fn() -> Pin<Box<dyn Future<Output = anyhow::Result<Arc<PaymentService>>> + Send>>
        + Send
        + Sync,
>;

/// Builds the `payments.v1` plugin over `source`.
pub fn payments_plugin(
    plugin_name: impl Into<String>,
    version: impl Into<String>,
    source: ServiceSource,
) -> InternalRpcPlugin {
    let version = version.into();
    InternalRpcPluginBuilder::new(PluginMetadata::new(
        plugin_name.into(),
        version.clone(),
        plugin_server_info(
            "mesh-payments",
            version,
            "Mesh payments",
            "Inference settlement: ledger, pricing, budgets and payment gates.",
            Some("Internal payments capability for the mesh-llm host. Not intended for direct agent use."),
        ),
    ))
    .with_capabilities(vec![CAPABILITY.into()])
    .with_manifest(mesh_llm_plugin::plugin_manifest![capability(CAPABILITY)])
    .with_operation_router(operation_router(source))
    .build()
}

fn operation_router(source: ServiceSource) -> OperationRouter {
    let mut router = OperationRouter::new();
    router.add_raw(
        operation_with_schema(
            ops::CONTROL,
            "Run one local operator command.",
            serde_json::Map::new(),
        ),
        move |request, _context| {
            let source = Arc::clone(&source);
            Box::pin(async move {
                let command: ControlCommand = match request.arguments() {
                    Ok(command) => command,
                    Err(error) => return op_result::<()>(Err(anyhow::anyhow!("{error}"))),
                };
                let result = async {
                    let service = source().await?;
                    // Durable settlement continues even if the caller goes away.
                    tokio::spawn(async move { service.control(command).await }).await?
                }
                .await;
                op_result(result)
            })
        },
    );
    router
}

fn op_result<T: Serialize>(result: anyhow::Result<T>) -> PluginResult<CallToolResult> {
    let internal =
        |error: serde_json::Error| mesh_llm_plugin::PluginError::internal(error.to_string());
    Ok(match result {
        Ok(value) => CallToolResult::structured(serde_json::to_value(value).map_err(internal)?),
        Err(error) => CallToolResult::structured_error(serde_json::json!({
            "message": error.to_string()
        })),
    })
}
