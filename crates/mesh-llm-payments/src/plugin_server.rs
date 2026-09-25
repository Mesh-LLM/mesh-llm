//! Serve the payments engine as the `payments.v1` plugin capability.
//!
//! The host registers [`payments_plugin`] as an in-process builtin and reaches
//! the engine through `invoke_operation_by_capability("payments.v1", ...)`, so
//! an external provider can replace it by capability, never by name.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use mesh_llm_payments_types::contract::{
    Empty, FinishRequest, OpError, RoutingBudgetRequest, SettleOutputRequest,
};
use mesh_llm_plugin::{
    InternalRpcPlugin, InternalRpcPluginBuilder, OperationRouter, PluginMetadata, PluginResult,
    capability, operation_with_schema, plugin_server_info,
};
use rmcp::model::CallToolResult;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::control::ControlCommand;
use crate::service::PaymentService;

pub use mesh_llm_payments_types::contract::{CAPABILITY, ops};

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
    add_op(
        &mut router,
        &source,
        ops::CONTROL,
        "Run one local operator command.",
        |service, command: ControlCommand| async move {
            // Durable settlement continues even if the caller goes away.
            tokio::spawn(async move { service.control(command).await }).await?
        },
    );
    add_op(
        &mut router,
        &source,
        ops::ROUTING_BUDGET,
        "Effective payment intent and spendable budget for routing.",
        |service, request: RoutingBudgetRequest| async move {
            Ok(service.routing_budget(request).await)
        },
    );
    add_op(
        &mut router,
        &source,
        ops::RECONCILE,
        "Reconcile charges and list approved requests owing output.",
        |service, _: Empty| async move { service.reconcile().await },
    );
    add_op(
        &mut router,
        &source,
        ops::SETTLE_OUTPUT,
        "Validate and pay a seller output invoice.",
        |service, request: SettleOutputRequest| async move { service.settle_output(request).await },
    );
    add_op(
        &mut router,
        &source,
        ops::FINISH,
        "Mark a request finished.",
        |service, request: FinishRequest| async move { service.ledger.finish(&request.id) },
    );
    router
}

/// Registers one operation: decode `Req`, open the engine, run `handler`.
fn add_op<Req, Res, F, Fut>(
    router: &mut OperationRouter,
    source: &ServiceSource,
    name: &'static str,
    description: &'static str,
    handler: F,
) where
    Req: DeserializeOwned + Send + 'static,
    Res: Serialize + Send + 'static,
    F: Fn(Arc<PaymentService>, Req) -> Fut + Send + Sync + Copy + 'static,
    Fut: Future<Output = anyhow::Result<Res>> + Send + 'static,
{
    let source = Arc::clone(source);
    router.add_raw(
        operation_with_schema(name, description, serde_json::Map::new()),
        move |request, _context| {
            let source = Arc::clone(&source);
            Box::pin(async move {
                let request: Req = match request.arguments() {
                    Ok(request) => request,
                    Err(error) => return op_result::<Res>(Err(anyhow::anyhow!("{error}"))),
                };
                let result = async { handler(source().await?, request).await }.await;
                op_result(result)
            })
        },
    );
}

fn op_result<T: Serialize>(result: anyhow::Result<T>) -> PluginResult<CallToolResult> {
    let internal =
        |error: serde_json::Error| mesh_llm_plugin::PluginError::internal(error.to_string());
    Ok(match result {
        Ok(value) => CallToolResult::structured(serde_json::to_value(value).map_err(internal)?),
        Err(error) => CallToolResult::structured_error(
            serde_json::to_value(OpError {
                message: error.to_string(),
            })
            .map_err(internal)?,
        ),
    })
}
