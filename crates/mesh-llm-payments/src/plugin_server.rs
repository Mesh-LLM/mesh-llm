//! Serve the payments engine as the `payments.v1` plugin capability.
//!
//! The host registers [`payments_plugin`] as an in-process builtin and reaches
//! the engine through `invoke_operation_by_capability("payments.v1", ...)`, so
//! an external provider can replace it by capability, never by name.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use mesh_llm_payments_types::contract::{
    ArrivalResponse, AuthorizeRequest, CancelRequest, Empty, FinishRequest, IdRequest,
    InvoiceRequest, OpError, OutputReceivableResponse, PayInputRequest, RecordDeliveredRequest,
    RoutingBudgetRequest, ServeBeginRequest, ServeFinishRequest, ServeInputInvoiceRequest,
    SettleOutputRequest,
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

impl mesh_llm_payments_types::engine::PaymentsEngine for PaymentService {
    fn pricing(
        &self,
    ) -> anyhow::Result<std::collections::BTreeMap<String, crate::pricing::Pricing>> {
        self.ledger.pricing()
    }

    fn has_wallet(&self) -> bool {
        PaymentService::has_wallet(self)
    }

    fn into_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync> {
        self
    }
}

/// This engine as a host [`PaymentsEngineProvider`]: the shipped binary
/// installs it; the host itself never links this crate.
///
/// [`PaymentsEngineProvider`]: mesh_llm_payments_types::engine::PaymentsEngineProvider
pub struct EngineProvider;

impl mesh_llm_payments_types::engine::PaymentsEngineProvider for EngineProvider {
    fn open(
        &self,
        directory: &std::path::Path,
        wallet: Arc<dyn crate::provisioning::WalletFactory>,
    ) -> anyhow::Result<Arc<dyn mesh_llm_payments_types::engine::PaymentsEngine>> {
        Ok(Arc::new(PaymentService::with_factory(directory, wallet)?))
    }

    fn serve(
        &self,
        plugin_name: &str,
        version: &str,
        source: mesh_llm_payments_types::engine::EngineSource,
        stream: mesh_llm_plugin::LocalStream,
    ) -> mesh_llm_payments_types::engine::BoxFuture<anyhow::Result<()>> {
        let source: ServiceSource = Arc::new(move || {
            let source = Arc::clone(&source);
            Box::pin(async move {
                source()
                    .await?
                    .into_any()
                    .downcast::<PaymentService>()
                    .map_err(|_| anyhow::anyhow!("payments engine is not this provider's"))
            })
        });
        let plugin = payments_plugin(plugin_name, version, source);
        Box::pin(mesh_llm_plugin::PluginRuntime::run_with_stream(
            plugin, stream,
        ))
    }
}

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
        |service, request: FinishRequest| async move {
            service.ledger.finish(&request.id).map(|()| Empty {})
        },
    );
    add_op(
        &mut router,
        &source,
        ops::PAYMENT_INTENT,
        "The operator's profile payment intent.",
        |service, _: Empty| async move { service.ledger.payment_intent() },
    );
    add_op(
        &mut router,
        &source,
        ops::PREFETCH,
        "Start reading the balance for a request.",
        |service, request: IdRequest| async move { service.prefetch(request.id).map(|()| Empty {}) },
    );
    add_op(
        &mut router,
        &source,
        ops::AUTHORIZE,
        "Propose and approve request terms.",
        |service, request: AuthorizeRequest| async move {
            service.authorize(request).await.map(|()| Empty {})
        },
    );
    add_op(
        &mut router,
        &source,
        ops::CANCEL,
        "Release a request that has not started paying.",
        |service, request: CancelRequest| async move { service.cancel(request).map(|()| Empty {}) },
    );
    add_op(
        &mut router,
        &source,
        ops::PAY_INPUT,
        "Pay the input invoice of an authorized request.",
        |service, request: PayInputRequest| async move {
            // Durable submission continues even if the caller goes away.
            tokio::spawn(async move { service.pay_input(request).await }).await?
        },
    );
    add_serving_ops(&mut router, &source);
    router
}

fn add_serving_ops(router: &mut OperationRouter, source: &ServiceSource) {
    add_op(
        router,
        source,
        ops::SERVE_BEGIN,
        "Check prices, wait out prior debt and open serving.",
        |service, request: ServeBeginRequest| async move {
            service.serve_begin(request).await.map(|()| Empty {})
        },
    );
    add_op(
        router,
        source,
        ops::SERVE_INPUT_INVOICE,
        "Fix the output allowance and issue the input invoice.",
        |service, request: ServeInputInvoiceRequest| async move {
            service.serve_input_invoice(request).await
        },
    );
    add_op(
        router,
        source,
        ops::AWAIT_ARRIVAL,
        "Wait for receiver-side evidence of an input payment.",
        |service, request: InvoiceRequest| async move {
            let claiming = service
                .arrival(&request.invoice, crate::lifetimes::INPUT_ARRIVAL_WAIT)
                .await?;
            Ok(ArrivalResponse { claiming })
        },
    );
    add_op(
        router,
        source,
        ops::SETTLE_RECEIVED,
        "Wait for an invoice to settle and record it received.",
        |service, request: InvoiceRequest| async move {
            // Durable settlement continues even if the caller goes away.
            tokio::spawn(async move { service.settle_received(&request.invoice).await }).await??;
            Ok(Empty {})
        },
    );
    add_op(
        router,
        source,
        ops::RECORD_DELIVERED,
        "Raise the delivered-token watermark.",
        |service, request: RecordDeliveredRequest| async move {
            service
                .ledger
                .record_delivered_tokens(&request.id, request.tokens)
                .map(|()| Empty {})
        },
    );
    add_op(
        router,
        source,
        ops::SERVE_FINISH,
        "Record the final delivered-token watermark and close serving accounting atomically.",
        |service, request: ServeFinishRequest| async move {
            service
                .ledger
                .finish_serving_at(&request.id, request.tokens)
                .map(|()| Empty {})
        },
    );
    add_op(
        router,
        source,
        ops::OUTPUT_RECEIVABLE,
        "Issue or return the output invoice.",
        |service, request: IdRequest| async move {
            Ok(OutputReceivableResponse {
                output: service.output_invoice(&request.id).await?,
            })
        },
    );
    add_op(
        router,
        source,
        ops::SERVE_RECOVER,
        "Answer a payer's recovery probe.",
        |service, request: IdRequest| async move { service.serve_recover(&request.id).await },
    );
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
