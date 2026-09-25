//! Host-side client of the `payments.v1` capability. Every per-request payment
//! decision the host makes goes through here, to whichever provider serves
//! the capability; the host never holds the engine for these calls.

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use mesh_llm_payments_types::contract::{CAPABILITY, OpError, deadline};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::mesh::Node;
use crate::plugin::PluginManager;

/// Invokes `operation` on the `payments.v1` provider resolved for this call.
/// Each call resolves the provider afresh, so callers that own an exchange
/// should hold a [`Payments`] instead and reach one provider throughout.
pub(crate) async fn call<Req: Serialize, Res: DeserializeOwned>(
    plugins: &PluginManager,
    operation: &str,
    request: &Req,
) -> Result<Res> {
    let provider = plugins
        .available_provider_for_capability(CAPABILITY)
        .await?
        .ok_or_else(|| anyhow!("no provider for '{CAPABILITY}'"))?;
    invoke(plugins, &provider.plugin_name, operation, request).await
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

/// Invokes `operation` on the named provider, applying the operation's bound.
/// Settlement and wallet operations may legitimately outlast the default RPC
/// deadline and own their durability, so only the operations the contract
/// classifies as bookkeeping are bounded here.
async fn invoke<Req: Serialize, Res: DeserializeOwned>(
    plugins: &PluginManager,
    provider: &str,
    operation: &str,
    request: &Req,
) -> Result<Res> {
    let input = serde_json::to_string(request)?;
    let call = plugins.invoke_operation_without_timeout(provider, operation, &input);
    let result = match deadline(operation) {
        Some(limit) => tokio::time::timeout(limit, call).await.map_err(|_| {
            anyhow!("payments operation '{operation}' did not answer within {limit:?}")
        })??,
        None => call.await?,
    };
    if result.is_error {
        let message = serde_json::from_str::<OpError>(&result.content_json)
            .map(|error| error.message)
            .unwrap_or_else(|_| format!("payments operation '{operation}' failed"));
        bail!("{message}");
    }
    Ok(serde_json::from_str(&result.content_json)?)
}

/// A handle to the `payments.v1` provider for one paid exchange.
///
/// The provider is resolved once, when the exchange starts, and every operation
/// of that exchange is addressed to it. Re-resolving per operation would let a
/// provider that appears, is replaced or disappears mid-exchange split one
/// exchange across two ledgers; with the pin, the exchange fails closed instead.
#[derive(Clone)]
pub(crate) struct Payments {
    plugins: PluginManager,
    provider: Option<Arc<str>>,
}

impl Payments {
    /// Resolves and pins the provider this handle will keep addressing.
    async fn resolve(plugins: PluginManager) -> Self {
        let provider = plugins
            .available_provider_for_capability(CAPABILITY)
            .await
            .ok()
            .flatten()
            .map(|provider| Arc::from(provider.plugin_name.as_str()));
        Self { plugins, provider }
    }

    /// Serves `service` from `node` over an in-process `payments.v1`, as
    /// runtime startup does, and returns a client to it.
    #[cfg(test)]
    pub(crate) async fn attach_for_tests(
        node: &Node,
        service: std::sync::Arc<mesh_llm_payments::service::PaymentService>,
    ) -> Result<Self> {
        node.payments
            .set(service)
            .map_err(|_| anyhow!("payments already initialized"))?;
        let plugins = super::node_ext::attach_payments_plugin(node).await?;
        Ok(Self::resolve(plugins).await)
    }

    /// Pins the `payments.v1` provider for one exchange on `node`.
    pub(crate) async fn for_node(node: &Node) -> Result<Self> {
        let plugins = node
            .plugin_manager()
            .await
            .ok_or_else(|| anyhow!("payments are not available yet"))?;
        Ok(Self::resolve(plugins).await)
    }

    /// Pins the `payments.v1` provider for one exchange on an existing manager.
    pub(crate) async fn for_plugins(plugins: PluginManager) -> Self {
        Self::resolve(plugins).await
    }

    /// A client with no `payments.v1` provider: every operation fails.
    #[cfg(test)]
    pub(crate) async fn unavailable_for_tests() -> Result<Self> {
        let (mesh_tx, _mesh_rx) = tokio::sync::mpsc::channel(8);
        let plugins = PluginManager::start_with_in_process(
            &crate::plugin::ResolvedPlugins {
                externals: Vec::new(),
                inactive: Vec::new(),
            },
            crate::plugin::PluginHostMode {
                mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
            },
            mesh_tx,
            crate::plugin::InProcessPlugins::default(),
        )
        .await?;
        Ok(Self::resolve(plugins).await)
    }

    /// Invokes `operation` on this exchange's pinned provider.
    pub(crate) async fn call<Req: Serialize, Res: DeserializeOwned>(
        &self,
        operation: &str,
        request: &Req,
    ) -> Result<Res> {
        let provider = self
            .provider
            .as_deref()
            .ok_or_else(|| anyhow!("no provider for '{CAPABILITY}'"))?;
        invoke(&self.plugins, provider, operation, request).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use mesh_llm_payments_types::contract::{ArrivalResponse, Empty, InvoiceRequest, ops};
    use rmcp::model::CallToolResult;
    use serde_json::json;

    use super::*;
    use crate::plugin::{self, PluginRpcBridge};

    /// A scripted `payments.v1` provider. It records which plugin each operation
    /// was addressed to, and can be told never to answer one operation.
    #[derive(Default)]
    struct FakePayments {
        calls: Mutex<Vec<(String, String)>>,
        hang: Mutex<Vec<String>>,
    }

    impl FakePayments {
        fn hang(&self, operation: &str) {
            self.hang.lock().unwrap().push(operation.to_owned());
        }

        fn calls(&self) -> Vec<(String, String)> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl PluginRpcBridge for Arc<FakePayments> {
        fn handle_request(
            &self,
            plugin_name: String,
            _method: String,
            params_json: String,
        ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
            let this = Arc::clone(self);
            Box::pin(async move {
                let request: mesh_llm_plugin::OperationRequest =
                    serde_json::from_str(&params_json).map_err(transport_error)?;
                this.calls
                    .lock()
                    .unwrap()
                    .push((plugin_name, request.name.clone()));
                if this.hang.lock().unwrap().contains(&request.name) {
                    // A provider that accepts the operation and never answers.
                    return std::future::pending().await;
                }
                Ok(plugin::RpcResult {
                    result_json: serde_json::to_string(&CallToolResult::structured(json!({})))
                        .map_err(transport_error)?,
                })
            })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> plugin::BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    fn transport_error(error: impl std::fmt::Display) -> plugin::proto::ErrorResponse {
        plugin::proto::ErrorResponse {
            code: rmcp::model::ErrorCode::INTERNAL_ERROR.0,
            message: error.to_string(),
            data_json: String::new(),
        }
    }

    fn manifest(capabilities: &[&str]) -> mesh_llm_plugin::proto::PluginManifest {
        mesh_llm_plugin::proto::PluginManifest {
            capabilities: capabilities
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            ..Default::default()
        }
    }

    /// A manager serving a scripted `payments.v1` provider per entry.
    async fn manager(
        entries: &[(&str, &[&str])],
        bridge: Arc<dyn PluginRpcBridge>,
    ) -> PluginManager {
        let names: Vec<&str> = entries.iter().map(|(name, _)| *name).collect();
        let manager = PluginManager::for_test_bridge(&names, bridge);
        let manifests = entries
            .iter()
            .map(|(name, capabilities)| ((*name).to_string(), manifest(capabilities)))
            .collect();
        manager.set_test_manifests(manifests).await;
        manager
    }

    fn sample_invoice() -> mesh_llm_wallet::invoice::Invoice {
        use bitcoin::secp256k1::{Secp256k1, SecretKey};
        use lightning_invoice::{Currency, InvoiceBuilder, PaymentHash, PaymentSecret};
        let secret = SecretKey::from_slice(&[7; 32]).unwrap();
        let bolt11 = InvoiceBuilder::new(Currency::Bitcoin)
            .description("test".into())
            .payment_hash(PaymentHash([9; 32]))
            .payment_secret(PaymentSecret([42; 32]))
            .current_timestamp()
            .expiry_time(std::time::Duration::from_secs(3600))
            .min_final_cltv_expiry_delta(144)
            .amount_milli_satoshis(1000)
            .build_signed(|hash| Secp256k1::new().sign_ecdsa_recoverable(hash, &secret))
            .unwrap()
            .to_string();
        mesh_llm_wallet::invoice::Invoice::parse(&bolt11).unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn a_bookkeeping_operation_is_bounded() {
        let fake = Arc::new(FakePayments::default());
        fake.hang(ops::SERVE_FINISH);
        let payments = Payments::for_plugins(
            manager(
                &[("payments-fake", &[CAPABILITY])],
                Arc::new(Arc::clone(&fake)),
            )
            .await,
        )
        .await;

        let error = payments
            .call::<_, Empty>(ops::SERVE_FINISH, &Empty {})
            .await
            .expect_err("a hung bookkeeping operation must fail instead of hanging the caller");
        assert!(error.to_string().contains("did not answer"), "{error}");
        assert_eq!(
            fake.calls(),
            [(String::from("payments-fake"), ops::SERVE_FINISH.to_owned())],
            "the operation is attempted exactly once"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_wallet_wait_is_not_bounded_by_the_bookkeeping_deadline() {
        let fake = Arc::new(FakePayments::default());
        fake.hang(ops::AWAIT_ARRIVAL);
        let payments = Payments::for_plugins(
            manager(
                &[("payments-fake", &[CAPABILITY])],
                Arc::new(Arc::clone(&fake)),
            )
            .await,
        )
        .await;

        let request = InvoiceRequest {
            invoice: sample_invoice(),
        };
        let waiting = tokio::time::timeout(
            mesh_llm_payments_types::contract::BOOKKEEPING_DEADLINE * 3,
            payments.call::<_, ArrivalResponse>(ops::AWAIT_ARRIVAL, &request),
        )
        .await;
        assert!(
            waiting.is_err(),
            "a wallet wait must outlive the bookkeeping bound"
        );
    }

    #[tokio::test]
    async fn one_exchange_keeps_addressing_the_provider_it_pinned() {
        let fake = Arc::new(FakePayments::default());
        let manager = manager(
            &[("a-payments", &[CAPABILITY]), ("b-payments", &[CAPABILITY])],
            Arc::new(Arc::clone(&fake)),
        )
        .await;
        let exchange = Payments::for_plugins(manager.clone()).await;
        exchange
            .call::<_, Empty>(ops::PAYMENT_INTENT, &Empty {})
            .await
            .unwrap();
        // "a-payments" stops advertising the capability, so a fresh resolution
        // would pick "b-payments"; this exchange keeps the provider it pinned.
        manager
            .set_test_manifests(BTreeMap::from([
                (String::from("a-payments"), manifest(&[])),
                (String::from("b-payments"), manifest(&[CAPABILITY])),
            ]))
            .await;
        exchange
            .call::<_, Empty>(ops::PAYMENT_INTENT, &Empty {})
            .await
            .unwrap();
        assert_eq!(
            fake.calls(),
            [
                (String::from("a-payments"), ops::PAYMENT_INTENT.to_owned()),
                (String::from("a-payments"), ops::PAYMENT_INTENT.to_owned()),
            ]
        );
        // A new exchange resolves the provider that is available now.
        let fresh = Payments::for_plugins(manager).await;
        fresh
            .call::<_, Empty>(ops::PAYMENT_INTENT, &Empty {})
            .await
            .unwrap();
        assert_eq!(fake.calls().last().unwrap().0, "b-payments");
    }
}
