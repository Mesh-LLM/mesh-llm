//! The payments engine slot on [`Node`]. Lives here, not in `mesh/`, so mesh
//! core only sees the pure types crate and never names the engine.

use std::collections::BTreeMap;
use std::sync::Arc;

use mesh_llm_payments_types::contract::{AdvertisedPricing, Empty, ops};
use mesh_llm_payments_types::engine::AdvertisedPrices;
use mesh_llm_payments_types::pricing::Pricing;

use super::engine::PaymentsEngine;
use crate::mesh::Node;

/// Lazily opened payments engine, shared by every clone of a [`Node`].
type EngineFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = anyhow::Result<Arc<dyn PaymentsEngine>>> + Send + 'a>,
>;

pub(crate) type PaymentsSlot = Arc<tokio::sync::OnceCell<Arc<dyn PaymentsEngine>>>;

/// How often the host reconciles durable payment state with the provider.
const RECOVERY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(15);

impl Node {
    /// Prices this node advertises to the mesh, projected through the
    /// `payments.v1` capability rather than read from the engine.
    pub(crate) async fn advertised_payment_offers(
        &self,
    ) -> anyhow::Result<BTreeMap<String, Pricing>> {
        Ok(self.advertised_pricing().await?.prices)
    }

    /// Advertised prices, and whether this node's payments provider has any
    /// state at all. The capability answers both, so an external provider is
    /// authoritative for them; a node with no provider advertises nothing.
    pub(crate) async fn advertised_pricing(&self) -> anyhow::Result<AdvertisedPricing> {
        if self.plugin_manager().await.is_none() {
            return Ok(AdvertisedPricing::default());
        }
        super::client::call_node(self, ops::PRICING, &Empty {}).await
    }

    /// The builtin engine's advertised prices, read without opening a ledger
    /// this node does not have. Supplied to the in-process engine plugin, so
    /// the host never has to name the builtin; an external provider answers
    /// `payments.v1` pricing for itself.
    fn advertised_prices_source(&self) -> AdvertisedPrices {
        let node = self.clone();
        Arc::new(move || {
            let node = node.clone();
            Box::pin(async move {
                let directory = node.config_state.lock().await.payment_directory();
                if node.payments.get().is_none() && !directory.join("payments.sqlite3").exists() {
                    return Ok(None);
                }
                Ok(Some(node.payment_engine().await?.pricing()?))
            })
        })
    }

    pub(crate) fn payment_engine(&self) -> EngineFuture<'_> {
        Box::pin(async move {
            let service = self
                .payments
                .get_or_try_init(|| async {
                    let directory = self.config_state.lock().await.payment_directory();
                    // The wallet is a plugin (`wallet.v1`); the ledger stays
                    // in-process. The factory holds the plugin-manager slot,
                    // not a manager: this can run during startup (gossip
                    // advertises prices) before `set_plugin_manager`.
                    let factory = crate::network::payments::wallet_plugin::PluginWalletFactory::new(
                        Arc::clone(&self.plugin_manager),
                    );
                    let provider = super::engine::provider()
                        .ok_or_else(|| anyhow::anyhow!("no payments engine installed"))?;
                    provider.open(&directory, Arc::new(factory))
                })
                .await?;
            Ok(Arc::clone(service))
        })
    }
}

/// Runs the periodic payment recovery loop.
///
/// The runtime starts this once the plugin manager is installed, so it
/// reconciles whichever `payments.v1` provider serves this node — the builtin
/// or an external one — instead of only the builtin's engine slot. The loop
/// asks the provider whether it has payments state before reconciling, so a
/// node that never configured payments does not create a ledger to find
/// nothing.
pub(crate) fn spawn_payment_recovery(node: &Node) {
    let node = node.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(RECOVERY_INTERVAL).await;
            if node.endpoint.is_closed() {
                break;
            }
            let Ok(advertised) = node.advertised_pricing().await else {
                continue;
            };
            if !advertised.configured {
                continue;
            }
            let _ = crate::network::openai::payment_recovery::recover(&node).await;
        }
    });
}

/// In-process builtins this node supplies to its plugin manager: the payments
/// engine as `payments.v1`, opened lazily through this node's slot.
pub(crate) fn in_process_plugins(node: &Node) -> crate::plugin::InProcessPlugins {
    let Some(provider) = super::engine::provider() else {
        return crate::plugin::InProcessPlugins::default();
    };
    let node = node.clone();
    let prices = node.advertised_prices_source();
    let source: super::engine::EngineSource = Arc::new(move || {
        let node = node.clone();
        Box::pin(async move { node.payment_engine().await })
    });
    let runner: crate::plugin::InProcessPluginRunner = Arc::new(move |stream| {
        provider.serve(
            crate::plugin::PAYMENTS_PLUGIN_ID,
            crate::VERSION,
            Arc::clone(&source),
            Arc::clone(&prices),
            stream,
        )
    });
    crate::plugin::InProcessPlugins::default().with(crate::plugin::PAYMENTS_PLUGIN_ID, runner)
}

/// Starts a plugin manager serving this node's payments engine in-process and
/// installs it on the node, as runtime startup does.
#[cfg(test)]
pub(crate) async fn attach_payments_plugin(
    node: &Node,
) -> anyhow::Result<crate::plugin::PluginManager> {
    install_test_engine();
    let specs = crate::plugin::ResolvedPlugins {
        externals: vec![crate::plugin::in_process_builtin_spec(
            crate::plugin::PAYMENTS_PLUGIN_ID,
        )],
        inactive: Vec::new(),
    };
    let (mesh_tx, _mesh_rx) = tokio::sync::mpsc::channel(8);
    let manager = crate::plugin::PluginManager::start_with_in_process(
        &specs,
        crate::plugin::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        in_process_plugins(node),
    )
    .await?;
    node.set_plugin_manager(manager.clone()).await;
    Ok(manager)
}

/// Installs the real engine for tests, as the shipped binary does.
#[cfg(test)]
pub(crate) fn install_test_engine() {
    super::engine::install_payments_engine(Arc::new(
        mesh_llm_payments::plugin_server::EngineProvider,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_payments::plugin_server::{CAPABILITY, ops};

    #[tokio::test]
    async fn payments_engine_is_served_in_process_by_capability() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let node = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        let service = Arc::new(mesh_llm_payments::service::PaymentService::open(
            directory.path(),
        )?);
        node.payments
            .set(service.clone())
            .map_err(|_| anyhow::anyhow!("already set"))?;
        let manager = attach_payments_plugin(&node).await?;

        let set = r#"{"command":"set_pricing","model":"m","value":{"input_msat_per_million":1,"output_msat_per_million":2,"minimum_invoice_msat":3}}"#;
        let result = manager
            .invoke_operation_by_capability(CAPABILITY, ops::CONTROL, set)
            .await?;
        assert!(!result.is_error, "{}", result.content_json);
        let pricing: serde_json::Value = serde_json::from_str(&result.content_json)?;
        assert_eq!(pricing["m"]["output_msat_per_million"], 2);
        // Same engine the host holds: the plugin wraps the node's service.
        assert!(service.ledger.pricing()?.contains_key("m"));
        assert!(node.payment_engine().await?.pricing()?.contains_key("m"));
        // Advertised prices now come from the capability, and agree.
        let advertised = node.advertised_payment_offers().await?;
        assert!(advertised.contains_key("m"));

        let bad = manager
            .invoke_operation_by_capability(CAPABILITY, ops::CONTROL, r#"{"command":"nope"}"#)
            .await?;
        assert!(bad.is_error);
        manager.shutdown().await;
        Ok(())
    }
}
