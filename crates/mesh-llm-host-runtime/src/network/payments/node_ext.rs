//! The payments engine slot on [`Node`]. Lives here, not in `mesh/`, so mesh
//! core only sees the pure types crate and never names `PaymentService`.

use std::sync::Arc;

use mesh_llm_payments::service::PaymentService;

use crate::mesh::Node;

/// Lazily opened payments engine, shared by every clone of a [`Node`].
pub(crate) type PaymentsSlot = Arc<tokio::sync::OnceCell<Arc<PaymentService>>>;

impl Node {
    pub(crate) async fn advertised_payment_offers(
        &self,
    ) -> anyhow::Result<std::collections::BTreeMap<String, mesh_llm_payments::pricing::Pricing>>
    {
        let directory = self.config_state.lock().await.payment_directory();
        if self.payments.get().is_none() && !directory.join("payments.sqlite3").exists() {
            return Ok(Default::default());
        }
        self.payment_service().await?.ledger.pricing()
    }

    pub(crate) fn payment_service(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<Arc<PaymentService>>> + Send + '_>,
    > {
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
                    let service =
                        Arc::new(PaymentService::with_factory(&directory, Arc::new(factory))?);
                    let recovery_service = Arc::downgrade(&service);
                    let node = self.clone();
                    tokio::spawn(async move {
                        loop {
                            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
                            if node.endpoint.is_closed() {
                                break;
                            }
                            let Some(service) = recovery_service.upgrade() else {
                                break;
                            };
                            let _ =
                                crate::network::openai::payment_recovery::recover(&node, &service)
                                    .await;
                        }
                    });
                    Ok::<_, anyhow::Error>(service)
                })
                .await?;
            Ok(Arc::clone(service))
        })
    }
}

/// In-process builtins this node supplies to its plugin manager: the payments
/// engine as `payments.v1`, opened lazily through this node's slot.
pub(crate) fn in_process_plugins(node: &Node) -> crate::plugin::InProcessPlugins {
    let node = node.clone();
    let source: mesh_llm_payments::plugin_server::ServiceSource = Arc::new(move || {
        let node = node.clone();
        Box::pin(async move { node.payment_service().await })
    });
    let runner: crate::plugin::InProcessPluginRunner = Arc::new(move |stream| {
        let plugin = mesh_llm_payments::plugin_server::payments_plugin(
            crate::plugin::PAYMENTS_PLUGIN_ID,
            crate::VERSION,
            Arc::clone(&source),
        );
        Box::pin(mesh_llm_plugin::PluginRuntime::run_with_stream(
            plugin, stream,
        ))
    });
    crate::plugin::InProcessPlugins::default().with(crate::plugin::PAYMENTS_PLUGIN_ID, runner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_payments::plugin_server::{CAPABILITY, ops};

    #[tokio::test]
    async fn payments_engine_is_served_in_process_by_capability() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let node = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        node.payments
            .set(Arc::new(PaymentService::open(directory.path())?))
            .map_err(|_| anyhow::anyhow!("already set"))?;
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
            in_process_plugins(&node),
        )
        .await?;

        let set = r#"{"command":"set_pricing","model":"m","value":{"input_msat_per_million":1,"output_msat_per_million":2,"minimum_invoice_msat":3}}"#;
        let result = manager
            .invoke_operation_by_capability(CAPABILITY, ops::CONTROL, set)
            .await?;
        assert!(!result.is_error, "{}", result.content_json);
        let pricing: serde_json::Value = serde_json::from_str(&result.content_json)?;
        assert_eq!(pricing["m"]["output_msat_per_million"], 2);
        // Same engine the host holds: the plugin wraps the node's service.
        assert!(
            node.payment_service()
                .await?
                .ledger
                .pricing()?
                .contains_key("m")
        );

        let bad = manager
            .invoke_operation_by_capability(CAPABILITY, ops::CONTROL, r#"{"command":"nope"}"#)
            .await?;
        assert!(bad.is_error);
        manager.shutdown().await;
        Ok(())
    }
}
