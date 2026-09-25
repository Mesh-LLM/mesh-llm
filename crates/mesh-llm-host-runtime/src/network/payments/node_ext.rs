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
