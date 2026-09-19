use std::sync::Arc;

use mesh_llm_payments::service::PaymentService;

impl super::Node {
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

    pub(crate) async fn peer_payment_offer(
        &self,
        peer: iroh::EndpointId,
        model: &str,
    ) -> Option<mesh_llm_payments::pricing::Pricing> {
        self.state
            .lock()
            .await
            .peers
            .get(&peer)?
            .lightning_offers
            .get(model)
            .cloned()
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
                    let service = Arc::new(PaymentService::open(&directory)?);
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
