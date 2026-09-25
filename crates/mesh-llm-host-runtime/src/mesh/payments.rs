impl super::Node {
    pub(crate) async fn peer_payment_offer(
        &self,
        peer: iroh::EndpointId,
        model: &str,
    ) -> Option<mesh_llm_payments_types::pricing::Pricing> {
        self.state
            .lock()
            .await
            .peers
            .get(&peer)?
            .lightning_offers
            .get(model)
            .cloned()
    }
}
