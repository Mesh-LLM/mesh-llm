impl super::Node {
    pub(crate) async fn peer_payment_offer(
        &self,
        peer: iroh::EndpointId,
        model: &str,
    ) -> Option<mesh_llm_payments_types::pricing::Pricing> {
        let state = self.state.lock().await;
        let peer = state.peers.get(&peer)?;
        peer.lightning_offers.get(model).cloned().or_else(|| {
            // Discovery may expose the peer's public model ID rather than its
            // runtime name. Resolve only aliases advertised by this peer.
            peer.lightning_offers.iter().find_map(|(name, price)| {
                (peer.routes_http_model(name)
                    && peer.public_model_id_for_routable_model(name) == model)
                    .then(|| price.clone())
            })
        })
    }
}
