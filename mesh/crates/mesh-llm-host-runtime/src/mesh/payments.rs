impl super::Node {
    /// The price `peer` advertises for `model`, or `None` if it serves the
    /// model for free or is unknown.
    pub(crate) async fn peer_payment_offer(
        &self,
        peer: iroh::EndpointId,
        model: &str,
    ) -> Option<mesh_llm_payments_types::pricing::Pricing> {
        let state = self.state.lock().await;
        peer_offer_for_model(state.peers.get(&peer)?, model).cloned()
    }
}

/// The price a peer advertises for `model`, as routing reads it.
pub(crate) fn peer_offer_for_model<'a>(
    peer: &'a super::PeerInfo,
    model: &str,
) -> Option<&'a mesh_llm_payments_types::pricing::Pricing> {
    peer.lightning_offers.get(model).or_else(|| {
        // Discovery may expose the peer's public model ID rather than its
        // runtime name. Resolve only aliases advertised by this peer.
        peer.lightning_offers.iter().find_map(|(name, price)| {
            (super::routes_http_model(peer, name)
                && super::peer_state::public_model_id_for_routable_model(peer, name) == model)
                .then_some(price)
        })
    })
}
