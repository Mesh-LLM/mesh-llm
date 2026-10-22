use super::*;

pub(super) fn peer_requires_provider(peer: &ProtocolPeer, descriptor: &ProtocolDescriptor) -> bool {
    peer.protocols.iter().any(|protocol| {
        protocol.name == descriptor.name
            && protocol.major == descriptor.major
            && protocol
                .features
                .iter()
                .any(|feature| feature == REQUIRED_STAGE_PROVIDER_FEATURE)
    })
}

impl Node {
    pub(crate) fn stage_transport_policy(&self) -> StageTransportPolicy {
        self.application_protocols
            .policy
            .get()
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn stage_provider_context_allowed(&self, descriptor: &ProtocolDescriptor) -> bool {
        self.stage_transport_policy() != StageTransportPolicy::RequireProvider
            || self
                .application_protocols
                .registrations
                .get()
                .is_some_and(|registry| {
                    registry.values().any(|entry| {
                        entry.descriptor.name == descriptor.name
                            && entry.descriptor.major == descriptor.major
                            && entry.handler.stage_transport().is_some()
                    })
                })
    }

    pub(crate) async fn stage_required_peer_allowed(
        &self,
        peer: PeerId,
        check_available: bool,
    ) -> bool {
        if self.stage_transport_policy() != StageTransportPolicy::RequireProvider {
            return true;
        }
        let Some(registry) = self.application_protocols.registrations.get() else {
            return false;
        };
        let state = self.state.lock().await;
        let Some(remote) = state.peers.get(&peer).filter(|remote| remote.is_admitted()) else {
            return false;
        };
        let advertised = ProtocolPeer {
            id: peer,
            protocols: remote
                .application_protocols
                .iter()
                .map(|protocol| ProtocolDescriptor {
                    name: protocol.name.clone(),
                    major: protocol.major,
                    features: protocol.features.clone(),
                })
                .collect(),
        };
        drop(state);
        registry.values().any(|entry| {
            entry.handler.stage_transport().is_some_and(|provider| {
                peer_requires_provider(&advertised, &entry.descriptor)
                    && (!check_available || provider.stage_available(peer))
            })
        })
    }

    pub(crate) async fn verify_required_stage_provider(&self, peer: PeerId) -> Result<()> {
        if self.stage_transport_policy() == StageTransportPolicy::RequireProvider {
            ensure!(
                self.stage_required_peer_allowed(peer, true).await,
                "required stage provider is unavailable or peer does not enforce the policy"
            );
            let stream = self
                .open_application_stage_stream(peer)
                .await?
                .ok_or_else(|| {
                    anyhow::anyhow!("required stage provider declined setup verification")
                })?;
            // Setup verification carries no stage bytes and holds no capacity after this check.
            drop(stream);
        }
        Ok(())
    }

    pub(crate) fn ensure_default_stage_transport_allowed(&self) -> Result<()> {
        ensure!(
            self.stage_transport_policy() != StageTransportPolicy::RequireProvider,
            "default activation transport is disabled by the required provider policy"
        );
        Ok(())
    }
}
