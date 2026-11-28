//! Provider eligibility from plugin-verified evidence in signed node records.
use super::{Node, NodeRole, PeerInfo};
use crate::plugin::PluginManager;
use crate::proto::node::NodeRecord;
use futures_util::{StreamExt, stream::FuturesUnordered};
use iroh::EndpointId;
use mesh_llm_membership::signed_record::VerifiedRecord;
use prost::Message;
use std::collections::{BTreeMap, HashSet};

/// How many verifier calls run at once.
const MAX_CONCURRENT_VERIFICATIONS: usize = 8;

/// Peers that offer compute need a verdict; pure clients do not.
fn is_provider(peer: &PeerInfo) -> bool {
    !matches!(peer.role, NodeRole::Client)
        || !peer.serving_models.is_empty()
        || !peer.hosted_models.is_empty()
        || !peer.served_model_descriptors.is_empty()
        || peer.experts_summary.is_some()
}

/// The same test on a signed record, which a direct peer's unsigned
/// self-description cannot override.
fn record_is_provider(record: &NodeRecord) -> bool {
    record.role != crate::proto::node::NodeRole::Client as i32
        || !record.serving_models.is_empty()
        || !record.hosted_models.is_empty()
        || !record.served_models.is_empty()
        || record.experts_summary.is_some()
}

/// Judges peers against a verifier's cached verdicts.
pub(super) struct EvidenceGate {
    manager: PluginManager,
    mesh: String,
}

impl EvidenceGate {
    /// Whether the peer is a client, or a provider whose signed `record`
    /// carries evidence the verifier currently accepts.
    pub(super) fn admits(&self, peer: &PeerInfo, record: Option<&VerifiedRecord>) -> bool {
        match record.and_then(|held| NodeRecord::decode(held.body()).ok()) {
            Some(record) if is_provider(peer) || record_is_provider(&record) => self
                .manager
                .peer_evidence()
                .provider_cached(&self.mesh, &peer.id.to_string(), &record.peer_evidence),
            Some(_) => true,
            // A provider without a signed record has no evidence to accept.
            None => !is_provider(peer),
        }
    }
}

impl Node {
    pub(crate) async fn has_peer_evidence_verifier(&self) -> bool {
        self.plugin_manager()
            .await
            .is_some_and(|manager| manager.peer_evidence().has_verifier())
    }

    /// The evidence each provider signed into its current node record, for
    /// peers that describe themselves or sign themselves as providers.
    async fn provider_evidence(
        &self,
        peers: &[PeerInfo],
    ) -> Vec<(EndpointId, BTreeMap<String, Vec<u8>>)> {
        let state = self.state.lock().await;
        let now = super::requirements::current_time_unix_ms();
        peers
            .iter()
            .filter_map(|peer| {
                let held = state.node_records.fresh(&peer.id, now)?;
                let record = NodeRecord::decode(held.body()).ok()?;
                (is_provider(peer) || record_is_provider(&record))
                    .then_some((peer.id, record.peer_evidence))
            })
            .collect()
    }

    /// The verifier's view of which peers are eligible, or `None` without a
    /// verifier, when every peer is.
    pub(super) async fn evidence_gate(&self) -> Option<EvidenceGate> {
        let manager = self.plugin_manager().await?;
        if !manager.peer_evidence().has_verifier() {
            return None;
        }
        let mesh = self.mesh_id().await.unwrap_or_default();
        Some(EvidenceGate { manager, mesh })
    }

    /// Keeps clients and providers whose signed evidence the verifier
    /// currently accepts.
    pub(crate) async fn filter_current_peer_evidence(&self, peers: &mut Vec<PeerInfo>) {
        let Some(gate) = self.evidence_gate().await else {
            return;
        };
        let state = self.state.lock().await;
        let now = super::requirements::current_time_unix_ms();
        peers.retain(|peer| gate.admits(peer, state.node_records.fresh(&peer.id, now)));
    }

    pub(super) async fn request_peer_evidence_refresh(&self) {
        if let Some(manager) = self.plugin_manager().await {
            manager.peer_evidence().refresh.notify_one();
        }
    }

    /// Verifies providers without a current verdict whenever peers change,
    /// a verifier restarts or a verdict expires, and renews acceptances
    /// before they expire. Calls run side by side and each result counts as
    /// soon as it arrives, so a slow call holds up neither other providers'
    /// renewals nor the announcement of an acceptance.
    pub(super) async fn maintain_peer_evidence(&self) {
        let Some(manager) = self.plugin_manager().await else {
            return;
        };
        let registry = manager.peer_evidence();
        if !registry.has_verifier() {
            return;
        }
        let mut running = FuturesUnordered::new();
        let mut busy = HashSet::new();
        loop {
            let mesh = self.mesh_id().await.unwrap_or_default();
            let free = MAX_CONCURRENT_VERIFICATIONS - running.len();
            let due = self.due_providers(&manager, &mesh, &busy).await;
            for (node, evidence) in due.into_iter().take(free) {
                busy.insert(node.clone());
                let (manager, mesh) = (manager.clone(), mesh.clone());
                running.push(async move {
                    let verification = manager
                        .verify_provider_evidence(&mesh, &node, &evidence)
                        .await;
                    (node, verification)
                });
            }
            // With every slot taken, only a finished call can start another.
            let next_due = (running.len() < MAX_CONCURRENT_VERIFICATIONS)
                .then(|| registry.next_due(|node| busy.contains(node)))
                .flatten();
            let due_wake = async {
                match next_due {
                    Some(at) => tokio::time::sleep_until(at.into()).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                _ = self.endpoint.closed() => return,
                _ = registry.refresh.notified() => {}
                _ = due_wake => {}
                Some((node, verification)) = running.next(), if !running.is_empty() => {
                    busy.remove(&node);
                    if verification.newly_eligible {
                        // Only an acceptance makes a provider newly routable,
                        // and relayable: peers further away learn of it now
                        // rather than at the next periodic gossip.
                        self.peer_change_tx.send_modify(|_| {});
                        self.regossip().await;
                    }
                }
            }
        }
    }

    /// The providers due for verification, except those `busy` names:
    /// renewals first, nearest expiry first, since a lapse makes a provider
    /// in use ineligible; then the providers without a verdict.
    async fn due_providers(
        &self,
        manager: &PluginManager,
        mesh: &str,
        busy: &HashSet<String>,
    ) -> Vec<(String, BTreeMap<String, Vec<u8>>)> {
        let peers = self.peers().await;
        let providers: Vec<_> = self
            .provider_evidence(&peers)
            .await
            .into_iter()
            .map(|(id, evidence)| (id.to_string(), evidence))
            .collect();
        let registry = manager.peer_evidence();
        registry.retain_current(mesh, &providers);
        let mut due: Vec<_> = providers
            .into_iter()
            .filter(|(node, evidence)| {
                !busy.contains(node) && registry.needs_verification(mesh, node, evidence)
            })
            .map(|(node, evidence)| {
                let expiry = registry.acceptance_expiry(mesh, &node, &evidence);
                ((expiry.is_none(), expiry), node, evidence)
            })
            .collect();
        due.sort_by_key(|(order, ..)| *order);
        due.into_iter()
            .map(|(_, node, evidence)| (node, evidence))
            .collect()
    }

    /// Verifies every due provider in turn, and returns whether any became
    /// eligible.
    #[cfg(test)]
    pub(super) async fn verify_undecided_providers(&self, manager: &PluginManager) -> bool {
        let mesh = self.mesh_id().await.unwrap_or_default();
        let mut newly_eligible = false;
        for (node, evidence) in self.due_providers(manager, &mesh, &HashSet::new()).await {
            newly_eligible |= manager
                .verify_provider_evidence(&mesh, &node, &evidence)
                .await
                .newly_eligible;
        }
        newly_eligible
    }
}
