//! Provider eligibility from opaque plugin evidence and verified origin records.
use super::{Node, NodeRole, PeerAnnouncement, PeerInfo};
use crate::{proto::node::NodeRecord, protocol::node_record::node_record_to_local};
use iroh::EndpointId;
use prost::Message;
use std::sync::atomic::Ordering;

pub(super) fn is_provider(record: &NodeRecord) -> bool {
    record.role != crate::proto::node::NodeRole::Client as i32
        || !record.serving_models.is_empty()
        || !record.hosted_models.is_empty()
        || !record.served_models.is_empty()
        || record.experts_summary.is_some()
}

pub(super) fn announcement_is_provider(ann: &PeerAnnouncement) -> bool {
    !matches!(ann.role, NodeRole::Client)
        || !ann.serving_models.is_empty()
        || ann
            .hosted_models
            .as_ref()
            .is_some_and(|models| !models.is_empty())
        || !ann.served_model_descriptors.is_empty()
        || ann.experts_summary.is_some()
}

impl Node {
    #[cfg(test)]
    pub(super) async fn peer_evidence_record(&self, peer: EndpointId) -> Option<NodeRecord> {
        let state = self.state.lock().await;
        let held = state
            .node_records
            .fresh(&peer, super::requirements::current_time_unix_ms())?;
        NodeRecord::decode(held.body()).ok()
    }

    /// With a configured verifier, supply comes from the origin's signed
    /// body. A relay cannot graft models onto another provider's identity.
    pub(super) async fn evidence_announcement(
        &self,
        peer: EndpointId,
        ann: &PeerAnnouncement,
    ) -> Option<PeerAnnouncement> {
        self.evidence_announcement_with_record(peer, ann, false)
            .await
            .map(|(ann, _)| ann)
    }

    pub(super) async fn evidence_announcement_with_record(
        &self,
        peer: EndpointId,
        ann: &PeerAnnouncement,
        pin_record: bool,
    ) -> Option<(PeerAnnouncement, Option<Vec<u8>>)> {
        let Some(manager) = self.plugin_manager().await else {
            return Some((ann.clone(), None));
        };
        if !manager.peer_evidence().has_verifiers() {
            return Some((ann.clone(), None));
        }
        let (record, pinned, signature) = {
            let state = self.state.lock().await;
            state
                .node_records
                .fresh(&peer, super::requirements::current_time_unix_ms())
                .and_then(|held| {
                    NodeRecord::decode(held.body()).ok().map(|record| {
                        (
                            Some(record),
                            pin_record.then(|| held.signed_bytes().to_vec()),
                            Some(*held.signature()),
                        )
                    })
                })
                .unwrap_or((None, None, None))
        };
        let provider = announcement_is_provider(ann) || record.as_ref().is_some_and(is_provider);
        if !provider {
            return Some((ann.clone(), pinned));
        }
        let Some(record) = record else {
            let mesh = self.mesh_id().await;
            let state = self.state.lock().await;
            return (state
                .node_records
                .fresh(&peer, super::requirements::current_time_unix_ms())
                .is_none()
                && state.has_fresh_direct_admission(&peer)
                && manager.peer_evidence().provider_cached(
                    mesh.as_deref().unwrap_or_default(),
                    &peer.to_string(),
                    &Default::default(),
                ))
            .then(|| (ann.clone(), None));
        };
        if self
            .mesh_id()
            .await
            .as_deref()
            .is_some_and(|mesh| mesh != record.mesh_id)
            || !manager.peer_evidence().provider_cached(
                &record.mesh_id,
                &peer.to_string(),
                &record.peer_evidence,
            )
        {
            return None;
        }
        let (_, mut canonical) = node_record_to_local(peer, &record, None)?;
        if !self.provider_supply_allowed(peer, &canonical).await {
            return None;
        }
        canonical.cache_affinity = self.signed_cache_affinity(peer).await;
        canonical.cache_affinity = crate::protocol::sanitize_cache_affinity_for_ann(&canonical);
        canonical.direct_admission_proof = ann.direct_admission_proof.clone();
        canonical.model_demand = ann.model_demand.clone();
        canonical.latency_ms = ann.latency_ms;
        canonical.latency_source = ann.latency_source;
        canonical.latency_age_ms = ann.latency_age_ms;
        canonical.latency_observer_id = ann.latency_observer_id;
        if self
            .mesh_id()
            .await
            .as_deref()
            .is_some_and(|mesh| mesh != record.mesh_id)
        {
            return None;
        }
        let state = self.state.lock().await;
        if state
            .node_records
            .fresh(&peer, super::requirements::current_time_unix_ms())
            .is_none_or(|held| signature != Some(*held.signature()))
            || !manager.peer_evidence().provider_cached(
                &record.mesh_id,
                &peer.to_string(),
                &record.peer_evidence,
            )
        {
            return None;
        }
        Some((canonical, pinned))
    }

    async fn provider_supply_allowed(&self, peer: EndpointId, ann: &PeerAnnouncement) -> bool {
        mesh_llm_membership::announcements::version_allowed_for_rebroadcast(ann.version.as_deref())
            && self
                .validate_peer_announcement_against_active_policy(peer, ann)
                .await
                .is_ok()
            && super::peer_state::policy_accepts_peer(
                self.trust_policy,
                &self.direct_peer_owner_summary(peer, ann).await,
            )
    }

    pub(super) async fn filter_current_peer_evidence(&self, peers: &mut Vec<PeerInfo>) {
        let Some(manager) = self.plugin_manager().await else {
            return;
        };
        if !manager.peer_evidence().has_verifiers() {
            return;
        }
        let mut eligible = Vec::with_capacity(peers.len());
        for mut peer in std::mem::take(peers) {
            let ann = mesh_llm_membership::announcements::announcement_from_peer(&peer);
            let Some(canonical) = self.evidence_announcement(peer.id, &ann).await else {
                continue;
            };
            let id = peer.id;
            mesh_llm_membership::announcements::apply_transitive_ann(
                &mut peer,
                &canonical.addr,
                &canonical,
                id,
                super::requirements::current_time_unix_ms(),
            );
            peer.experts_summary = canonical.experts_summary;
            peer.available_models = canonical.available_models;
            peer.inference_admission_state = canonical.inference_admission_state;
            peer.refuses_peer_model_downloads = canonical.refuses_peer_model_downloads;
            peer.cache_affinity = canonical.cache_affinity;
            eligible.push(peer);
        }
        *peers = eligible;
    }

    pub(super) async fn queue_peer_evidence_refresh(&self) {
        let Some(manager) = self.plugin_manager().await else {
            return;
        };
        if !manager.peer_evidence().has_verifiers() {
            return;
        }
        manager
            .peer_evidence()
            .refresh_requested
            .store(true, Ordering::Release);
        let Ok(guard) = manager.peer_evidence().refresh.clone().try_lock_owned() else {
            return;
        };
        let node = self.clone();
        tokio::spawn(async move {
            let mut guard = guard;
            loop {
                manager
                    .peer_evidence()
                    .refresh_requested
                    .store(false, Ordering::Release);
                node.refresh_peer_evidence().await;
                // Release before checking so a concurrent request either owns
                // the next pass or leaves its pending flag for this worker.
                drop(guard);
                if !manager
                    .peer_evidence()
                    .refresh_requested
                    .load(Ordering::Acquire)
                {
                    break;
                }
                let Ok(next) = manager.peer_evidence().refresh.clone().try_lock_owned() else {
                    break;
                };
                guard = next;
            }
        });
    }

    pub(super) async fn maintain_peer_evidence(&self) {
        let mut peer_changes = self.peer_change_rx.clone();
        let mut refreshed_through = std::time::Instant::now();
        self.queue_peer_evidence_refresh().await;
        loop {
            let (deadline, cache_changed) = {
                let manager = self.plugin_manager().await;
                manager.as_ref().map_or((None, None), |manager| {
                    let registry = manager.peer_evidence();
                    (
                        registry.next_refresh_deadline(refreshed_through),
                        Some(registry.cache_changed.clone()),
                    )
                })
            };
            let expiry = async {
                if let Some(deadline) = deadline {
                    tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
                } else {
                    std::future::pending::<()>().await;
                }
            };
            let cache_change = async {
                if let Some(changed) = &cache_changed {
                    changed.notified().await;
                } else {
                    std::future::pending::<()>().await;
                }
            };
            tokio::select! {
                _ = expiry => {
                    refreshed_through = std::time::Instant::now();
                    self.queue_peer_evidence_refresh().await;
                }
                _ = cache_change => {
                    self.peer_change_tx.send_modify(|_| {});
                }
                result = peer_changes.changed() => {
                    if result.is_err() {
                        break;
                    }
                }
            }
        }
    }

    pub(super) async fn require_current_peer_evidence(
        &self,
        peer: EndpointId,
    ) -> anyhow::Result<()> {
        self.require_current_peer_evidence_set(&[peer]).await
    }

    pub(super) async fn require_current_peer_evidence_set(
        &self,
        peers: &[EndpointId],
    ) -> anyhow::Result<()> {
        let Some(manager) = self.plugin_manager().await else {
            return Ok(());
        };
        if peers.is_empty() || !manager.peer_evidence().has_verifiers() {
            return Ok(());
        }
        let mut pins = Vec::with_capacity(peers.len());
        for id in peers {
            let peer = self
                .state
                .lock()
                .await
                .peers
                .get(id)
                .filter(|peer| peer.is_admitted())
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("provider has no direct admission"))?;
            let ann = mesh_llm_membership::announcements::announcement_from_peer(&peer);
            let (canonical, pin) = self
                .evidence_announcement_with_record(*id, &ann, true)
                .await
                .ok_or_else(|| anyhow::anyhow!("provider has no current eligibility decision"))?;
            anyhow::ensure!(
                announcement_is_provider(&canonical),
                "peer does not advertise provider supply"
            );
            pins.push((*id, pin));
        }
        let state = self.state.lock().await;
        let mesh = self
            .mesh_id
            .try_lock()
            .map_err(|_| anyhow::anyhow!("mesh identity is changing"))?;
        let now = super::requirements::current_time_unix_ms();
        let mut views = Vec::with_capacity(pins.len());
        for (id, pin) in pins {
            anyhow::ensure!(
                state.peers.get(&id).is_some_and(PeerInfo::is_admitted),
                "provider lost admission"
            );
            let (record_mesh, evidence) = match (pin, state.node_records.fresh(&id, now)) {
                (Some(expected), Some(held)) if held.signed_bytes() == expected => {
                    let record = NodeRecord::decode(held.body())?;
                    anyhow::ensure!(is_provider(&record), "peer withdrew provider supply");
                    anyhow::ensure!(
                        mesh.as_deref().is_none_or(|mesh| mesh == record.mesh_id),
                        "provider belongs to another mesh"
                    );
                    (record.mesh_id, record.peer_evidence)
                }
                (None, None) if state.has_fresh_direct_admission(&id) => {
                    let peer = &state.peers[&id];
                    anyhow::ensure!(
                        announcement_is_provider(
                            &mesh_llm_membership::announcements::announcement_from_peer(peer)
                        ),
                        "peer withdrew provider supply"
                    );
                    (mesh.clone().unwrap_or_default(), Default::default())
                }
                _ => anyhow::bail!("provider record changed during admission"),
            };
            views.push((record_mesh, id.to_string(), evidence));
        }
        anyhow::ensure!(
            manager.peer_evidence().providers_cached(&views),
            "provider eligibility changed during admission"
        );
        Ok(())
    }

    pub(super) async fn refresh_peer_evidence(&self) {
        let Some(manager) = self.plugin_manager().await else {
            return;
        };
        if !manager.peer_evidence().has_verifiers() {
            return;
        }
        let mesh = self.mesh_id().await;
        let mut records: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .iter()
                .filter_map(|(id, peer)| {
                    let record = state
                        .node_records
                        .fresh(id, super::requirements::current_time_unix_ms())
                        .and_then(|held| NodeRecord::decode(held.body()).ok());
                    let provider = !matches!(peer.role, NodeRole::Client)
                        || !peer.serving_models.is_empty()
                        || !peer.hosted_models.is_empty()
                        || !peer.served_model_descriptors.is_empty()
                        || peer.experts_summary.is_some()
                        || record.as_ref().is_some_and(is_provider);
                    if !provider || (record.is_none() && !state.has_fresh_direct_admission(id)) {
                        return None;
                    }
                    let signed_ann = record.as_ref().and_then(|record| {
                        node_record_to_local(*id, record, None).map(|(_, ann)| ann)
                    });
                    let (record_mesh, evidence) = record.map_or_else(
                        || (mesh.clone().unwrap_or_default(), Default::default()),
                        |record| (record.mesh_id, record.peer_evidence),
                    );
                    (mesh.as_deref().is_none_or(|mesh| mesh == record_mesh)
                        && manager
                            .peer_evidence()
                            .provider_decision(&record_mesh, &id.to_string(), &evidence)
                            .is_none())
                    .then_some((*id, record_mesh, evidence, signed_ann))
                })
                .collect()
        };
        // A slow verifier must not repeatedly starve the same retained records.
        use rand::seq::SliceRandom;
        records.shuffle(&mut rand::rng());
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(12);
        let mut changed = false;
        for (id, record_mesh, evidence, signed_ann) in records {
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            if let Some(ann) = signed_ann
                && !self.provider_supply_allowed(id, &ann).await
            {
                continue;
            }
            manager
                .verify_provider_evidence_until(&record_mesh, &id.to_string(), &evidence, deadline)
                .await;
            changed |= manager
                .peer_evidence()
                .provider_decision(&record_mesh, &id.to_string(), &evidence)
                .is_some();
        }
        if changed {
            let _ = self.peer_change_tx.send(self.peers().await.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn idle_compute_workers_are_subject_to_verifier_policy() {
        let mut record = NodeRecord {
            role: crate::proto::node::NodeRole::Worker as i32,
            ..Default::default()
        };
        assert!(is_provider(&record));
        record.role = crate::proto::node::NodeRole::Client as i32;
        assert!(!is_provider(&record));
        record.serving_models.push("model".into());
        assert!(is_provider(&record));
    }
}
