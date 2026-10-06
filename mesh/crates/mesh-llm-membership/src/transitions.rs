//! Peer table transitions; the host publishes their resulting observations and events.

use crate::announcements::{
    apply_transitive_ann, peer_meaningfully_changed, update_existing_direct_peer,
};
use crate::peer_state::{
    DEAD_PEER_TTL, DEPARTED_PEER_TRANSITIVE_BLOCK_TTL, PEER_DOWN_REPORTER_COOLDOWN_SECS,
    PEER_STALE_SECS, PeerAnnouncement, PeerInfo,
};
use crate::state::MembershipState;
use iroh::{EndpointAddr, EndpointId};
use mesh_llm_identity::OwnershipSummary;
use std::time::{Duration, Instant};

pub struct DirectPeerUpdate {
    pub peer: PeerInfo,
    pub changed: bool,
    pub publish_count: bool,
    pub admitted_count: usize,
}

pub enum TransitivePeerUpdate {
    Ignored,
    Added(PeerInfo),
    Updated {
        peer: PeerInfo,
        changed: bool,
        admitted_count: Option<usize>,
    },
}

impl MembershipState {
    pub fn admitted_peer_count(&self) -> usize {
        self.peers
            .values()
            .filter(|peer| peer.is_admitted())
            .count()
    }

    /// Remove only the announcement, retaining connection and rejection state.
    pub fn remove_disallowed_peer(&mut self, id: EndpointId) -> Option<usize> {
        self.peers.remove(&id)?;
        Some(self.admitted_peer_count())
    }

    pub fn upsert_existing_direct_peer(
        &mut self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
        now: Instant,
    ) -> Option<DirectPeerUpdate> {
        self.policy_rejected_peers.remove(&id);
        let existing = self.peers.get_mut(&id)?;
        let (peer, changed, role_changed, serving_changed) =
            update_existing_direct_peer(existing, addr, ann, owner_summary, now);
        Some(DirectPeerUpdate {
            peer,
            changed,
            publish_count: role_changed || serving_changed,
            admitted_count: self.admitted_peer_count(),
        })
    }

    pub fn insert_new_direct_peer(
        &mut self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
    ) -> (PeerInfo, usize) {
        self.policy_rejected_peers.remove(&id);
        let mut peer = PeerInfo::from_announcement(id, addr, ann, owner_summary);
        peer.admitted = true;
        self.peers.insert(id, peer.clone());
        (peer, self.admitted_peer_count())
    }

    /// Apply a transitive announcement after version, demand and ownership checks.
    /// A bridge refreshes mention age, never direct proof-of-life or admission.
    pub fn apply_accepted_transitive_peer(
        &mut self,
        local_id: EndpointId,
        id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
        bridge_id: EndpointId,
        owner_summary: OwnershipSummary,
    ) -> TransitivePeerUpdate {
        if id == local_id
            || self
                .dead_peers
                .get(&id)
                .is_some_and(|t| t.elapsed() < DEAD_PEER_TTL)
            || self
                .departed_peers
                .get(&id)
                .is_some_and(|t| t.elapsed() < DEPARTED_PEER_TRANSITIVE_BLOCK_TTL)
        {
            return TransitivePeerUpdate::Ignored;
        }
        if let Some(existing) = self.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let serving_changed = apply_transitive_ann(existing, addr, ann, bridge_id);
            existing.owner_summary = owner_summary;
            existing.last_mentioned = Instant::now();
            let peer = existing.clone();
            let changed = peer_meaningfully_changed(&old_peer, &peer);
            TransitivePeerUpdate::Updated {
                peer,
                changed,
                admitted_count: serving_changed.then(|| self.admitted_peer_count()),
            }
        } else {
            let mut peer = PeerInfo::from_announcement(id, addr.clone(), ann, owner_summary);
            // Capability provenance must be direct: a bridge cannot grant eligibility.
            peer.local_gguf_content_id_supported = false;
            peer.decode_batch_policy_supported = false;
            peer.stage_protocol_generation_supported = false;
            peer.admitted = false;
            peer.last_seen = Instant::now() - Duration::from_secs(PEER_STALE_SECS * 2);
            self.peers.insert(id, peer.clone());
            TransitivePeerUpdate::Added(peer)
        }
    }
}

pub struct RemovedPeer {
    pub peer: PeerInfo,
    pub had_connection: bool,
    pub admitted_count: usize,
    pub remaining_count: usize,
}

impl MembershipState {
    /// Clear admission rejection history even when the peer was already removed.
    /// Connections and dead-peer quarantine have independent recovery lifetimes.
    pub fn remove_peer(&mut self, id: EndpointId) -> Option<RemovedPeer> {
        self.policy_rejected_peers.remove(&id);
        let had_connection = self.connections.contains_key(&id);
        self.requirement_rejected_peers.remove(&id);
        let peer = self.peers.remove(&id)?;
        Some(RemovedPeer {
            peer,
            had_connection,
            admitted_count: self.admitted_peer_count(),
            remaining_count: self.peers.len(),
        })
    }

    pub fn stale_peers(&self, cutoff: Instant) -> Vec<EndpointId> {
        self.peers
            .iter()
            .filter(|(_, peer)| peer.last_seen < cutoff && peer.last_mentioned < cutoff)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Expire quarantine and reporter/direct-path cooldowns without touching peers.
    pub fn retain_live_heartbeat_state(
        &mut self,
        direct_path_cooldown: Duration,
    ) -> Vec<EndpointId> {
        let expired: Vec<_> = self
            .dead_peers
            .iter()
            .filter_map(|(id, ts)| (ts.elapsed() >= DEAD_PEER_TTL).then_some(*id))
            .collect();
        self.dead_peers.retain(|_, ts| ts.elapsed() < DEAD_PEER_TTL);
        self.departed_peers
            .retain(|_, ts| ts.elapsed() < DEPARTED_PEER_TRANSITIVE_BLOCK_TTL);
        self.peer_down_rejections
            .retain(|_, ts| ts.elapsed().as_secs() < PEER_DOWN_REPORTER_COOLDOWN_SECS);
        self.direct_path_request_last_at
            .retain(|_, ts| ts.elapsed() < direct_path_cooldown);
        expired
    }
}

#[cfg(test)]
mod tests;
