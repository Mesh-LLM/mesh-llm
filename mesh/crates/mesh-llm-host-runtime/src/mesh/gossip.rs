//! Gossip protocol: peer announcement exchange, transitive peer tracking,
//! and peer list management (add/remove/update).

use super::{
    DEAD_PEER_TTL, InviteTokenMaterial, MeshOperationalEvent, MeshPeerRemovalReason,
    MeshPolicyRejectionReason, Node, PEER_CONNECT_AND_GOSSIP_TIMEOUT, PEER_STALE_SECS,
    PeerAnnouncement, PeerInfo, connect_mesh, elapsed_ms_u64, emit_mesh_info,
    mesh_peer_operational_context, parse_invite_token, record_mesh_operational_event,
    record_mesh_operational_event_with_context,
};
use crate::crypto::{OwnershipSummary, verify_node_ownership};
use crate::mesh::peer_state::policy_accepts_peer;
use crate::mesh::requirements::current_time_unix_ms;
use crate::mesh::stage_transport::PeerLifecycleCaptureEvent;
use crate::protocol::{
    ControlProtocol, NODE_PROTOCOL_GENERATION, STREAM_GOSSIP, connection_protocol,
    decode_gossip_payload, read_len_prefixed, write_gossip_payload,
};
use anyhow::Result;
use iroh::{EndpointAddr, EndpointId, endpoint::Connection};
use mesh_llm_membership::announcements::{
    RebroadcastAnnouncements, peer_is_idle_transitive_client, version_allowed_for_rebroadcast,
};

use mesh_llm_membership::transitions::TransitivePeerUpdate;

mod admission;

/// Minimum peer version we accept into the local mesh table and re-broadcast.
///
/// Peers below this floor are rejected at ingest in both `add_peer`
/// (direct gossip exchange) and `update_transitive_peer` (gossip relayed
/// by a bridge peer). They do not appear in `/api/status`, do not appear
/// in the UI, and are not included in outbound gossip. A peer that updates
/// and re-announces with a version at or above the floor is accepted on
/// the next exchange.
///
/// v0.60.0 is the cut where the on-wire `hardware` block landed; peers
/// older than that predate several gossip fields the current mesh relies
/// on. Peers that don't advertise a version at all (some legacy nodes
/// leave the field unset) are conservatively accepted, on the theory that
/// a missing version is more likely to be a legitimate old node than a
/// targeted bypass.
const MIN_REBROADCAST_VERSION_MAJOR: u64 = 0;
const MIN_REBROADCAST_VERSION_MINOR: u64 = 60;
const CLIENT_AUTO_JOIN_PROBE_LIMIT: usize = 4;
const CLIENT_AUTO_JOIN_PROBE_TIMEOUT: std::time::Duration = PEER_CONNECT_AND_GOSSIP_TIMEOUT;

#[derive(Clone, Copy)]
pub(crate) struct AnnouncedPeerContext {
    remote: EndpointId,
    rtt_ms: Option<u32>,
    negotiated_protocol_generation: Option<u32>,
    direct_peer_requirements_validated: bool,
}

impl AnnouncedPeerContext {
    const fn direct(remote: EndpointId, negotiated_protocol_generation: Option<u32>) -> Self {
        Self {
            remote,
            rtt_ms: None,
            negotiated_protocol_generation,
            direct_peer_requirements_validated: true,
        }
    }
}

pub(crate) struct JoinProbeCandidate {
    token: String,
    mesh_name: Option<String>,
    addr: EndpointAddr,
}

pub(crate) struct JoinProbeSuccess {
    candidate: JoinProbeCandidate,
    conn: Connection,
    announcements: Vec<(EndpointAddr, PeerAnnouncement)>,
    rtt_ms: u32,
    elapsed: std::time::Duration,
}

#[cfg(test)]
impl JoinProbeSuccess {
    /// Test-only constructor so sibling test modules can drive
    /// `commit_join_probe_success` against a real QUIC connection.
    pub(super) fn new_for_tests(
        token: String,
        mesh_name: Option<String>,
        addr: EndpointAddr,
        conn: Connection,
        announcements: Vec<(EndpointAddr, PeerAnnouncement)>,
        rtt_ms: u32,
    ) -> Self {
        Self {
            candidate: JoinProbeCandidate {
                token,
                mesh_name,
                addr,
            },
            conn,
            announcements,
            rtt_ms,
            elapsed: std::time::Duration::from_millis(0),
        }
    }
}

pub(crate) fn emit_join_probe_race_started(candidate_count: usize) {
    tracing::info!(
        candidates = candidate_count,
        timeout_ms = CLIENT_AUTO_JOIN_PROBE_TIMEOUT.as_millis(),
        "Racing auto-join bootstrap candidates"
    );
    emit_mesh_info(format!(
        "Racing {candidate_count} auto-join bootstrap candidates"
    ));
}

pub(crate) fn emit_join_probe_fallback(last_error: Option<&anyhow::Error>) {
    if let Some(error) = last_error {
        tracing::debug!(
            "No auto-join candidate completed the fast probe; falling back to serial join: {error:#}"
        );
    }
    emit_mesh_info(
        "No auto-join candidate completed the fast probe; falling back to serial join".to_string(),
    );
}

impl Node {
    async fn validate_and_capture_inbound_gossip(
        &self,
        protocol: ControlProtocol,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        context: AnnouncedPeerContext,
    ) -> Result<()> {
        self.validate_direct_announcement_before_payload_apply(their_announcements, context)
            .await?;

        let (recovered_from_dead, prior_state) = {
            let mut state = self.state.lock().await;
            let recovered_from_dead = state.dead_peers.remove(&context.remote).is_some();
            state.departed_peers.remove(&context.remote);
            let prior_state = state
                .peers
                .get(&context.remote)
                .map(|peer| {
                    if peer.last_seen >= peer.last_mentioned {
                        "direct"
                    } else {
                        "transitive"
                    }
                })
                .unwrap_or("unknown")
                .to_string();
            if recovered_from_dead {
                super::emit_mesh_info(format!(
                    "🔄 Dead peer {} is gossiping — clearing dead status",
                    context.remote.fmt_short()
                ));
            }
            (recovered_from_dead, prior_state)
        };
        self.capture_gossip_inbound(context.remote, protocol, their_announcements.len());
        self.capture_direct_proof_of_life(
            context.remote,
            protocol,
            their_announcements.len(),
            recovered_from_dead,
            &prior_state,
        );
        Ok(())
    }

    pub(crate) async fn apply_announced_peer(
        &self,
        peer_id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
        context: AnnouncedPeerContext,
    ) -> Result<()> {
        let remote = context.remote;
        if peer_id == self.endpoint.id() {
            // Our own announcement echoed back is normal gossip: peers
            // rebroadcast their full peer table, including us. A peer sharing
            // our key is detected at join time instead, where the token
            // names our own id (#1699).
            return Ok(());
        }
        if peer_id == remote {
            if !context.direct_peer_requirements_validated
                && let Err(reason) = self
                    .validate_direct_peer_requirements(
                        remote,
                        ann,
                        context.negotiated_protocol_generation,
                    )
                    .await
            {
                self.record_mesh_requirement_rejection(
                    super::requirements::MeshRequirementRejectionSource::Gossip,
                    Some(remote),
                    reason.clone(),
                )
                .await;
                self.state
                    .lock()
                    .await
                    .requirement_rejected_peers
                    .insert(remote);
                anyhow::bail!(
                    "peer {} rejected by mesh requirements: {}",
                    remote.fmt_short(),
                    reason.code()
                );
            }
            if self
                .add_peer_after_direct_requirements_validated(
                    remote,
                    addr.clone(),
                    ann,
                    context.negotiated_protocol_generation,
                )
                .await
            {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                if let Some(rtt_ms) = context.rtt_ms {
                    self.update_peer_rtt(remote, rtt_ms).await;
                }
            }
            return Ok(());
        }
        if let Err(err) = self
            .validate_peer_announcement_against_active_policy(peer_id, ann)
            .await
        {
            tracing::debug!(
                "ignoring transitive peer {} because its policy announcement did not match the active mesh: {}",
                peer_id.fmt_short(),
                err.code()
            );
            return Ok(());
        }
        self.update_transitive_peer(peer_id, addr, ann, remote)
            .await;
        Ok(())
    }

    pub(crate) async fn apply_announced_peers(
        &self,
        remote: EndpointId,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        rtt_ms: Option<u32>,
        negotiated_protocol_generation: Option<u32>,
        direct_peer_requirements_validated: bool,
    ) -> Result<()> {
        let context = AnnouncedPeerContext {
            remote,
            rtt_ms,
            negotiated_protocol_generation,
            direct_peer_requirements_validated,
        };
        if !direct_peer_requirements_validated {
            self.validate_direct_announcement_before_payload_apply(their_announcements, context)
                .await?;
        } else {
            // Requirement validation is not an ownership admission proof.
            self.validate_direct_owner_before_payload_apply(their_announcements, context)
                .await?;
        }
        let context = AnnouncedPeerContext {
            direct_peer_requirements_validated: true,
            ..context
        };
        for (addr, ann) in their_announcements {
            self.apply_announced_peer(addr.id, addr, ann, context)
                .await?;
        }
        Ok(())
    }

    pub(crate) async fn refresh_gossip_path_rtt(
        &self,
        remote: EndpointId,
        ceiling_rtt_ms: Option<u32>,
    ) {
        let conn = self.state.lock().await.connections.get(&remote).cloned();
        let Some(conn) = conn else {
            return;
        };
        self.refresh_gossip_path_rtt_for_connection(remote, &conn, ceiling_rtt_ms)
            .await;
    }

    pub(crate) async fn refresh_gossip_path_rtt_for_connection(
        &self,
        remote: EndpointId,
        conn: &Connection,
        ceiling_rtt_ms: Option<u32>,
    ) {
        let capture_source = if ceiling_rtt_ms.is_some() {
            "gossip_round_trip_path"
        } else {
            "inbound_gossip_path"
        };
        let Some(observation) = self.capture_selected_connection_path(remote, conn, capture_source)
        else {
            return;
        };
        if let Some(path_rtt_ms) = observation.rtt_ms {
            if ceiling_rtt_ms.is_some_and(|ceiling| path_rtt_ms >= ceiling) {
                self.update_peer_selected_path(remote, observation).await;
                return;
            }
            super::emit_mesh_info(format!(
                "📡 Peer {} RTT: {}ms ({}){}",
                remote.fmt_short(),
                path_rtt_ms,
                observation.path_type,
                if ceiling_rtt_ms.is_some() {
                    " [path info]"
                } else {
                    ""
                }
            ));
        }
        self.update_peer_selected_path(remote, observation).await;
    }

    pub(crate) async fn maybe_connect_discovered_peer(
        &self,
        my_role: &super::NodeRole,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        known_peer_check_uses_connections: bool,
        log_discovery_failure_as_warning: bool,
    ) {
        let peer_id = addr.id;
        if self.should_skip_discovered_peer(my_role, peer_id, ann)
            || self
                .discovered_peer_already_known(peer_id, known_peer_check_uses_connections)
                .await
            || Self::discovered_peer_is_filtered(peer_id, ann)
        {
            return;
        }
        if let Err(error) = Box::pin(self.connect_to_peer(addr)).await {
            if log_discovery_failure_as_warning {
                tracing::warn!("Failed to discover peer: {error}");
            } else {
                tracing::debug!(
                    "Could not connect to discovered peer {}: {error}",
                    peer_id.fmt_short()
                );
            }
        }
    }

    pub(crate) async fn connect_discovered_peers(
        &self,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        known_peer_check_uses_connections: bool,
        log_discovery_failure_as_warning: bool,
    ) {
        let my_role = self.role.lock().await.clone();
        for (addr, ann) in their_announcements {
            self.maybe_connect_discovered_peer(
                &my_role,
                addr.clone(),
                ann,
                known_peer_check_uses_connections,
                log_discovery_failure_as_warning,
            )
            .await;
        }
    }

    pub(crate) fn spawn_discovered_peer_connects(
        &self,
        their_announcements: Vec<(EndpointAddr, PeerAnnouncement)>,
        known_peer_check_uses_connections: bool,
        log_discovery_failure_as_warning: bool,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            node.connect_discovered_peers(
                &their_announcements,
                known_peer_check_uses_connections,
                log_discovery_failure_as_warning,
            )
            .await;
        });
    }

    /// Returns `true` if the announcement would be rejected by the same
    /// gates that filter ingest. Skipping the dial here avoids spending
    /// 30s per host walking through unreachable ghost addresses
    /// sequentially in the gossip exchange dial loop — the wedge that
    /// caused `--auto` startup to hang.
    pub(crate) fn discovered_peer_is_filtered(peer_id: EndpointId, ann: &PeerAnnouncement) -> bool {
        if !version_allowed_for_rebroadcast(ann.version.as_deref())
            || peer_is_idle_transitive_client(ann)
        {
            tracing::debug!(
                "Skipping discovered peer {} (filtered: version={:?} role={:?})",
                peer_id.fmt_short(),
                ann.version,
                ann.role
            );
            return true;
        }
        false
    }

    pub(crate) fn should_skip_discovered_peer(
        &self,
        my_role: &super::NodeRole,
        peer_id: EndpointId,
        ann: &PeerAnnouncement,
    ) -> bool {
        peer_id == self.endpoint.id()
            || (matches!(my_role, super::NodeRole::Client)
                && matches!(ann.role, super::NodeRole::Client))
    }

    pub(crate) async fn discovered_peer_already_known(
        &self,
        peer_id: EndpointId,
        use_connections: bool,
    ) -> bool {
        let mut state = self.state.lock().await;
        if state.pending_connection_is_active(peer_id) {
            return true;
        }
        if use_connections {
            state.connections.contains_key(&peer_id)
        } else {
            state.peers.contains_key(&peer_id)
        }
    }

    pub(crate) async fn remove_disallowed_peer(&self, id: EndpointId) {
        let mut state = self.state.lock().await;
        if let Some(admitted_count) = state.remove_disallowed_peer(id) {
            let _ = self.peer_change_tx.send(admitted_count);
        }
    }

    pub(crate) async fn direct_peer_owner_summary(
        &self,
        id: EndpointId,
        ann: &PeerAnnouncement,
    ) -> OwnershipSummary {
        let trust_store = self.trust_store.lock().await.clone();
        verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        )
    }

    pub(crate) async fn reject_direct_peer_for_policy(
        &self,
        id: EndpointId,
        owner_summary: &OwnershipSummary,
    ) -> bool {
        if policy_accepts_peer(self.trust_policy, owner_summary) {
            return false;
        }

        let mut state = self.state.lock().await;
        let last_status = state.policy_rejected_peers.get(&id).cloned();
        let newly_rejected = last_status.as_ref() != Some(&owner_summary.status);
        if newly_rejected {
            tracing::warn!(
                "Rejecting peer {} due to owner policy: {:?}",
                id.fmt_short(),
                owner_summary.status
            );
            state
                .policy_rejected_peers
                .insert(id, owner_summary.status.clone());
        }
        if let Some(admitted_count) = state.remove_disallowed_peer(id) {
            let _ = self.peer_change_tx.send(admitted_count);
        }
        drop(state);
        if newly_rejected
            && let Some(reason) =
                MeshPolicyRejectionReason::from_ownership_status(&owner_summary.status)
        {
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::GossipPolicyRejected(reason),
                mesh_peer_operational_context(id, self.authenticated_peer_path(id).await),
            );
        }
        true
    }

    pub(crate) async fn publish_direct_peer_update(
        &self,
        updated_peer: PeerInfo,
        changed: bool,
        should_publish_count: bool,
        count: usize,
    ) {
        let capture_event = if should_publish_count {
            "peer_direct_update"
        } else {
            "peer_direct_seen"
        };
        self.capture_peer_observation(capture_event, &updated_peer, "direct", None);
        if should_publish_count {
            let _ = self.peer_change_tx.send(count);
        }
        if changed {
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                Some(&updated_peer),
                String::new(),
            )
            .await;
        }
    }

    pub(crate) async fn upsert_existing_direct_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
        now: std::time::Instant,
    ) -> bool {
        let mut state = self.state.lock().await;
        let Some(update) = state.upsert_existing_direct_peer(id, addr, ann, owner_summary, now)
        else {
            return false;
        };
        drop(state);
        self.publish_direct_peer_update(
            update.peer,
            update.changed,
            update.publish_count,
            update.admitted_count,
        )
        .await;
        true
    }

    pub(crate) async fn insert_new_direct_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
    ) {
        let mut state = self.state.lock().await;
        tracing::info!(
            "Peer added: {} role={:?} vram={:.1}GB assigned={:?} catalog={:?} (total: {})",
            id.fmt_short(),
            ann.role,
            ann.vram_bytes as f64 / 1e9,
            ann.serving_models.first(),
            ann.available_models,
            state.peers.len() + 1
        );
        let (peer, count) = state.insert_new_direct_peer(id, addr, ann, owner_summary);
        drop(state);
        self.capture_peer_observation("peer_direct_add", &peer, "direct", None);
        record_mesh_operational_event_with_context(
            MeshOperationalEvent::GossipDirectPeerPromoted,
            mesh_peer_operational_context(id, self.authenticated_peer_path(id).await)
                .numeric_summary("direct_peers", count as u64),
        );
        let _ = self.peer_change_tx.send(count);
        self.emit_plugin_mesh_event(
            crate::plugin::proto::mesh_event::Kind::PeerUp,
            Some(&peer),
            String::new(),
        )
        .await;
    }

    /// Open a gossip stream on an existing connection to exchange peer info.
    pub(super) async fn initiate_gossip(&self, conn: Connection, remote: EndpointId) -> Result<()> {
        // Timeout only the gossip round-trip. A misbehaving peer may accept the
        // QUIC connection and even the bi-stream but never send a gossip response,
        // blocking the join path indefinitely and preventing fallback to other
        // candidates.
        match tokio::time::timeout(
            PEER_CONNECT_AND_GOSSIP_TIMEOUT,
            self.gossip_round_trip(&conn, remote),
        )
        .await
        {
            Ok(Ok((their_announcements, rtt_ms))) => {
                self.apply_gossip_announcements(remote, rtt_ms, &their_announcements, true)
                    .await?;
                if !self.state.lock().await.connections.contains_key(&remote) {
                    self.refresh_gossip_path_rtt_for_connection(remote, &conn, Some(rtt_ms))
                        .await;
                }
                Ok(())
            }
            Ok(Err(e)) => Err(e),
            Err(_) => anyhow::bail!(
                "gossip exchange with {} timed out ({}s)",
                remote.fmt_short(),
                PEER_CONNECT_AND_GOSSIP_TIMEOUT.as_secs()
            ),
        }
    }

    pub(crate) async fn join_first_responsive_candidate(
        &self,
        join_attempts: &[(String, Option<String>)],
    ) -> Result<Option<(String, Option<String>)>> {
        let candidates = self.collect_join_probe_candidates(join_attempts).await;
        if candidates.len() <= 1 {
            tracing::debug!(
                valid_candidates = candidates.len(),
                "auto-join probe skipped"
            );
            return Ok(None);
        }

        emit_join_probe_race_started(candidates.len());
        match self.race_join_probe_candidates(candidates).await {
            Some(success) => self.commit_join_probe_success(success).await.map(Some),
            None => Ok(None),
        }
    }

    pub(crate) async fn collect_join_probe_candidates(
        &self,
        join_attempts: &[(String, Option<String>)],
    ) -> Vec<JoinProbeCandidate> {
        if join_attempts.len() <= 1 {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let mut invalid = 0usize;
        for (token, mesh_name) in join_attempts.iter().take(CLIENT_AUTO_JOIN_PROBE_LIMIT) {
            match self
                .prepare_join_probe_candidate(token, mesh_name.clone())
                .await
            {
                Ok(Some(candidate)) => candidates.push(candidate),
                Ok(None) => {}
                Err(error) => {
                    invalid += 1;
                    tracing::debug!("Skipping invalid auto-join candidate: {error:#}");
                }
            }
        }
        tracing::debug!(
            valid_candidates = candidates.len(),
            invalid_candidates = invalid,
            "collected auto-join probe candidates"
        );
        candidates
    }

    pub(crate) async fn race_join_probe_candidates(
        &self,
        candidates: Vec<JoinProbeCandidate>,
    ) -> Option<JoinProbeSuccess> {
        let mut probes = tokio::task::JoinSet::new();
        for candidate in candidates {
            let node = self.clone();
            probes.spawn(async move { node.probe_join_candidate(candidate).await });
        }

        let mut last_error = None;
        while let Some(result) = probes.join_next().await {
            match result {
                Ok(Ok(success)) => {
                    probes.abort_all();
                    return Some(success);
                }
                Ok(Err(error)) => {
                    tracing::debug!("auto-join candidate probe failed: {error:#}");
                    last_error = Some(error);
                }
                Err(error) => {
                    tracing::debug!("auto-join candidate probe task failed: {error:#}");
                }
            }
        }

        emit_join_probe_fallback(last_error.as_ref());
        None
    }

    pub(crate) async fn prepare_join_probe_candidate(
        &self,
        token: &str,
        mesh_name: Option<String>,
    ) -> Result<Option<JoinProbeCandidate>> {
        let addr = match parse_invite_token(token)
            .map_err(|reason| anyhow::anyhow!("join rejected: {}", reason.code()))?
        {
            InviteTokenMaterial::Legacy(addr) => addr,
            // Requirement-aware bootstrap tokens may require installing the
            // signed policy before gossip. Keep those on the established
            // serial join path rather than probing them out-of-band.
            InviteTokenMaterial::Signed(_) => return Ok(None),
        };

        if addr.id == self.endpoint.id() {
            return Ok(None);
        }

        let mut state = self.state.lock().await;
        if state.connections.contains_key(&addr.id) || state.pending_connection_is_active(addr.id) {
            return Ok(None);
        }
        if state
            .dead_peers
            .get(&addr.id)
            .is_some_and(|t| t.elapsed() < DEAD_PEER_TTL)
        {
            return Ok(None);
        }
        drop(state);

        Ok(Some(JoinProbeCandidate {
            token: token.to_string(),
            mesh_name,
            addr,
        }))
    }

    pub(crate) async fn probe_join_candidate(
        &self,
        candidate: JoinProbeCandidate,
    ) -> Result<JoinProbeSuccess> {
        let peer_id = candidate.addr.id;
        let started = std::time::Instant::now();
        let result = tokio::time::timeout(CLIENT_AUTO_JOIN_PROBE_TIMEOUT, async {
            let conn = connect_mesh(&self.endpoint, candidate.addr.clone()).await?;
            let (announcements, rtt_ms) = self.gossip_round_trip(&conn, peer_id).await?;
            Ok::<_, anyhow::Error>((conn, announcements, rtt_ms))
        })
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "candidate {} timed out after {}s",
                peer_id.fmt_short(),
                CLIENT_AUTO_JOIN_PROBE_TIMEOUT.as_secs()
            )
        })??;

        Ok(JoinProbeSuccess {
            candidate,
            conn: result.0,
            announcements: result.1,
            rtt_ms: result.2,
            elapsed: started.elapsed(),
        })
    }

    pub(super) async fn commit_join_probe_success(
        &self,
        success: JoinProbeSuccess,
    ) -> Result<(String, Option<String>)> {
        let JoinProbeSuccess {
            candidate,
            conn,
            announcements,
            rtt_ms,
            elapsed,
        } = success;
        let peer_id = candidate.addr.id;

        {
            let mut state = self.state.lock().await;
            state.dead_peers.remove(&peer_id);
            state.departed_peers.remove(&peer_id);
            state.connections.insert(peer_id, conn.clone());
        }
        let node_for_dispatch = self.clone();
        let conn_for_dispatch = conn.clone();
        tokio::spawn(async move {
            node_for_dispatch
                .dispatch_streams(conn_for_dispatch, peer_id)
                .await;
        });

        if let Err(error) = self
            .apply_gossip_announcements(peer_id, rtt_ms, &announcements, false)
            .await
        {
            // Drop the tracked entry AND close the QUIC connection. The
            // dispatcher task above holds its own `conn` clone, so removing the
            // map entry alone would leave a live, keep-alive'd connection and a
            // running dispatcher for a peer nobody tracks (and one whose
            // close-recovery path could even reconnect it). Closing here makes
            // the dispatcher's `accept_*` calls error so it unwinds cleanly.
            self.state.lock().await.connections.remove(&peer_id);
            conn.close(0u32.into(), b"join announcement-apply failed");
            record_mesh_operational_event(MeshOperationalEvent::AutoJoinFailed);
            return Err(error);
        }

        // Match `connect_to_peer`: the probe gossip RTT above likely reflects
        // relay latency, so refresh the selected-path/RTT after holepunch.
        self.schedule_selected_path_recheck(peer_id);
        self.spawn_discovered_peer_connects(announcements, true, false);

        tracing::info!(
            peer = %peer_id.fmt_short(),
            elapsed_ms = elapsed_ms_u64(elapsed),
            rtt_ms,
            "Fast auto-join probe selected bootstrap candidate"
        );
        emit_mesh_info(format!(
            "Fast auto-join selected peer {} in {}ms",
            peer_id.fmt_short(),
            elapsed_ms_u64(elapsed)
        ));
        record_mesh_operational_event(MeshOperationalEvent::AutoJoinSucceeded);

        Ok((candidate.token, candidate.mesh_name))
    }

    pub(super) async fn initiate_gossip_inner(
        &self,
        conn: Connection,
        remote: EndpointId,
        discover_peers: bool,
    ) -> Result<()> {
        let (their_announcements, rtt_ms) = self.gossip_round_trip(&conn, remote).await?;
        self.apply_gossip_announcements(remote, rtt_ms, &their_announcements, discover_peers)
            .await?;
        if !self.state.lock().await.connections.contains_key(&remote) {
            self.refresh_gossip_path_rtt_for_connection(remote, &conn, Some(rtt_ms))
                .await;
        }
        Ok(())
    }

    pub(crate) async fn gossip_round_trip(
        &self,
        conn: &Connection,
        remote: EndpointId,
    ) -> Result<(Vec<(EndpointAddr, PeerAnnouncement)>, u32)> {
        let protocol = connection_protocol(conn);
        let t0 = std::time::Instant::now();
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_GOSSIP]).await?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let buf = read_len_prefixed(&mut recv).await?;
        let rtt_ms = t0.elapsed().as_millis() as u32;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;

        let _ = recv.read_to_end(0).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Ok((their_announcements, rtt_ms))
    }

    pub(crate) async fn apply_gossip_announcements(
        &self,
        remote: EndpointId,
        rtt_ms: u32,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        discover_peers: bool,
    ) -> Result<()> {
        self.apply_announced_peers(
            remote,
            their_announcements,
            Some(rtt_ms),
            Some(NODE_PROTOCOL_GENERATION),
            false,
        )
        .await?;

        // Also check the connection's actual path info — the gossip round-trip
        // time above may reflect relay latency even if a direct path is now active.
        self.refresh_gossip_path_rtt(remote, Some(rtt_ms)).await;

        if discover_peers {
            self.connect_discovered_peers(their_announcements, true, false)
                .await;
        }

        Ok(())
    }

    pub(super) async fn handle_gossip_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        tracing::info!("Inbound gossip from {}", remote.fmt_short());

        let buf = read_len_prefixed(&mut recv).await?;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;
        let negotiated_protocol_generation = match protocol {
            ControlProtocol::ProtoV1 => Some(NODE_PROTOCOL_GENERATION),
        };
        let context = AnnouncedPeerContext::direct(remote, negotiated_protocol_generation);
        self.validate_and_capture_inbound_gossip(protocol, &their_announcements, context)
            .await?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let _ = recv.read_to_end(0).await;

        self.apply_announced_peers(
            remote,
            &their_announcements,
            None,
            negotiated_protocol_generation,
            true,
        )
        .await?;
        self.refresh_gossip_path_rtt(remote, None).await;

        self.connect_discovered_peers(&their_announcements, false, true)
            .await;

        Ok(())
    }
    pub(super) async fn remove_peer(&self, id: EndpointId, reason: MeshPeerRemovalReason) {
        let mut state = self.state.lock().await;
        if let Some(removed) = state.remove_peer(id) {
            let peer = removed.peer;
            let had_connection = removed.had_connection;
            let last_seen_age_ms = super::elapsed_ms_u64(peer.last_seen.elapsed());
            let last_mentioned_age_ms = super::elapsed_ms_u64(peer.last_mentioned.elapsed());
            let bridge_id = peer
                .propagated_latency
                .as_ref()
                .and_then(|latency| latency.observer_id);
            tracing::info!(
                "Peer removed: {} (total: {})",
                id.fmt_short(),
                removed.remaining_count
            );
            let count = removed.admitted_count;
            drop(state);
            self.capture_peer_lifecycle_event(PeerLifecycleCaptureEvent {
                event: "peer_removed",
                peer: id,
                reason: reason.reason_code(),
                reporter: None,
                last_seen_age_ms: Some(last_seen_age_ms),
                last_mentioned_age_ms: Some(last_mentioned_age_ms),
                had_connection: Some(had_connection),
                bridge_id,
            });
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::GossipPeerRemoved(reason),
                mesh_peer_operational_context(id, peer.selected_path)
                    .numeric_summary("direct_peers", count as u64),
            );
            let _ = self.peer_change_tx.send(count);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerDown,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    #[cfg(test)]
    pub(super) async fn add_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        negotiated_protocol_generation: Option<u32>,
    ) {
        if let Err(reason) = self
            .validate_direct_peer_requirements(id, ann, negotiated_protocol_generation)
            .await
        {
            self.record_mesh_requirement_rejection(
                super::requirements::MeshRequirementRejectionSource::Gossip,
                Some(id),
                reason.clone(),
            )
            .await;
            tracing::warn!(
                "Rejecting peer {} before promotion: {}",
                id.fmt_short(),
                reason.code()
            );
            let mut state = self.state.lock().await;
            state.requirement_rejected_peers.insert(id);
            if let Some(admitted_count) = state.remove_disallowed_peer(id) {
                let _ = self.peer_change_tx.send(admitted_count);
            }
            return;
        }
        self.add_peer_after_direct_requirements_validated(
            id,
            addr,
            ann,
            negotiated_protocol_generation,
        )
        .await;
    }

    pub(crate) async fn add_peer_after_direct_requirements_validated(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        _negotiated_protocol_generation: Option<u32>,
    ) -> bool {
        // Reject ingest from peers below the supported version floor. They
        // are not added to local state, do not appear in /api/status, and
        // are not re-broadcast. A peer that updates and re-announces will
        // be accepted on the next exchange.
        if !version_allowed_for_rebroadcast(ann.version.as_deref()) {
            tracing::debug!(
                "Refusing direct peer {} below version floor (advertised {:?})",
                id.fmt_short(),
                ann.version
            );
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::GossipIncompatibleVersionRejected,
                mesh_peer_operational_context(id, self.authenticated_peer_path(id).await),
            );
            self.remove_disallowed_peer(id).await;
            return false;
        }
        let owner_summary = self.direct_peer_owner_summary(id, ann).await;
        if self.reject_direct_peer_for_policy(id, &owner_summary).await {
            self.capture_peer_rejected(id, &addr, ann, &owner_summary, "direct", None);
            return false;
        }
        let mut state = self.state.lock().await;
        state.policy_rejected_peers.remove(&id);
        state.requirement_rejected_peers.remove(&id);
        if id == self.endpoint.id() {
            return false;
        }
        let now = std::time::Instant::now();
        // If this peer was previously dead, clear it — add_peer is only called
        // after a successful gossip exchange, which is proof of life.
        let recovered = state.dead_peers.remove(&id).is_some();
        state.departed_peers.remove(&id);
        if recovered {
            super::emit_mesh_info(format!(
                "🔄 Peer {} back from the dead (successful gossip)",
                id.fmt_short()
            ));
        }
        let peer_exists = state.peers.contains_key(&id);
        drop(state);
        if peer_exists
            && self
                .upsert_existing_direct_peer(id, addr.clone(), ann, owner_summary.clone(), now)
                .await
        {
            return true;
        }
        self.insert_new_direct_peer(id, addr, ann, owner_summary)
            .await;
        true
    }

    /// Update a peer learned transitively through gossip (not directly connected).
    /// Updates assigned/hosted state so models_being_served() includes their models.
    /// Refreshes `last_mentioned` (not `last_seen`) so the peer survives pruning
    /// and gossip propagation as long as a bridge peer keeps mentioning it, but
    /// PeerDown silencing uses only `last_seen` (direct proof-of-life).
    /// Does NOT trigger peer_change events for new transitive peers
    /// (avoids re-election storms at scale).
    pub(super) async fn update_transitive_peer(
        &self,
        id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
        bridge_id: EndpointId,
    ) {
        // Refuse transitive ingest from peers below the supported version
        // floor. Keeps the local table free of pre-floor gossip filler;
        // /api/status, the UI, and routing all stop seeing them.
        if !version_allowed_for_rebroadcast(ann.version.as_deref()) {
            let mut state = self.state.lock().await;
            if let Some(admitted_count) = state.remove_disallowed_peer(id) {
                let _ = self.peer_change_tx.send(admitted_count);
            }
            return;
        }
        // Refuse transitive ingest of idle clients — clients that aren't
        // asking for any model, aren't serving anything, and aren't hosting
        // anything. They contribute nothing the mesh can use:
        //   - not routable to (no model to serve)
        //   - not findable (clients-don't-dial-clients by design)
        //   - no demand signal (empty requested_models)
        //   - not relaying for us (no connection — purely transitive)
        // The moment any of those become non-empty, this filter stops firing
        // and the peer is admitted normally. Direct connections (`add_peer`)
        // are never affected — a client that actually contacts us still
        // gets in.
        if peer_is_idle_transitive_client(ann) {
            let mut state = self.state.lock().await;
            if let Some(admitted_count) = state.remove_disallowed_peer(id) {
                let _ = self.peer_change_tx.send(admitted_count);
            }
            return;
        }
        let trust_store = self.trust_store.lock().await.clone();
        let owner_summary = verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        );
        if !policy_accepts_peer(self.trust_policy, &owner_summary) {
            let mut state = self.state.lock().await;
            if let Some(admitted_count) = state.remove_disallowed_peer(id) {
                let _ = self.peer_change_tx.send(admitted_count);
            }
            drop(state);
            self.capture_peer_rejected(
                id,
                addr,
                ann,
                &owner_summary,
                "transitive",
                Some(bridge_id),
            );
            return;
        }
        let update = self.state.lock().await.apply_accepted_transitive_peer(
            self.endpoint.id(),
            id,
            addr,
            ann,
            bridge_id,
            owner_summary,
        );
        match update {
            TransitivePeerUpdate::Ignored => {}
            TransitivePeerUpdate::Added(peer) => {
                self.capture_peer_observation(
                    "peer_transitive_add",
                    &peer,
                    "transitive",
                    Some(bridge_id),
                );
                self.emit_plugin_mesh_event(
                    crate::plugin::proto::mesh_event::Kind::PeerUp,
                    Some(&peer),
                    String::new(),
                )
                .await;
            }
            TransitivePeerUpdate::Updated {
                peer,
                changed,
                admitted_count,
            } => {
                let event = if admitted_count.is_some() {
                    "peer_transitive_update"
                } else {
                    "peer_transitive_seen"
                };
                self.capture_peer_observation(event, &peer, "transitive", Some(bridge_id));
                if let Some(count) = admitted_count {
                    let _ = self.peer_change_tx.send(count);
                }
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&peer),
                        String::new(),
                    )
                    .await;
                }
            }
        }
    }

    pub(super) async fn collect_announcements(&self) -> Vec<PeerAnnouncement> {
        let stale_cutoff =
            std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS);
        let local = self.snapshot_local_announcement_data().await;
        let RebroadcastAnnouncements {
            mut announcements,
            filtered_old_version,
        } = self.collect_rebroadcast_announcements(stale_cutoff).await;
        if filtered_old_version > 0 {
            tracing::debug!(
                filtered = filtered_old_version,
                "gossip: omitting {} peer(s) below v{}.{}.0 from outbound rebroadcast",
                filtered_old_version,
                MIN_REBROADCAST_VERSION_MAJOR,
                MIN_REBROADCAST_VERSION_MINOR,
            );
        }
        announcements.push(self.build_local_announcement(local));
        announcements
    }
}

#[cfg(test)]
mod tests {
    // The announcement builders moved to `super::super::announcements`; the
    // tests here still construct the types they take.
    use crate::mesh::{ModelDemand, NodeRole};

    include!("tests/gossip.rs");
}
