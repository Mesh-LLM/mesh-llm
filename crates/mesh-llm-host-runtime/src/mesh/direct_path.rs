//! Direct UDP path maintenance for mesh peers.
//!
//! This module keeps direct-path repair close to the iroh/mesh layer. It is
//! deliberately targeted: one peer per tick, per-peer cooldowns on both sender
//! and receiver, and no gossip fanout.

use super::{
    ConnectionCaptureEvent, NODE_PROTOCOL_GENERATION, Node, connect_mesh, connection_protocol,
    endpoint_id_hex, heartbeat,
};
use crate::protocol::{
    STREAM_DIRECT_PATH_REQUEST, ValidateControlFrame, read_len_prefixed, write_len_prefixed,
};
use anyhow::{Context, Result};
use iroh::{EndpointAddr, EndpointId, TransportAddr, endpoint::Connection};
use prost::Message;
use std::collections::HashMap;

pub(super) const DIRECT_PATH_MAINTENANCE_CHECK_SECS: u64 = 30;
pub(super) const DIRECT_PATH_REPAIR_GRACE_SECS: u64 = 15;
pub(super) const DIRECT_PATH_REPAIR_COOLDOWN_SECS: u64 = 120;
pub(super) const DIRECT_PATH_REQUEST_COOLDOWN_SECS: u64 = 120;
const DIRECT_PATH_REQUEST_TIMEOUT_SECS: u64 = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DirectPathRepairReason {
    RelaySelected,
    UnknownSelected,
}

impl DirectPathRepairReason {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::RelaySelected => "selected path is relay",
            Self::UnknownSelected => "selected path is unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DirectPathPeerHealth {
    pub(super) non_direct_since: Option<std::time::Instant>,
    pub(super) last_request_at: Option<std::time::Instant>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectPathObservation {
    pub(super) peer_id: EndpointId,
    pub(super) snapshot: heartbeat::RelayPathSnapshot,
    pub(super) has_direct_candidate: bool,
}

#[derive(Default)]
pub(super) struct DirectPathMaintenanceController {
    peer_health: HashMap<EndpointId, DirectPathPeerHealth>,
}

impl DirectPathMaintenanceController {
    pub(super) fn plan_request<I>(
        &mut self,
        observations: I,
        now: std::time::Instant,
        inflight_requests: u64,
    ) -> Option<(EndpointId, DirectPathRepairReason)>
    where
        I: IntoIterator<Item = DirectPathObservation>,
    {
        let mut observations: Vec<DirectPathObservation> = observations.into_iter().collect();
        observations.sort_by_key(|observation| endpoint_id_hex(observation.peer_id));

        if observations.is_empty() {
            self.peer_health.clear();
            return None;
        }

        let active_peers: std::collections::HashSet<EndpointId> = observations
            .iter()
            .map(|observation| observation.peer_id)
            .collect();
        self.peer_health
            .retain(|peer_id, _| active_peers.contains(peer_id));

        if inflight_requests > 0 {
            for observation in observations {
                self.observe_peer(observation, now);
            }
            return None;
        }

        for observation in observations {
            let health = self.observe_peer(observation, now);
            if let Some(reason) = direct_path_repair_reason(health, observation, now) {
                return Some((observation.peer_id, reason));
            }
        }

        None
    }

    pub(crate) fn observe_peer(
        &mut self,
        observation: DirectPathObservation,
        now: std::time::Instant,
    ) -> &DirectPathPeerHealth {
        let health = self.peer_health.entry(observation.peer_id).or_default();
        match (observation.snapshot.kind, observation.has_direct_candidate) {
            (heartbeat::SelectedPathKind::Direct, _) | (_, false) => {
                health.non_direct_since = None;
            }
            (heartbeat::SelectedPathKind::Relay | heartbeat::SelectedPathKind::Unknown, true) => {
                if health.non_direct_since.is_none() {
                    health.non_direct_since = Some(now);
                }
            }
        }
        health
    }

    pub(super) fn record_request_attempt(&mut self, peer_id: EndpointId, now: std::time::Instant) {
        self.peer_health.entry(peer_id).or_default().last_request_at = Some(now);
    }

    #[cfg(test)]
    pub(super) fn peer_health(&self, peer_id: EndpointId) -> Option<&DirectPathPeerHealth> {
        self.peer_health.get(&peer_id)
    }
}

pub(super) fn direct_path_repair_reason(
    health: &DirectPathPeerHealth,
    observation: DirectPathObservation,
    now: std::time::Instant,
) -> Option<DirectPathRepairReason> {
    if !observation.has_direct_candidate
        || observation.snapshot.kind == heartbeat::SelectedPathKind::Direct
    {
        return None;
    }
    if health.last_request_at.is_some_and(|last| {
        now.duration_since(last) < std::time::Duration::from_secs(DIRECT_PATH_REPAIR_COOLDOWN_SECS)
    }) {
        return None;
    }
    if !health.non_direct_since.is_some_and(|started| {
        now.duration_since(started) >= std::time::Duration::from_secs(DIRECT_PATH_REPAIR_GRACE_SECS)
    }) {
        return None;
    }
    match observation.snapshot.kind {
        heartbeat::SelectedPathKind::Relay => Some(DirectPathRepairReason::RelaySelected),
        heartbeat::SelectedPathKind::Unknown => Some(DirectPathRepairReason::UnknownSelected),
        heartbeat::SelectedPathKind::Direct => None,
    }
}

pub(crate) fn endpoint_addr_has_direct_candidate(addr: &EndpointAddr) -> bool {
    addr.addrs
        .iter()
        .any(|candidate| matches!(candidate, TransportAddr::Ip(_)))
}

pub(super) fn endpoint_addr_with_previously_advertised_direct_candidates(
    mut requested: EndpointAddr,
    advertised: &EndpointAddr,
) -> Option<EndpointAddr> {
    if requested.id != advertised.id {
        return None;
    }
    requested.addrs.retain(|candidate| {
        matches!(candidate, TransportAddr::Ip(_)) && advertised.addrs.contains(candidate)
    });
    endpoint_addr_has_direct_candidate(&requested).then_some(requested)
}

impl Node {
    /// Start bounded mesh-level direct path maintenance.
    ///
    /// This is not gossip-driven. Each tick selects at most one admitted peer
    /// whose selected path is non-direct while a direct UDP candidate exists,
    /// then sends a targeted request asking that peer to dial our current
    /// advertised endpoint address. The receiver has its own per-peer cooldown.
    pub fn start_direct_path_maintenance(&self) {
        let node = self.clone();
        tokio::spawn(async move {
            let mut controller = DirectPathMaintenanceController::default();

            loop {
                tokio::time::sleep(std::time::Duration::from_secs(
                    DIRECT_PATH_MAINTENANCE_CHECK_SECS,
                ))
                .await;

                let now = std::time::Instant::now();
                let observations = node.direct_path_observations().await;
                let inflight_requests = node.inflight_requests();
                let Some((peer_id, reason)) =
                    controller.plan_request(observations, now, inflight_requests)
                else {
                    continue;
                };

                controller.record_request_attempt(peer_id, now);
                let _ = node.request_direct_path_from_peer(peer_id, reason).await;
            }
        });
    }

    pub(crate) async fn direct_path_observations(&self) -> Vec<DirectPathObservation> {
        let state = self.state.lock().await;
        state
            .peers
            .iter()
            .filter_map(|(peer_id, peer)| {
                if !peer.is_admitted() {
                    return None;
                }
                let conn = state.connections.get(peer_id)?;
                Some(DirectPathObservation {
                    peer_id: *peer_id,
                    snapshot: heartbeat::selected_path_snapshot(conn),
                    has_direct_candidate: endpoint_addr_has_direct_candidate(&peer.addr),
                })
            })
            .collect()
    }

    pub(crate) async fn request_direct_path_from_peer(
        &self,
        peer_id: EndpointId,
        reason: DirectPathRepairReason,
    ) -> bool {
        let Some(conn) = self.direct_path_request_connection(peer_id).await else {
            return false;
        };
        let Some(request) = self.build_direct_path_request(peer_id) else {
            return false;
        };

        tracing::debug!(
            peer = %peer_id.fmt_short(),
            reason = reason.label(),
            "Direct path maintenance requesting reverse dial"
        );
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(DIRECT_PATH_REQUEST_TIMEOUT_SECS),
            send_direct_path_request(conn, request),
        )
        .await;
        log_direct_path_request_result(peer_id, result)
    }

    pub(crate) fn build_direct_path_request(
        &self,
        peer_id: EndpointId,
    ) -> Option<crate::proto::node::DirectPathRequest> {
        let Ok(serialized_addr) = serde_json::to_vec(&self.endpoint_addr_for_advertisement())
        else {
            tracing::debug!(
                peer = %peer_id.fmt_short(),
                "Direct path maintenance could not serialize local endpoint address"
            );
            return None;
        };
        Some(crate::proto::node::DirectPathRequest {
            requester_id: self.endpoint.id().as_bytes().to_vec(),
            r#gen: NODE_PROTOCOL_GENERATION,
            serialized_addr,
        })
    }

    pub(crate) async fn direct_path_request_connection(
        &self,
        peer_id: EndpointId,
    ) -> Option<Connection> {
        let state = self.state.lock().await;
        state.connections.get(&peer_id).cloned()
    }

    pub(super) fn spawn_direct_path_request_stream(
        &self,
        remote: EndpointId,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(error) = node.handle_direct_path_request_stream(remote, recv).await {
                tracing::debug!(
                    "Direct path request from {} failed: {error}",
                    remote.fmt_short()
                );
            }
        });
    }

    pub(crate) async fn handle_direct_path_request_stream(
        &self,
        remote: EndpointId,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        let frame = self.read_direct_path_request(remote, &mut recv).await?;
        let addr: EndpointAddr = serde_json::from_slice(&frame.serialized_addr)
            .context("direct path request endpoint address is invalid")?;
        anyhow::ensure!(
            addr.id == remote,
            "direct path request endpoint id does not match QUIC peer"
        );
        let Some(addr) = self.direct_path_request_addr_for_peer(remote, addr).await else {
            tracing::debug!(
                peer = %remote.fmt_short(),
                "Direct path request ignored because no known direct candidate was supplied"
            );
            return Ok(());
        };
        if !self.record_direct_path_request(remote).await {
            tracing::debug!(
                peer = %remote.fmt_short(),
                "Direct path request ignored due to cooldown"
            );
            return Ok(());
        }
        self.dial_direct_path_request_peer(remote, addr).await;
        Ok(())
    }
}

pub(crate) async fn send_direct_path_request(
    conn: Connection,
    request: crate::proto::node::DirectPathRequest,
) -> Result<()> {
    let (mut send, _) = conn.open_bi().await?;
    send.write_all(&[STREAM_DIRECT_PATH_REQUEST]).await?;
    write_len_prefixed(&mut send, &request.encode_to_vec()).await?;
    let _ = send.finish();
    Ok(())
}

pub(crate) fn log_direct_path_request_result(
    peer_id: EndpointId,
    result: Result<Result<()>, tokio::time::error::Elapsed>,
) -> bool {
    match result {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            tracing::debug!(
                peer = %peer_id.fmt_short(),
                error = %error,
                "Direct path maintenance request failed"
            );
            false
        }
        Err(_) => {
            tracing::debug!(
                peer = %peer_id.fmt_short(),
                "Direct path maintenance request timed out"
            );
            false
        }
    }
}

impl Node {
    pub(crate) async fn read_direct_path_request(
        &self,
        remote: EndpointId,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Result<crate::proto::node::DirectPathRequest> {
        let proto_buf = read_len_prefixed(recv).await?;
        let frame = crate::proto::node::DirectPathRequest::decode(proto_buf.as_slice())
            .context("DirectPathRequest decode error")?;
        frame
            .validate_frame()
            .map_err(|error| anyhow::anyhow!("DirectPathRequest validation error: {error}"))?;
        anyhow::ensure!(
            frame.requester_id.as_slice() == remote.as_bytes(),
            "DirectPathRequest requester_id does not match QUIC peer"
        );
        Ok(frame)
    }

    pub(crate) async fn direct_path_request_addr_for_peer(
        &self,
        remote: EndpointId,
        requested: EndpointAddr,
    ) -> Option<EndpointAddr> {
        let state = self.state.lock().await;
        let peer = state.peers.get(&remote).filter(|peer| peer.admitted)?;
        endpoint_addr_with_previously_advertised_direct_candidates(requested, &peer.addr)
    }

    pub(crate) async fn record_direct_path_request(&self, remote: EndpointId) -> bool {
        let mut state = self.state.lock().await;
        let now = std::time::Instant::now();
        if state
            .direct_path_request_last_at
            .get(&remote)
            .is_some_and(|last| {
                now.duration_since(*last)
                    < std::time::Duration::from_secs(DIRECT_PATH_REQUEST_COOLDOWN_SECS)
            })
        {
            return false;
        }
        state.direct_path_request_last_at.insert(remote, now);
        true
    }

    pub(crate) async fn dial_direct_path_request_peer(
        &self,
        remote: EndpointId,
        addr: EndpointAddr,
    ) {
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(DIRECT_PATH_REQUEST_TIMEOUT_SECS),
            connect_mesh(&self.endpoint, addr),
        )
        .await;
        match result {
            Ok(Ok(conn)) => {
                self.install_direct_path_request_connection(remote, conn)
                    .await;
            }
            Ok(Err(error)) => {
                tracing::debug!(
                    peer = %remote.fmt_short(),
                    error = %error,
                    "Direct path request reverse dial failed"
                );
            }
            Err(_) => {
                tracing::debug!(
                    peer = %remote.fmt_short(),
                    "Direct path request reverse dial timed out"
                );
            }
        }
    }

    pub(crate) async fn install_direct_path_request_connection(
        &self,
        remote: EndpointId,
        conn: Connection,
    ) {
        let (existing, pending) = {
            let state = self.state.lock().await;
            (
                state.connections.get(&remote).cloned(),
                state.pending_connections.contains_key(&remote),
            )
        };
        if pending {
            conn.close(0u32.into(), b"direct-path-pending-handshake");
            return;
        }
        let expected_stable_id = existing.as_ref().map(Connection::stable_id);

        if let Err(error) = self
            .initiate_gossip_inner(conn.clone(), remote, false)
            .await
        {
            conn.close(0u32.into(), b"direct-path-gossip-failed");
            tracing::debug!(
                peer = %remote.fmt_short(),
                error = %error,
                "Direct path request gossip after reverse dial failed"
            );
            return;
        }

        {
            let mut state = self.state.lock().await;
            if state.pending_connections.contains_key(&remote) {
                drop(state);
                conn.close(0u32.into(), b"direct-path-raced-pending");
                return;
            }
            if state.connections.get(&remote).map(Connection::stable_id) != expected_stable_id {
                drop(state);
                conn.close(0u32.into(), b"direct-path-raced");
                return;
            }
            state.connections.insert(remote, conn.clone());
        }
        self.capture_connection_event(ConnectionCaptureEvent {
            event: "peer_connection_opened",
            remote,
            direction: "outbound",
            phase: "direct_path_request",
            protocol: Some(connection_protocol(&conn)),
            path_type: None,
            rtt_ms: None,
            admitted_peer: Some(true),
            reason: Some("reverse_dial"),
        });
        self.capture_selected_connection_path(remote, &conn, "direct_path_request_path");
        let node = self.clone();
        let conn_for_dispatch = conn.clone();
        tokio::spawn(async move {
            node.dispatch_streams(conn_for_dispatch, remote).await;
        });
        if let Some(existing) = existing {
            existing.close(0u32.into(), b"direct-path-replaced");
        }
    }
}
