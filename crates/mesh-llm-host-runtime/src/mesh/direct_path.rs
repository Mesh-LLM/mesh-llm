//! Compatibility handling for legacy direct-path repair requests.
//!
//! Modern nodes let iroh own NAT traversal and no longer emit these requests.
//! The receiver remains so mixed-version meshes can safely accept the additive
//! stream type from older peers.

use super::{ConnectionCaptureEvent, Node, connect_mesh, connection_protocol};
use crate::protocol::{ValidateControlFrame, read_len_prefixed};
use anyhow::{Context, Result};
use iroh::endpoint::Connection;
use iroh::{EndpointAddr, EndpointId, TransportAddr};
use prost::Message;

pub(super) const DIRECT_PATH_REQUEST_COOLDOWN_SECS: u64 = 120;
const DIRECT_PATH_REQUEST_TIMEOUT_SECS: u64 = 10;

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
