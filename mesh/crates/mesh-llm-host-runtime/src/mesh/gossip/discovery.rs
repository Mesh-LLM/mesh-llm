use super::{EndpointAddr, EndpointId, Node, PeerAnnouncement};
use crate::mesh::NodeRole;
use mesh_llm_membership::announcements::{
    peer_is_idle_transitive_client, version_allowed_for_rebroadcast,
};

impl Node {
    pub(crate) async fn maybe_connect_discovered_peer(
        &self,
        my_role: &NodeRole,
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
        my_role: &NodeRole,
        peer_id: EndpointId,
        ann: &PeerAnnouncement,
    ) -> bool {
        peer_id == self.endpoint.id()
            || (matches!(my_role, NodeRole::Client) && matches!(ann.role, NodeRole::Client))
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
}
