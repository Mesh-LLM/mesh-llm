use super::Node;

/// Cached mesh membership and control-connection counts.
///
/// This intentionally reads the local peer/connection maps without cloning
/// `PeerInfo` values. It is suitable for lightweight liveness/readiness views.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MeshConnectivitySnapshot {
    pub admitted_peer_count: usize,
    pub connected_peer_count: usize,
}

impl Node {
    /// Return admitted-peer and currently connected-peer counts without
    /// cloning the full peer records.
    pub(crate) async fn connectivity_snapshot(&self) -> MeshConnectivitySnapshot {
        let state = self.state.lock().await;
        let admitted_peer_count = state
            .peers
            .values()
            .filter(|peer| peer.is_admitted())
            .count();
        let connected_peer_count = state
            .peers
            .iter()
            .filter(|(peer_id, peer)| {
                peer.is_admitted() && state.connections.contains_key(*peer_id)
            })
            .count();
        MeshConnectivitySnapshot {
            admitted_peer_count,
            connected_peer_count,
        }
    }
}
