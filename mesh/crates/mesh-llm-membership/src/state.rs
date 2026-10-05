//! Live peer, control-connection and admission state, independent of plugins.

use crate::connection_reservation::PendingConnectionHandshake;
use crate::peer_state::PeerInfo;
use crate::requirements::MeshRequirementRejectionEvent;
use iroh::{EndpointId, endpoint::Connection};
use mesh_llm_identity::OwnershipStatus;
use std::collections::{HashMap, HashSet, VecDeque};

pub struct MembershipState {
    pub peers: HashMap<EndpointId, PeerInfo>,
    pub connections: HashMap<EndpointId, Connection>,
    pub pending_connections: HashMap<EndpointId, PendingConnectionHandshake>,
    pub next_pending_connection_attempt: u64,
    /// Remote peers' tunnel maps: peer_endpoint_id → { target_endpoint_id → tunnel_port_on_that_peer }
    pub remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>>,
    /// Peers confirmed dead — don't reconnect from gossip discovery.
    /// Cleared when the peer successfully reconnects via rejoin/join.
    /// Entries expire after [`crate::peer_state::DEAD_PEER_TTL`] so that reconnection attempts
    /// resume. Transitive re-admission of the id stays blocked for
    /// [`crate::peer_state::DEPARTED_PEER_TRANSITIVE_BLOCK_TTL`] via [`MembershipState::departed_peers`]
    /// so stale bridge announcements cannot resurrect it (issue #1756).
    pub dead_peers: HashMap<EndpointId, std::time::Instant>,
    /// Peer ids whose departure was confirmed (heartbeat failure or accepted
    /// PeerDown), with the instant of confirmation. Direct proof of life
    /// clears this wherever [`MembershipState::dead_peers`] is cleared; otherwise
    /// entries expire after [`crate::peer_state::DEPARTED_PEER_TRANSITIVE_BLOCK_TTL`].
    pub departed_peers: HashMap<EndpointId, std::time::Instant>,
    /// Tracks (reporter, target) pairs where a PeerDown claim was rejected
    /// (target was still reachable). Used to suppress repeated false reports
    /// from unreliable reporters (e.g. relay-partitioned nodes).
    pub peer_down_rejections: HashMap<(EndpointId, EndpointId), std::time::Instant>,
    /// Last accepted direct-path dial-back request per peer. This keeps path
    /// maintenance targeted even if a peer repeatedly asks us to reverse-dial.
    pub direct_path_request_last_at: HashMap<EndpointId, std::time::Instant>,
    /// Last policy-rejection status per peer — used to suppress duplicate log lines.
    /// Only logs when the status transitions (first rejection or status change).
    pub policy_rejected_peers: HashMap<EndpointId, OwnershipStatus>,
    /// Peers rejected by immutable mesh requirements. Used to keep pre-admission
    /// streams from disclosing topology after a deterministic requirement reject.
    pub requirement_rejected_peers: HashSet<EndpointId>,
    pub recent_mesh_rejections: VecDeque<MeshRequirementRejectionEvent>,
    /// Test seam for healthy admitted peers when an opaque iroh connection
    /// cannot be fabricated. Production leaves this set empty.
    pub test_peer_liveness: HashSet<EndpointId>,
}

impl Default for MembershipState {
    fn default() -> Self {
        Self {
            peers: HashMap::new(),
            connections: HashMap::new(),
            pending_connections: HashMap::new(),
            next_pending_connection_attempt: 1,
            remote_tunnel_maps: HashMap::new(),
            dead_peers: HashMap::new(),
            departed_peers: HashMap::new(),
            peer_down_rejections: HashMap::new(),
            direct_path_request_last_at: HashMap::new(),
            policy_rejected_peers: HashMap::new(),
            requirement_rejected_peers: HashSet::new(),
            recent_mesh_rejections: VecDeque::new(),
            test_peer_liveness: HashSet::new(),
        }
    }
}

impl MembershipState {
    pub fn peer_has_observed_liveness(&self, peer: &PeerInfo) -> bool {
        crate::peer_state::peer_has_observed_liveness(
            peer,
            self.connections.contains_key(&peer.id) || self.test_peer_liveness.contains(&peer.id),
        )
    }
}
