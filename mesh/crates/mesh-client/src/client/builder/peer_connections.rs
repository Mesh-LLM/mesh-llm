//! Peer connection observations that back `MeshClient::status()`.
//!
//! The client dials per request and closes the connection when that request
//! finishes, so there is no socket table to read at status time. This records
//! the most recent connection outcome per peer instead: a peer is counted while
//! its latest observed dial established a connection, and stops being counted
//! when a later dial to it fails or the client leaves the joined state.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// Peers this client has an observed connection to, shared by every request
/// made from one client so the observations outlive the individual dial.
#[derive(Clone, Debug, Default)]
pub(crate) struct PeerConnections(Arc<Mutex<BTreeSet<String>>>);

impl PeerConnections {
    /// Records that a dial to `peer` established a connection.
    pub(crate) fn record_established(&self, peer: &str) {
        self.lock().insert(peer.to_string());
    }

    /// Records that a dial to `peer` failed, dropping it from the count.
    pub(crate) fn record_failure(&self, peer: &str) {
        self.lock().remove(peer);
    }

    /// The number of peers whose most recent observed dial connected.
    pub(crate) fn count(&self) -> usize {
        self.lock().len()
    }

    /// Drops every observation, used when the client leaves the joined state.
    pub(crate) fn clear(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeSet<String>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::PeerConnections;

    #[test]
    fn counts_a_peer_once_across_repeated_connections() {
        let peers = PeerConnections::default();

        peers.record_established("peer-a");
        peers.record_established("peer-a");

        assert_eq!(peers.count(), 1);
    }

    #[test]
    fn drops_a_peer_whose_latest_dial_failed() {
        let peers = PeerConnections::default();
        peers.record_established("peer-a");

        peers.record_failure("peer-a");

        assert_eq!(peers.count(), 0);
    }

    #[test]
    fn a_failed_dial_never_adds_a_peer() {
        let peers = PeerConnections::default();

        peers.record_failure("peer-a");

        assert_eq!(peers.count(), 0);
    }

    #[test]
    fn clear_drops_every_observation() {
        let peers = PeerConnections::default();
        peers.record_established("peer-a");
        peers.record_established("peer-b");

        peers.clear();

        assert_eq!(peers.count(), 0);
    }

    #[test]
    fn clones_share_one_observation_set() {
        let peers = PeerConnections::default();
        let request_handle = peers.clone();

        request_handle.record_established("peer-a");

        assert_eq!(peers.count(), 1);
    }
}
