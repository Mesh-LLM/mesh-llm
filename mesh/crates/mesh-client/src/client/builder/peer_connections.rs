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
pub(crate) struct PeerConnections(Arc<Mutex<PeerConnectionState>>);

#[derive(Debug, Default)]
struct PeerConnectionState {
    generation: Arc<()>,
    peers: BTreeSet<String>,
}

/// A request-scoped handle that accepts observations only for its lifecycle.
#[derive(Clone, Debug)]
pub(crate) struct PeerObservation {
    connections: PeerConnections,
    generation: Arc<()>,
}

impl PeerConnections {
    /// Captures the current lifecycle for one request's dial observations.
    pub(crate) fn observation(&self) -> PeerObservation {
        let generation = self.lock().generation.clone();
        PeerObservation {
            connections: self.clone(),
            generation,
        }
    }

    /// Clears observations and makes every in-flight request from the prior
    /// lifecycle unable to change the count.
    pub(crate) fn advance_generation(&self) {
        let mut state = self.lock();
        state.generation = Arc::new(());
        state.peers.clear();
    }

    /// The number of peers whose most recent observed dial connected.
    pub(crate) fn count(&self) -> usize {
        self.lock().peers.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PeerConnectionState> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl PeerObservation {
    /// Records an established dial only while this request's lifecycle remains
    /// current.
    pub(crate) fn record_established(&self, peer: &str) {
        let mut state = self.connections.lock();
        if Arc::ptr_eq(&state.generation, &self.generation) {
            state.peers.insert(peer.to_string());
        }
    }

    /// Records a failed dial only while this request's lifecycle remains
    /// current.
    pub(crate) fn record_failure(&self, peer: &str) {
        let mut state = self.connections.lock();
        if Arc::ptr_eq(&state.generation, &self.generation) {
            state.peers.remove(peer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PeerConnections;

    #[test]
    fn counts_a_peer_once_across_repeated_connections() {
        let peers = PeerConnections::default();
        let observation = peers.observation();

        observation.record_established("peer-a");
        observation.record_established("peer-a");

        assert_eq!(peers.count(), 1);
    }

    #[test]
    fn drops_a_peer_whose_latest_dial_failed() {
        let peers = PeerConnections::default();
        let observation = peers.observation();
        observation.record_established("peer-a");

        observation.record_failure("peer-a");

        assert_eq!(peers.count(), 0);
    }

    #[test]
    fn a_failed_dial_never_adds_a_peer() {
        let peers = PeerConnections::default();
        let observation = peers.observation();

        observation.record_failure("peer-a");

        assert_eq!(peers.count(), 0);
    }

    #[test]
    fn advancing_generation_clears_every_observation() {
        let peers = PeerConnections::default();
        let observation = peers.observation();
        observation.record_established("peer-a");
        observation.record_established("peer-b");

        peers.advance_generation();

        assert_eq!(peers.count(), 0);
    }

    #[test]
    fn clones_share_one_observation_set() {
        let peers = PeerConnections::default();
        let request_observation = peers.clone().observation();

        request_observation.record_established("peer-a");

        assert_eq!(peers.count(), 1);
    }

    #[test]
    fn stale_established_dial_cannot_restore_a_cleared_peer() {
        let peers = PeerConnections::default();
        let stale_observation = peers.observation();

        peers.advance_generation();
        stale_observation.record_established("peer-a");

        assert_eq!(peers.count(), 0);
    }

    #[test]
    fn stale_failed_dial_cannot_remove_a_current_peer() {
        let peers = PeerConnections::default();
        let stale_observation = peers.observation();

        peers.advance_generation();
        peers.observation().record_established("peer-a");
        stale_observation.record_failure("peer-a");

        assert_eq!(peers.count(), 1);
    }
}
