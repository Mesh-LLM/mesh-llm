//! Ownership and lifetime of relay-disabled "direct rescue" endpoints.
//!
//! When the relay-health monitor recovers a peer from a relay-only path, the
//! replacement connection is dialled from a fresh, same-identity endpoint that
//! has relays disabled. That endpoint must outlive the connection it owns, so
//! it is parked here and released when the connection is replaced or closed,
//! or when the node shuts down.
//!
//! Keeping the registry in its own module means `node.rs` carries only the
//! `Node` wiring, and the "reject installs once shutdown has begun" rule lives
//! next to the drain that depends on it.

use iroh::{Endpoint, EndpointId};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A relay-disabled endpoint and the connection it owns.
///
/// `connection_stable_id` identifies the exact connection this endpoint was
/// bound for. Release is conditional on it so a late close of a superseded
/// connection cannot tear down the endpoint owning its replacement.
#[derive(Clone)]
pub(crate) struct DirectRescueEndpoint {
    pub(crate) endpoint: Endpoint,
    pub(crate) connection_stable_id: usize,
}

#[derive(Default)]
struct DirectRescueState {
    endpoints: HashMap<EndpointId, DirectRescueEndpoint>,
    /// Set by [`DirectRescueEndpoints::drain_for_shutdown`]. Once set, installs
    /// are refused so a monitor task that is still running cannot park an
    /// endpoint after the drain has already passed it by.
    shutting_down: bool,
}

/// Registry of live rescue endpoints, keyed by the peer whose connection they own.
#[derive(Clone, Default)]
pub(crate) struct DirectRescueEndpoints {
    state: Arc<Mutex<DirectRescueState>>,
}

impl DirectRescueEndpoints {
    /// Park `endpoint` as the owner of `peer_id`'s current connection.
    ///
    /// Returns the endpoint it displaced, if any, for the caller to close
    /// outside any lock it holds. Returns the *incoming* endpoint back to the
    /// caller as `Err` when shutdown has begun, because the drain will not see
    /// it and it must not be leaked.
    pub(crate) async fn install(
        &self,
        peer_id: EndpointId,
        rescue: DirectRescueEndpoint,
    ) -> Result<Option<Endpoint>, Endpoint> {
        let mut state = self.state.lock().await;
        if state.shutting_down {
            return Err(rescue.endpoint);
        }
        Ok(state
            .endpoints
            .insert(peer_id, rescue)
            .map(|replaced| replaced.endpoint))
    }

    /// Drop any endpoint parked for `peer_id`, returning it for the caller to close.
    ///
    /// Used when a replacement connection is installed that does not need a
    /// rescue endpoint of its own.
    pub(crate) async fn take(&self, peer_id: EndpointId) -> Option<Endpoint> {
        let mut state = self.state.lock().await;
        state
            .endpoints
            .remove(&peer_id)
            .map(|removed| removed.endpoint)
    }

    /// Drop the endpoint parked for `peer_id` only if it owns `closing_stable_id`.
    ///
    /// A connection that has already been superseded must not release the
    /// endpoint owning its successor, so the stable id is checked first.
    pub(crate) async fn take_if_owns(
        &self,
        peer_id: EndpointId,
        closing_stable_id: usize,
    ) -> Option<Endpoint> {
        let mut state = self.state.lock().await;
        if state
            .endpoints
            .get(&peer_id)
            .is_some_and(|rescue| rescue.connection_stable_id == closing_stable_id)
        {
            state
                .endpoints
                .remove(&peer_id)
                .map(|removed| removed.endpoint)
        } else {
            None
        }
    }

    /// Mark the registry closed to new installs and hand back every endpoint.
    ///
    /// Idempotent: a second call returns an empty vector. Callers close the
    /// returned endpoints; see [`Self::is_shutting_down`] for the flag this sets.
    pub(crate) async fn drain_for_shutdown(&self) -> Vec<Endpoint> {
        let mut state = self.state.lock().await;
        state.shutting_down = true;
        state
            .endpoints
            .drain()
            .map(|(_, rescue)| rescue.endpoint)
            .collect()
    }

    #[cfg(test)]
    pub(crate) async fn is_empty(&self) -> bool {
        self.state.lock().await.endpoints.is_empty()
    }

    #[cfg(test)]
    pub(crate) async fn stable_id_for(&self, peer_id: EndpointId) -> Option<usize> {
        self.state
            .lock()
            .await
            .endpoints
            .get(&peer_id)
            .map(|rescue| rescue.connection_stable_id)
    }

    #[cfg(test)]
    pub(crate) async fn endpoint_for(&self, peer_id: EndpointId) -> Option<Endpoint> {
        self.state
            .lock()
            .await
            .endpoints
            .get(&peer_id)
            .map(|rescue| rescue.endpoint.clone())
    }

    #[cfg(test)]
    pub(crate) async fn is_shutting_down(&self) -> bool {
        self.state.lock().await.shutting_down
    }
}
