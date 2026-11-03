//! Admission control for inbound tunneled HTTP request streams.
//!
//! The tunnel manager spawns a handler task for every `STREAM_TUNNEL_HTTP`
//! bi-stream, and each handler holds the QUIC streams and an inflight
//! request slot until the request completes or its header deadline fires.
//! The QUIC transport allows 1024 concurrent bi-directional streams per
//! connection, so an authenticated peer could otherwise keep an unbounded
//! number of handler tasks alive at once (#2206). The HTTP ingress listeners
//! bound their accepted connections with
//! [`crate::network::openai::accept::ConnectionSlots`]; this module does the
//! same for the tunneled ingress: a global quota and a per-peer quota, both
//! claimed before any per-stream work is spawned.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use iroh::EndpointId;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Most tunneled request handlers a node runs at once, across all peers.
pub(crate) const MAX_TUNNELED_HTTP_INFLIGHT: usize = 256;
/// Most tunneled request handlers one authenticated peer may keep alive at
/// once. Inference is slow and the bottleneck, so a busy routing peer needs
/// only a handful of concurrent requests.
pub(crate) const MAX_TUNNELED_HTTP_INFLIGHT_PER_PEER: usize = 32;

/// Whether one inbound tunneled request stream may start its handler.
pub(crate) enum TunnelHttpClaim {
    /// The handler holds the permit until the request finishes.
    Granted(TunnelHttpPermit),
    /// A quota is exhausted; the stream should be refused without spawning
    /// work.
    Full,
}

/// A global and per-peer permit for one tunneled request handler.
///
/// Dropping the permit frees the global slot and the peer's count when the
/// handler finishes.
pub(crate) struct TunnelHttpPermit {
    _global: OwnedSemaphorePermit,
    peers: Arc<Mutex<HashMap<EndpointId, u32>>>,
    peer: EndpointId,
}

impl Drop for TunnelHttpPermit {
    fn drop(&mut self) {
        let mut peers = self.peers.lock().unwrap();
        if let Some(count) = peers.get_mut(&self.peer) {
            *count -= 1;
            if *count == 0 {
                peers.remove(&self.peer);
            }
        }
    }
}

/// The tunneled HTTP ingress quotas.
#[derive(Clone)]
pub(crate) struct TunnelHttpAdmission {
    global: Arc<Semaphore>,
    peers: Arc<Mutex<HashMap<EndpointId, u32>>>,
}

impl TunnelHttpAdmission {
    pub(crate) fn new() -> Self {
        Self {
            global: Arc::new(Semaphore::new(MAX_TUNNELED_HTTP_INFLIGHT)),
            peers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Claims a global and a per-peer permit for one inbound tunneled
    /// request stream, before the handler task is spawned.
    pub(crate) fn claim(&self, peer: EndpointId) -> TunnelHttpClaim {
        let permit = match Arc::clone(&self.global).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => return TunnelHttpClaim::Full,
        };
        let mut peers = self.peers.lock().unwrap();
        let over_peer_quota = peers
            .get(&peer)
            .is_some_and(|count| *count >= MAX_TUNNELED_HTTP_INFLIGHT_PER_PEER as u32);
        if over_peer_quota {
            drop(permit);
            return TunnelHttpClaim::Full;
        }
        *peers.entry(peer).or_insert(0) += 1;
        TunnelHttpClaim::Granted(TunnelHttpPermit {
            _global: permit,
            peers: Arc::clone(&self.peers),
            peer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(index: u8) -> EndpointId {
        EndpointId::from(iroh::SecretKey::from_bytes(&[index; 32]).public())
    }

    #[test]
    fn a_peer_is_refused_once_it_reaches_the_per_peer_quota() {
        let admission = TunnelHttpAdmission::new();
        let mut held = Vec::new();
        for _ in 0..MAX_TUNNELED_HTTP_INFLIGHT_PER_PEER {
            match admission.claim(peer(1)) {
                TunnelHttpClaim::Granted(permit) => held.push(permit),
                TunnelHttpClaim::Full => panic!("under quota the claim must be granted"),
            }
        }
        assert!(
            matches!(admission.claim(peer(1)), TunnelHttpClaim::Full),
            "the peer's (MAX_TUNNELED_HTTP_INFLIGHT_PER_PEER + 1)-th request is refused"
        );
        // Another peer is unaffected while its own quota is free.
        assert!(matches!(
            admission.claim(peer(2)),
            TunnelHttpClaim::Granted(_)
        ));
        drop(held.pop().unwrap());
        assert!(matches!(
            admission.claim(peer(1)),
            TunnelHttpClaim::Granted(_)
        ));
    }

    #[test]
    fn the_global_quota_is_refused_across_many_peers() {
        let admission = TunnelHttpAdmission::new();
        let mut held = Vec::new();
        // One peer per claim: the per-peer quota never trips, so only the
        // global quota can refuse.
        for index in 0..MAX_TUNNELED_HTTP_INFLIGHT {
            match admission.claim(peer((index % 251) as u8)) {
                TunnelHttpClaim::Granted(permit) => held.push(permit),
                TunnelHttpClaim::Full => {
                    panic!("claim {index} must be granted under the global quota")
                }
            }
        }
        assert!(
            matches!(admission.claim(peer(0)), TunnelHttpClaim::Full),
            "the global quota refuses beyond MAX_TUNNELED_HTTP_INFLIGHT"
        );
    }
}
