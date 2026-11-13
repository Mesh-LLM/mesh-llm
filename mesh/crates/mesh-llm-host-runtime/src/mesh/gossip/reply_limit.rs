//! Bounds how often an inbound gossip exchange from one remote endpoint is
//! answered with this node's whole peer table (Mesh-LLM/mesh-llm#3070).
//!
//! A peer that reconnects many times a second would otherwise make this node
//! re-send every announcement and signed record on each connection. Within
//! [`FULL_REPLY_INTERVAL`] of the last full reply to an endpoint, the reply
//! carries only this node's own announcement, which is all a join needs.

use iroh::EndpointId;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Heartbeat gossip to one peer is spaced well beyond this, so a well-behaved
/// peer always gets the full table; a reconnect storm gets it once per interval.
pub(crate) const FULL_REPLY_INTERVAL: Duration = Duration::from_secs(15);

/// Upper bound on remembered endpoints. When full, expired entries go first,
/// then the oldest; an evicted endpoint just gets one extra full reply.
const MAX_TRACKED_ENDPOINTS: usize = 1024;

#[derive(Default)]
pub(crate) struct GossipReplyLimiter {
    last_full_reply: Mutex<HashMap<EndpointId, Instant>>,
}

impl GossipReplyLimiter {
    /// Returns true when `remote` may receive the full table now, and records
    /// that reply; false when it already got one within the interval.
    pub(crate) fn allow_full_reply(&self, remote: EndpointId) -> bool {
        self.allow_full_reply_at(remote, Instant::now())
    }

    fn allow_full_reply_at(&self, remote: EndpointId, now: Instant) -> bool {
        let mut map = self
            .last_full_reply
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(last) = map.get(&remote)
            && now.saturating_duration_since(*last) < FULL_REPLY_INTERVAL
        {
            return false;
        }
        if map.len() >= MAX_TRACKED_ENDPOINTS && !map.contains_key(&remote) {
            map.retain(|_, last| now.saturating_duration_since(*last) < FULL_REPLY_INTERVAL);
            if map.len() >= MAX_TRACKED_ENDPOINTS
                && let Some(oldest) = map.iter().min_by_key(|(_, last)| **last).map(|(id, _)| *id)
            {
                map.remove(&oldest);
            }
        }
        map.insert(remote, now);
        true
    }

    /// Pretends the last full reply to `remote` happened `by` ago.
    #[cfg(test)]
    pub(crate) fn age_for_tests(&self, remote: EndpointId, by: Duration) {
        let mut map = self
            .last_full_reply
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(last) = map.get_mut(&remote) {
            *last -= by;
        }
    }

    #[cfg(test)]
    fn tracked(&self) -> usize {
        self.last_full_reply
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;

    fn id() -> EndpointId {
        SecretKey::generate().public()
    }

    #[test]
    fn repeat_inside_interval_is_reduced_then_full_after() {
        let limiter = GossipReplyLimiter::default();
        let (peer, other) = (id(), id());
        let t0 = Instant::now();
        assert!(limiter.allow_full_reply_at(peer, t0));
        assert!(!limiter.allow_full_reply_at(peer, t0 + Duration::from_millis(50)));
        assert!(
            !limiter.allow_full_reply_at(peer, t0 + FULL_REPLY_INTERVAL - Duration::from_millis(1))
        );
        assert!(limiter.allow_full_reply_at(other, t0 + Duration::from_secs(1)));
        assert!(limiter.allow_full_reply_at(peer, t0 + FULL_REPLY_INTERVAL));
    }

    #[test]
    fn tracked_endpoints_stay_bounded() {
        let limiter = GossipReplyLimiter::default();
        let t0 = Instant::now();
        for i in 0..(MAX_TRACKED_ENDPOINTS * 2) {
            assert!(limiter.allow_full_reply_at(id(), t0 + Duration::from_millis(i as u64)));
        }
        assert_eq!(limiter.tracked(), MAX_TRACKED_ENDPOINTS);
        assert!(limiter.allow_full_reply_at(id(), t0 + FULL_REPLY_INTERVAL * 10));
        assert!(limiter.tracked() <= MAX_TRACKED_ENDPOINTS);
    }
}
