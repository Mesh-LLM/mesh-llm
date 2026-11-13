//! Bounds how often an inbound gossip exchange from one remote endpoint is
//! answered with this node's whole peer table (Mesh-LLM/mesh-llm#3070).
//!
//! A peer that reconnects many times a second would otherwise make this node
//! re-send every announcement and signed record on each connection. Each
//! endpoint gets [`FULL_REPLIES_PER_INTERVAL`] full replies per
//! [`FULL_REPLY_INTERVAL`]; further exchanges in that window carry only this
//! node's own announcement, which is all a join needs.
//!
//! The limit is keyed by endpoint id, so a peer that presents a new key on each
//! connection still gets the full table every time. It trims a reconnect loop's
//! egress; it is not a denial-of-service bound.

use iroh::EndpointId;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Length of the window in which an endpoint gets a limited number of full replies.
pub(crate) const FULL_REPLY_INTERVAL: Duration = Duration::from_secs(15);

/// More than one, so a restart or a retry after a failed write inside the
/// window still receives the table, while a 20/s loop is cut about 100x.
const FULL_REPLIES_PER_INTERVAL: u8 = 3;

/// Upper bound on remembered endpoints. When full, expired entries go first,
/// then the oldest; an evicted endpoint just gets one extra full reply.
const MAX_TRACKED_ENDPOINTS: usize = 1024;

struct Window {
    start: Instant,
    full_replies: u8,
}

#[derive(Default)]
pub(crate) struct GossipReplyLimiter {
    windows: Mutex<HashMap<EndpointId, Window>>,
}

impl GossipReplyLimiter {
    /// Returns true when `remote` may receive the full table now, and counts
    /// that reply; false when it used up its full replies for this window.
    pub(crate) fn allow_full_reply(&self, remote: EndpointId) -> bool {
        self.allow_full_reply_at(remote, Instant::now())
    }

    fn allow_full_reply_at(&self, remote: EndpointId, now: Instant) -> bool {
        let mut map = self
            .windows
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(window) = map.get_mut(&remote)
            && now.saturating_duration_since(window.start) < FULL_REPLY_INTERVAL
        {
            if window.full_replies >= FULL_REPLIES_PER_INTERVAL {
                return false;
            }
            window.full_replies += 1;
            return true;
        }
        if map.len() >= MAX_TRACKED_ENDPOINTS && !map.contains_key(&remote) {
            map.retain(|_, window| {
                now.saturating_duration_since(window.start) < FULL_REPLY_INTERVAL
            });
            if map.len() >= MAX_TRACKED_ENDPOINTS
                && let Some(oldest) = map
                    .iter()
                    .min_by_key(|(_, window)| window.start)
                    .map(|(id, _)| *id)
            {
                map.remove(&oldest);
            }
        }
        map.insert(
            remote,
            Window {
                start: now,
                full_replies: 1,
            },
        );
        true
    }

    /// Pretends the window for `remote` started `by` ago.
    #[cfg(test)]
    pub(crate) fn age_for_tests(&self, remote: EndpointId, by: Duration) {
        let mut map = self
            .windows
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(window) = map.get_mut(&remote) {
            match window.start.checked_sub(by) {
                Some(start) => window.start = start,
                // The clock has not run `by` yet; any such window is expired.
                None => {
                    map.remove(&remote);
                }
            }
        }
    }

    #[cfg(test)]
    fn tracked(&self) -> usize {
        self.windows
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
    fn repeats_past_the_allowance_are_reduced_then_full_after_the_interval() {
        let limiter = GossipReplyLimiter::default();
        let (peer, other) = (id(), id());
        let t0 = Instant::now();
        for i in 0..FULL_REPLIES_PER_INTERVAL {
            assert!(limiter.allow_full_reply_at(peer, t0 + Duration::from_millis(u64::from(i))));
        }
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

    #[test]
    fn rewinding_past_the_clock_start_does_not_panic() {
        let limiter = GossipReplyLimiter::default();
        let peer = id();
        assert!(limiter.allow_full_reply(peer));
        limiter.age_for_tests(peer, Duration::from_secs(u64::MAX / 4));
        assert!(limiter.allow_full_reply(peer));
    }
}
