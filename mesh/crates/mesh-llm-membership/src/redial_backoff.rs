//! Back off before redialing a peer whose connections keep closing (#3081).
//!
//! A closed connection is redialed at once, which is right for an isolated
//! drop. When the peer closes again soon after each redial (a remote that
//! refuses or drops every connection after one gossip exchange, as in #3070),
//! redialing at once becomes a loop: each round is a handshake and a full
//! gossip exchange, many times a second. A redial that follows another one
//! to the same peer within [`STABLE_AFTER`] therefore waits, doubling from
//! [`FIRST_DELAY`] up to [`MAX_DELAY`] with some jitter. A connection that
//! stayed up past [`STABLE_AFTER`] starts the count again.
//!
//! Success here is a connection that lasts, not a completed gossip exchange:
//! in that loop every exchange completes before the close.

use iroh::EndpointId;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A redial at least this long after the previous one counts as a fresh drop.
pub const STABLE_AFTER: Duration = Duration::from_secs(60);
/// Delay before the second redial in a row.
pub const FIRST_DELAY: Duration = Duration::from_secs(1);
/// Longest delay between two redials.
pub const MAX_DELAY: Duration = Duration::from_secs(60);
/// Each delay is spread by up to this share either way.
const JITTER: f64 = 0.2;
/// Peers tracked at once; the oldest entry goes first beyond it.
const MAX_PEERS: usize = 1024;

/// Recent redials per peer. Lives in [`crate::state::MembershipState`].
#[derive(Debug, Default)]
pub struct RedialBackoff {
    peers: HashMap<EndpointId, RedialRecord>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RedialRecord {
    /// When the previous redial to this peer was (or will be) made.
    redialed_at: Instant,
    /// Redials in a row, each within [`STABLE_AFTER`] of the previous one.
    quick_redials: u32,
}

impl RedialBackoff {
    /// How long to wait before redialing `peer`, whose connection just
    /// closed at `now`, and record that redial. Zero for an isolated drop.
    /// `jitter_unit` is uniform in [0, 1).
    pub fn delay_before_redial(
        &mut self,
        peer: EndpointId,
        now: Instant,
        jitter_unit: f64,
    ) -> Duration {
        let quick_redials = quick_redials_after(self.peers.get(&peer).copied(), now);
        let delay = with_jitter(base_delay(quick_redials), jitter_unit);
        if !self.peers.contains_key(&peer)
            && self.peers.len() >= MAX_PEERS
            && let Some(oldest) = self
                .peers
                .iter()
                .min_by_key(|(_, record)| record.redialed_at)
                .map(|(id, _)| *id)
        {
            self.peers.remove(&oldest);
        }
        self.peers.insert(
            peer,
            RedialRecord {
                redialed_at: now + delay,
                quick_redials,
            },
        );
        delay
    }
}

/// Redials in a row once this one is made: one more than before when the
/// previous redial is recent, none otherwise.
fn quick_redials_after(previous: Option<RedialRecord>, now: Instant) -> u32 {
    match previous {
        Some(record) if now.saturating_duration_since(record.redialed_at) < STABLE_AFTER => {
            record.quick_redials.saturating_add(1)
        }
        _ => 0,
    }
}

/// Zero for the first redial, then [`FIRST_DELAY`] doubling up to [`MAX_DELAY`].
fn base_delay(quick_redials: u32) -> Duration {
    if quick_redials == 0 {
        return Duration::ZERO;
    }
    let doublings = (quick_redials - 1).min(16);
    FIRST_DELAY.saturating_mul(1 << doublings).min(MAX_DELAY)
}

/// Spread `base` by up to [`JITTER`] either way; `unit` is uniform in [0, 1).
fn with_jitter(base: Duration, unit: f64) -> Duration {
    if base.is_zero() {
        return base;
    }
    let factor = 1.0 - JITTER + 2.0 * JITTER * unit.clamp(0.0, 1.0);
    base.mul_f64(factor).min(MAX_DELAY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(redialed_at: Instant, quick_redials: u32) -> RedialRecord {
        RedialRecord {
            redialed_at,
            quick_redials,
        }
    }

    fn peer() -> EndpointId {
        iroh::SecretKey::generate().public()
    }

    #[test]
    fn an_isolated_drop_is_redialed_at_once() {
        let now = Instant::now();
        assert_eq!(quick_redials_after(None, now), 0);
        assert_eq!(base_delay(0), Duration::ZERO);
        // A previous redial past the stable window starts the count again.
        let long_ago = now - STABLE_AFTER - Duration::from_secs(1);
        assert_eq!(quick_redials_after(Some(record(long_ago, 7)), now), 0);
    }

    #[test]
    fn a_peer_that_keeps_closing_waits_longer_each_time_up_to_the_cap() {
        let now = Instant::now();
        let just_now = now - Duration::from_millis(50);
        assert_eq!(quick_redials_after(Some(record(just_now, 0)), now), 1);
        assert_eq!(quick_redials_after(Some(record(just_now, 3)), now), 4);
        let delays: Vec<u64> = (1..=8).map(|n| base_delay(n).as_secs()).collect();
        assert_eq!(delays, [1, 2, 4, 8, 16, 32, 60, 60]);
        assert_eq!(base_delay(u32::MAX), MAX_DELAY);
    }

    #[test]
    fn jitter_stays_within_its_share_and_under_the_cap() {
        let base = Duration::from_secs(10);
        assert_eq!(with_jitter(base, 0.0), Duration::from_secs(8));
        assert_eq!(with_jitter(base, 0.5), Duration::from_secs(10));
        assert!(with_jitter(base, 0.999_999) <= Duration::from_secs(12));
        assert_eq!(with_jitter(MAX_DELAY, 0.999), MAX_DELAY);
        assert_eq!(with_jitter(Duration::ZERO, 0.7), Duration::ZERO);
    }

    #[test]
    fn a_redial_loop_backs_off_and_a_stable_connection_resets_it() {
        let mut backoff = RedialBackoff::default();
        let peer = peer();
        let mut now = Instant::now();
        // Twenty closes, each 50 ms after the redial that preceded it, with
        // the smallest jitter so the bound below holds for any draw.
        let delays: Vec<Duration> = (0..20)
            .map(|_| {
                let delay = backoff.delay_before_redial(peer, now, 0.0);
                now += delay + Duration::from_millis(50);
                delay
            })
            .collect();
        assert_eq!(delays[0], Duration::ZERO);
        assert_eq!(delays[1], Duration::from_millis(800));
        assert!(delays.iter().all(|delay| *delay <= MAX_DELAY));
        let total: Duration = delays.iter().sum();
        // Without backoff these twenty redials took one second.
        assert!(total > Duration::from_secs(300), "total {total:?}");
        // A connection that stays up past the window redials at once again.
        now += STABLE_AFTER + MAX_DELAY;
        assert_eq!(backoff.delay_before_redial(peer, now, 0.5), Duration::ZERO);
    }

    #[test]
    fn the_table_stays_bounded() {
        let mut backoff = RedialBackoff::default();
        let now = Instant::now();
        for _ in 0..(MAX_PEERS + 10) {
            backoff.delay_before_redial(peer(), now, 0.5);
        }
        assert_eq!(backoff.peers.len(), MAX_PEERS);
    }
}
