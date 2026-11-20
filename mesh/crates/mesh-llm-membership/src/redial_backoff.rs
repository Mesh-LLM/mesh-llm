//! Back off before redialing a peer whose connections keep closing (#3081).
//!
//! A closed connection is redialed at once, which is right for an isolated
//! drop. When the peer closes again soon after each redial (a remote that
//! refuses or drops every connection after one gossip exchange, as in #3070),
//! redialing at once becomes a loop: each round is a handshake and a full
//! gossip exchange, many times a second. A connection that closes less than
//! [`STABLE_AFTER`] after it was put in place therefore makes the next redial
//! wait, doubling from [`FIRST_DELAY`] up to [`MAX_DELAY`] with some jitter.
//! A connection that stayed up past [`STABLE_AFTER`] starts the count again.
//!
//! Success here is a connection that lasts, not a completed gossip exchange:
//! in that loop every exchange completes before the close. A connection's
//! lifetime runs from [`RedialBackoff::connected`], called wherever one is put
//! in place, inbound replacements included, to the close that asks for a
//! redial.
//!
//! The first close, or the first after a connection that was not the redial
//! of a recent close, is redialed at once: only closes in a row back off.
//!
//! Each close hands out a [`RedialTicket`]; only the latest ticket for a peer
//! may still dial once its wait is over ([`RedialBackoff::may_redial`]), so a
//! replacement that closes while an older wait is pending does not leave two
//! recoveries dialing the same peer. Which recovery owns a peer's redial is
//! kept apart from the bounded history, so evicting a peer's history while
//! its recovery waits does not cancel that recovery, and generations come
//! from one counter, so a recreated entry never revives an old ticket.
//!
//! Scope: this paces the closed-connection recovery loop only. Request-driven
//! dials (`connection_to_peer`), discovery joins and heartbeat reconnects do
//! not consult it, but they record the connections they put in place.

use iroh::EndpointId;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A connection that lasted at least this long counts as a fresh drop.
pub const STABLE_AFTER: Duration = Duration::from_secs(60);
/// Delay before the second redial in a row.
pub const FIRST_DELAY: Duration = Duration::from_secs(1);
/// Longest delay between two redials.
pub const MAX_DELAY: Duration = Duration::from_secs(60);
/// Each delay is spread by up to this share either way.
const JITTER: f64 = 0.2;
/// Peers tracked at once; the least recently touched entry goes first beyond it.
const MAX_PEERS: usize = 1024;
/// A connection put in place this soon after a close is taken as that close's
/// redial: the longest wait plus the dial and gossip timeouts, with room to
/// spare. One that came later starts a new run of closes.
const REDIAL_WITHIN: Duration = Duration::from_secs(120);

/// Recent connections and redials per peer. Lives in
/// [`crate::state::MembershipState`].
#[derive(Debug, Default)]
pub struct RedialBackoff {
    /// Bounded history, evicted least recently touched first.
    peers: HashMap<EndpointId, RedialRecord>,
    /// The generation of the recovery that owns each peer's next redial,
    /// while one is in progress. Never evicted: an entry lives from a close
    /// until its recovery calls [`RedialBackoff::finish_redial`] or a newer
    /// close replaces it.
    owners: HashMap<EndpointId, u64>,
    /// Source of every ticket's generation, shared by all peers.
    next_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RedialRecord {
    /// When the latest connection to this peer was put in place, if seen.
    connected_at: Option<Instant>,
    /// When the latest redial to this peer was decided.
    closed_at: Option<Instant>,
    /// Quick closes in a row, each less than [`STABLE_AFTER`] after the
    /// connection or redial before it.
    quick_redials: u32,
    /// For eviction: the latest call that touched this entry.
    touched_at: Instant,
}

/// What a close gets back: how long to wait, and the right to dial after it
/// while no newer close for the same peer has replaced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RedialTicket {
    pub delay: Duration,
    generation: u64,
}

impl RedialBackoff {
    /// Record that a connection to `peer` was put in place at `now`.
    pub fn connected(&mut self, peer: EndpointId, now: Instant) {
        let record = self.entry(peer, now);
        record.connected_at = Some(now);
        record.touched_at = now;
    }

    /// How long to wait before redialing `peer`, whose connection just
    /// closed at `now`. Zero for an isolated drop. The returned ticket
    /// replaces any earlier one for this peer. `jitter_unit` is uniform in
    /// [0, 1).
    pub fn delay_before_redial(
        &mut self,
        peer: EndpointId,
        now: Instant,
        jitter_unit: f64,
    ) -> RedialTicket {
        let record = self.entry(peer, now);
        record.quick_redials = quick_redials_after(record, now);
        record.closed_at = Some(now);
        record.touched_at = now;
        let delay = with_jitter(base_delay(record.quick_redials), jitter_unit);
        self.next_generation = self.next_generation.wrapping_add(1);
        let generation = self.next_generation;
        self.owners.insert(peer, generation);
        RedialTicket { delay, generation }
    }

    /// Whether the recovery holding `ticket` may still dial `peer`: no close
    /// for that peer has handed out a newer ticket since.
    pub fn may_redial(&self, peer: EndpointId, ticket: RedialTicket) -> bool {
        self.owners.get(&peer) == Some(&ticket.generation)
    }

    /// The recovery holding `ticket` is done with `peer`, whether it dialed
    /// or stood down. A newer recovery's ownership is left in place.
    pub fn finish_redial(&mut self, peer: EndpointId, ticket: RedialTicket) {
        if self.may_redial(peer, ticket) {
            self.owners.remove(&peer);
        }
    }

    fn entry(&mut self, peer: EndpointId, now: Instant) -> &mut RedialRecord {
        if !self.peers.contains_key(&peer)
            && self.peers.len() >= MAX_PEERS
            && let Some(oldest) = self
                .peers
                .iter()
                .min_by_key(|(_, record)| record.touched_at)
                .map(|(id, _)| *id)
        {
            self.peers.remove(&oldest);
        }
        self.peers.entry(peer).or_insert(RedialRecord {
            connected_at: None,
            closed_at: None,
            quick_redials: 0,
            touched_at: now,
        })
    }
}

/// Quick closes in a row once this close at `now` counts: one more than before
/// when the connection that just closed was short-lived and was the redial of
/// the previous close, none otherwise. The connection's start is the later of
/// its recorded connect and the previous close, so a close with no connect
/// seen since the previous one is measured from that previous close. With no
/// previous close, or one long before the connection started, this is a first
/// drop and is redialed at once.
fn quick_redials_after(record: &RedialRecord, now: Instant) -> u32 {
    let Some(closed) = record.closed_at else {
        return 0;
    };
    let started = record
        .connected_at
        .map_or(closed, |connected| connected.max(closed));
    if started.saturating_duration_since(closed) > REDIAL_WITHIN {
        return 0;
    }
    if now.saturating_duration_since(started) < STABLE_AFTER {
        record.quick_redials.saturating_add(1)
    } else {
        0
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

    fn peer() -> EndpointId {
        iroh::SecretKey::generate().public()
    }

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn an_isolated_drop_is_redialed_at_once() {
        let mut backoff = RedialBackoff::default();
        let peer = peer();
        let start = Instant::now();
        // A connection put in place normally that closes within the first
        // minute is still a first drop.
        backoff.connected(peer, start - secs(10));
        assert_eq!(
            backoff.delay_before_redial(peer, start, 0.5).delay,
            Duration::ZERO
        );
        // A connection that then stays up past the window starts the count again.
        backoff.connected(peer, start + secs(1));
        let later = start + secs(1) + STABLE_AFTER;
        assert_eq!(
            backoff.delay_before_redial(peer, later, 0.5).delay,
            Duration::ZERO
        );
    }

    #[test]
    fn a_peer_that_keeps_closing_waits_longer_each_time_up_to_the_cap() {
        let delays: Vec<u64> = (1..=8).map(|n| base_delay(n).as_secs()).collect();
        assert_eq!(delays, [1, 2, 4, 8, 16, 32, 60, 60]);
        assert_eq!(base_delay(u32::MAX), MAX_DELAY);
    }

    #[test]
    fn jitter_stays_within_its_share_and_under_the_cap() {
        let base = secs(10);
        assert_eq!(with_jitter(base, 0.0), secs(8));
        assert_eq!(with_jitter(base, 0.5), secs(10));
        assert!(with_jitter(base, 0.999_999) <= secs(12));
        assert_eq!(with_jitter(MAX_DELAY, 0.999), MAX_DELAY);
        assert_eq!(with_jitter(Duration::ZERO, 0.7), Duration::ZERO);
    }

    #[test]
    fn a_redial_loop_backs_off_and_a_stable_connection_resets_it() {
        let mut backoff = RedialBackoff::default();
        let peer = peer();
        let mut now = Instant::now();
        // Twenty rounds of: close, wait, redial that connects, close 50 ms
        // later; the smallest jitter so the bound below holds for any draw.
        let delays: Vec<Duration> = (0..20)
            .map(|_| {
                let delay = backoff.delay_before_redial(peer, now, 0.0).delay;
                now += delay;
                backoff.connected(peer, now);
                now += Duration::from_millis(50);
                delay
            })
            .collect();
        assert_eq!(delays[0], Duration::ZERO);
        assert_eq!(delays[1], Duration::from_millis(800));
        assert!(delays.iter().all(|delay| *delay <= MAX_DELAY));
        let total: Duration = delays.iter().sum();
        // Without backoff these twenty redials took one second.
        assert!(total > secs(300), "total {total:?}");
        // A connection that stays up past the window redials at once again.
        backoff.connected(peer, now);
        now += STABLE_AFTER;
        assert_eq!(
            backoff.delay_before_redial(peer, now, 0.5).delay,
            Duration::ZERO
        );
    }

    #[test]
    fn a_replacement_that_closes_during_an_older_wait_leaves_one_dialer() {
        // Saruman's case on #3095: A waits until t=60; an inbound replacement
        // arrives at t=1 and closes at t=2, starting B. Only B may dial.
        let mut backoff = RedialBackoff::default();
        let peer = peer();
        let t0 = Instant::now();
        backoff.delay_before_redial(peer, t0 - secs(1), 0.5);
        let a = backoff.delay_before_redial(peer, t0, 0.5);
        assert!(a.delay > Duration::ZERO);
        backoff.connected(peer, t0 + secs(1));
        let b = backoff.delay_before_redial(peer, t0 + secs(2), 0.5);

        assert!(
            !backoff.may_redial(peer, a),
            "the older recovery must stand down"
        );
        assert!(backoff.may_redial(peer, b));
    }

    #[test]
    fn a_replacement_that_lasted_past_the_window_is_a_fresh_drop() {
        // A wait scheduled at t=0 for t=60 is skipped because an inbound
        // connection arrived at t=1; it closes at t=62 after 61 seconds up.
        let mut backoff = RedialBackoff::default();
        let peer = peer();
        let t0 = Instant::now();
        for n in (1..=8).rev() {
            backoff.delay_before_redial(peer, t0 - Duration::from_millis(50 * n), 0.5);
        }
        assert!(backoff.delay_before_redial(peer, t0, 0.5).delay > secs(30));
        backoff.connected(peer, t0 + secs(1));

        assert_eq!(
            backoff.delay_before_redial(peer, t0 + secs(62), 0.5).delay,
            Duration::ZERO
        );
    }

    #[test]
    fn a_connection_long_after_the_last_close_starts_a_new_run() {
        let mut backoff = RedialBackoff::default();
        let peer = peer();
        let t0 = Instant::now();
        for n in (1..=4).rev() {
            backoff.delay_before_redial(peer, t0 - Duration::from_millis(50 * n), 0.5);
        }
        assert!(backoff.delay_before_redial(peer, t0, 0.5).delay > secs(5));
        // An hour later the peer connects again and drops after ten seconds.
        let later = t0 + secs(3600);
        backoff.connected(peer, later);
        assert_eq!(
            backoff
                .delay_before_redial(peer, later + secs(10), 0.5)
                .delay,
            Duration::ZERO
        );
    }

    #[test]
    fn a_ticket_for_another_peer_may_not_dial() {
        let mut backoff = RedialBackoff::default();
        let known = peer();
        let ticket = backoff.delay_before_redial(known, Instant::now(), 0.5);
        assert!(!backoff.may_redial(peer(), ticket));
    }

    #[test]
    fn evicting_a_waiting_peers_history_keeps_its_recovery() {
        let mut backoff = RedialBackoff::default();
        let waiting = peer();
        let now = Instant::now();
        let ticket = backoff.delay_before_redial(waiting, now, 0.5);
        // Enough other peers come and go to push its history out.
        for _ in 0..MAX_PEERS {
            backoff.connected(peer(), now + secs(1));
        }
        assert!(!backoff.peers.contains_key(&waiting));
        assert!(backoff.may_redial(waiting, ticket));
        backoff.finish_redial(waiting, ticket);
        assert!(!backoff.may_redial(waiting, ticket));
        assert!(backoff.owners.is_empty());
    }

    #[test]
    fn a_recreated_entry_never_revives_an_older_ticket() {
        let mut backoff = RedialBackoff::default();
        let target = peer();
        let now = Instant::now();
        let older = backoff.delay_before_redial(target, now, 0.5);
        for _ in 0..MAX_PEERS {
            backoff.connected(peer(), now + secs(1));
        }
        // The history is gone; a new close recreates it.
        assert!(!backoff.peers.contains_key(&target));
        let newer = backoff.delay_before_redial(target, now + secs(2), 0.5);
        assert_ne!(older, newer);
        assert!(!backoff.may_redial(target, older));
        assert!(backoff.may_redial(target, newer));
        // The older recovery finishing does not take the newer one's place.
        backoff.finish_redial(target, older);
        assert!(backoff.may_redial(target, newer));
    }

    #[test]
    fn the_table_stays_bounded() {
        let mut backoff = RedialBackoff::default();
        let now = Instant::now();
        for _ in 0..(MAX_PEERS + 10) {
            backoff.delay_before_redial(peer(), now, 0.5);
            backoff.connected(peer(), now);
        }
        assert_eq!(backoff.peers.len(), MAX_PEERS);
    }
}
