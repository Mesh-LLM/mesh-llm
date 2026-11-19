//! Strikes against peers: what this node counts when a peer did something
//! only this node saw, or did not answer an audit.
//!
//! One lost answer must never block an honest peer, and a peer that never
//! answers must not go unnoticed. So each such turn is one strike, and
//! [`STRIKE_LIMIT`] strikes within [`STRIKE_WINDOW`] block the peer. A
//! strike ends by age only: a turn that passes its audit clears none, or a
//! peer could answer one audit in two and never be blocked.
//!
//! Strikes are kept in memory: a restart forgets them. Their number is
//! bounded per peer by the limit and over all peers by [`MAX_PEERS`].

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use iroh::EndpointId;

/// Strikes within [`STRIKE_WINDOW`] that block a peer.
pub(crate) const STRIKE_LIMIT: usize = 3;
/// How long a strike counts.
pub(crate) const STRIKE_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);
/// The most peers strikes are kept for. Past it, the peer whose last strike
/// is oldest is forgotten.
const MAX_PEERS: usize = 4096;

/// One strike: when, for which receipt, and why. The texts are already
/// safe to print (`peer_text`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Strike {
    pub(super) at_ms: u64,
    pub(super) receipt_id: String,
    pub(super) reason: String,
}

/// What a strike came to.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Count {
    /// The peer has this many strikes that still count.
    Counted(usize),
    /// The strike was the last one allowed: these block the peer. They are
    /// forgotten, so the count starts again after the block.
    Limit(Vec<Strike>),
}

#[derive(Default)]
pub(super) struct Strikes {
    by_peer: HashMap<EndpointId, VecDeque<Strike>>,
}

impl Strikes {
    /// Counts `strike` against `peer`.
    pub(super) fn record(&mut self, peer: EndpointId, strike: Strike) -> Count {
        let now_ms = strike.at_ms;
        self.expire(now_ms);
        if !self.by_peer.contains_key(&peer) && self.by_peer.len() >= MAX_PEERS {
            self.forget_stalest();
        }
        let strikes = self.by_peer.entry(peer).or_default();
        strikes.push_back(strike);
        if strikes.len() < STRIKE_LIMIT {
            return Count::Counted(strikes.len());
        }
        Count::Limit(self.by_peer.remove(&peer).unwrap_or_default().into())
    }

    /// The strikes against `peer` that count at `now_ms`.
    #[cfg(test)]
    pub(super) fn counting(&mut self, peer: &EndpointId, now_ms: u64) -> usize {
        self.expire(now_ms);
        self.by_peer.get(peer).map_or(0, VecDeque::len)
    }

    fn expire(&mut self, now_ms: u64) {
        let window_ms = u64::try_from(STRIKE_WINDOW.as_millis()).unwrap_or(u64::MAX);
        let counts = |strike: &Strike| now_ms.saturating_sub(strike.at_ms) < window_ms;
        self.by_peer.retain(|_, strikes| {
            strikes.retain(counts);
            !strikes.is_empty()
        });
    }

    fn forget_stalest(&mut self) {
        let last = |strikes: &VecDeque<Strike>| strikes.back().map_or(0, |s| s.at_ms);
        let stalest = self
            .by_peer
            .iter()
            .min_by_key(|(_, strikes)| last(strikes))
            .map(|(peer, _)| *peer);
        if let Some(peer) = stalest {
            self.by_peer.remove(&peer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: u64 = 60 * 60 * 1000;

    fn peer(seed: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[seed; 32]).public()
    }

    fn strike(at_ms: u64) -> Strike {
        Strike {
            at_ms,
            receipt_id: format!("r{at_ms}"),
            reason: "no answer".into(),
        }
    }

    #[test]
    fn the_limit_is_three_strikes_in_a_day() {
        assert_eq!(STRIKE_LIMIT, 3);
        assert_eq!(STRIKE_WINDOW.as_millis(), u128::from(24 * HOUR));
    }

    #[test]
    fn strikes_count_up_to_the_limit_and_then_block() {
        let mut strikes = Strikes::default();
        assert_eq!(strikes.record(peer(1), strike(0)), Count::Counted(1));
        assert_eq!(strikes.record(peer(1), strike(HOUR)), Count::Counted(2));
        // another peer's strikes are its own
        assert_eq!(strikes.record(peer(2), strike(HOUR)), Count::Counted(1));
        let Count::Limit(blocking) = strikes.record(peer(1), strike(2 * HOUR)) else {
            panic!("the third strike does not block");
        };
        assert_eq!(
            blocking,
            vec![strike(0), strike(HOUR), strike(2 * HOUR)],
            "the block lists every strike"
        );
        // the count starts again after the block
        assert_eq!(strikes.counting(&peer(1), 2 * HOUR), 0);
        assert_eq!(strikes.record(peer(1), strike(3 * HOUR)), Count::Counted(1));
        assert_eq!(strikes.counting(&peer(2), 3 * HOUR), 1);
    }

    #[test]
    fn a_strike_ends_by_age() {
        let mut strikes = Strikes::default();
        strikes.record(peer(1), strike(0));
        strikes.record(peer(1), strike(HOUR));
        assert_eq!(strikes.counting(&peer(1), 24 * HOUR - 1), 2);
        assert_eq!(strikes.counting(&peer(1), 24 * HOUR), 1, "the first ended");
        // two old strikes and a new one a day later are not three in a day
        let mut strikes = Strikes::default();
        strikes.record(peer(1), strike(0));
        strikes.record(peer(1), strike(HOUR));
        assert_eq!(
            strikes.record(peer(1), strike(24 * HOUR + 1)),
            Count::Counted(2)
        );
        assert_eq!(
            strikes.record(peer(1), strike(26 * HOUR)),
            Count::Counted(2),
            "the second ended before the fourth came"
        );
        assert!(matches!(
            strikes.record(peer(1), strike(27 * HOUR)),
            Count::Limit(_)
        ));
    }

    #[test]
    fn the_peers_strikes_are_kept_for_are_bounded() {
        let mut strikes = Strikes::default();
        let many = |n: usize| {
            let mut seed = [0u8; 32];
            seed[..8].copy_from_slice(&(n as u64).to_le_bytes());
            iroh::SecretKey::from_bytes(&seed).public()
        };
        for n in 0..MAX_PEERS + 10 {
            strikes.record(many(n), strike(n as u64));
        }
        assert_eq!(strikes.by_peer.len(), MAX_PEERS);
        // the stalest went first, the newest stayed
        let now = (MAX_PEERS + 10) as u64;
        assert_eq!(strikes.counting(&many(0), now), 0);
        assert_eq!(strikes.counting(&many(MAX_PEERS + 9), now), 1);
    }
}
