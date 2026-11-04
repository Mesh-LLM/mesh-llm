//! Which peers this node can audit, as their receipts show it, and what a
//! peer that stops being auditable loses.
//!
//! A peer chooses what its receipt says: its version, runtime, backend and
//! cache type, and the model it names. Some of these this node cannot
//! audit, and a turn it cannot audit says nothing about the peer. But a
//! peer that knows which of its turns it cheated on could make exactly
//! those unauditable. So, as decided for the `receipts` tier
//! (`mesh/docs/design/RECEIPTS.md`, "Unauditable receipts"):
//!
//! - A peer whose receipt this node cannot audit, for a reason in the
//!   receipt itself, stops satisfying the `receipts` tier on this node for
//!   [`DEMOTION`]. Ordinary routing is unaffected and nothing is blocked.
//!   The demotion lives in memory and is logged once.
//!   A peer that is demoted again within [`REPEAT_WINDOW`] of an earlier
//!   demotion that ended stays demoted for [`REPEAT_DEMOTION`]: without
//!   that, a peer could take one tier turn a day and never be audited.
//! - A peer whose receipt for a model this node audited within
//!   [`DEMOTION`], and that then sends one it cannot audit for the same
//!   model, is struck as well: an honest peer does not change its runtime
//!   between turns.
//!
//! What is kept is bounded by [`MAX_KEPT`] entries of each kind; past it,
//! the stalest go first.

use std::collections::HashMap;
use std::time::Duration;

use iroh::EndpointId;

/// How long a peer whose receipt this node could not audit does not
/// satisfy the `receipts` tier, and how long an audited receipt counts.
pub(crate) const DEMOTION: Duration = Duration::from_secs(24 * 60 * 60);
/// How long a demotion that ended still counts against a new one.
pub(crate) const REPEAT_WINDOW: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// How long a peer that is demoted again within [`REPEAT_WINDOW`] does not
/// satisfy the tier.
pub(crate) const REPEAT_DEMOTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// The most entries of each kind kept.
const MAX_KEPT: usize = 4096;

fn window_ms() -> u64 {
    u64::try_from(DEMOTION.as_millis()).unwrap_or(u64::MAX)
}

/// What an unauditable receipt came to.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Unaudited {
    /// The peer was not demoted before: say so, once.
    pub(super) newly_demoted: bool,
    /// This node audited a receipt of the peer for the model within
    /// [`DEMOTION`]: the peer changed what it runs.
    pub(super) audited_before: bool,
}

#[derive(Default)]
pub(super) struct Auditability {
    /// Peers that do not satisfy the tier, until when.
    demoted: HashMap<EndpointId, u64>,
    /// When a peer was last demoted, kept for [`REPEAT_WINDOW`].
    last_demoted: HashMap<EndpointId, u64>,
    /// When this node last audited a receipt of a peer for a model.
    audited: HashMap<(EndpointId, String), u64>,
}

impl Auditability {
    /// This node audited a receipt `peer` issued for `model`.
    pub(super) fn audited(&mut self, peer: EndpointId, model: &str, now_ms: u64) {
        self.expire(now_ms);
        let key = (peer, model.to_owned());
        if !self.audited.contains_key(&key) && self.audited.len() >= MAX_KEPT {
            forget_stalest(&mut self.audited);
        }
        self.audited.insert(key, now_ms);
    }

    /// `peer` sent a receipt for `model` that this node cannot audit, for
    /// a reason in the receipt.
    pub(super) fn unauditable(&mut self, peer: EndpointId, model: &str, now_ms: u64) -> Unaudited {
        let newly_demoted = self.demote(peer, now_ms);
        let audited_before = self.audited.contains_key(&(peer, model.to_owned()));
        Unaudited {
            newly_demoted,
            audited_before,
        }
    }

    /// Demotes `peer` from the tier from `now_ms`; `true` when it was not
    /// demoted before.
    pub(super) fn demote(&mut self, peer: EndpointId, now_ms: u64) -> bool {
        self.expire(now_ms);
        if !self.demoted.contains_key(&peer) && self.demoted.len() >= MAX_KEPT {
            // the demotion that ends first goes first
            forget_stalest(&mut self.demoted);
        }
        // a demotion that ended and comes again is a pattern, not an accident
        let again = !self.demoted.contains_key(&peer) && self.last_demoted.contains_key(&peer);
        let length = if again { REPEAT_DEMOTION } else { DEMOTION };
        let length_ms = u64::try_from(length.as_millis()).unwrap_or(u64::MAX);
        if !self.last_demoted.contains_key(&peer) && self.last_demoted.len() >= MAX_KEPT {
            forget_stalest(&mut self.last_demoted);
        }
        self.last_demoted.insert(peer, now_ms);
        // a peer that is still demoted keeps the later of its two ends
        let until = now_ms.saturating_add(length_ms);
        let until = self
            .demoted
            .get(&peer)
            .map_or(until, |held| until.max(*held));
        self.demoted.insert(peer, until).is_none()
    }

    /// Whether `peer` does not satisfy the tier at `now_ms`.
    pub(super) fn demoted(&self, peer: &EndpointId, now_ms: u64) -> bool {
        self.demoted.get(peer).is_some_and(|until| now_ms < *until)
    }

    /// Whether this node audited a receipt `peer` issued for `model`
    /// within [`DEMOTION`] of `now_ms`.
    #[cfg(test)]
    pub(super) fn was_audited(&self, peer: EndpointId, model: &str, now_ms: u64) -> bool {
        self.audited
            .get(&(peer, model.to_owned()))
            .is_some_and(|at| now_ms.saturating_sub(*at) < window_ms())
    }

    fn expire(&mut self, now_ms: u64) {
        self.demoted.retain(|_, until| now_ms < *until);
        let repeat = u64::try_from(REPEAT_WINDOW.as_millis()).unwrap_or(u64::MAX);
        self.last_demoted
            .retain(|_, at| now_ms.saturating_sub(*at) < repeat);
        let window = window_ms();
        self.audited
            .retain(|_, at| now_ms.saturating_sub(*at) < window);
    }
}

/// Forgets the entry of `map` with the smallest time.
fn forget_stalest<K: Clone + Eq + std::hash::Hash>(map: &mut HashMap<K, u64>) {
    if let Some(key) = map
        .iter()
        .min_by_key(|(_, at)| **at)
        .map(|(k, _)| k.clone())
    {
        map.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(seed: u8) -> EndpointId {
        EndpointId::from(iroh::SecretKey::from_bytes(&[seed; 32]).public())
    }

    const HOUR_MS: u64 = 3_600_000;

    #[test]
    fn an_unauditable_receipt_demotes_the_peer_for_a_day() {
        let mut kept = Auditability::default();
        let (a, b) = (peer(1), peer(2));
        assert!(!kept.demoted(&a, 0));
        let first = kept.unauditable(a, "m", 0);
        assert_eq!(
            first,
            Unaudited {
                newly_demoted: true,
                audited_before: false
            }
        );
        // said once
        assert!(!kept.unauditable(a, "m", HOUR_MS).newly_demoted);
        assert!(kept.demoted(&a, 23 * HOUR_MS) && !kept.demoted(&b, HOUR_MS));
        // a day after the last one, the peer qualifies again
        assert!(kept.demoted(&a, 24 * HOUR_MS));
        assert!(!kept.demoted(&a, 25 * HOUR_MS));
    }

    #[test]
    fn a_peer_demoted_again_within_a_week_stays_out_for_a_week() {
        let mut kept = Auditability::default();
        let a = peer(1);
        kept.demote(a, 0);
        assert!(!kept.demoted(&a, 25 * HOUR_MS), "the first demotion ends");
        // it took one more turn under the tier and was unauditable again
        kept.demote(a, 26 * HOUR_MS);
        assert!(kept.demoted(&a, (26 + 7 * 24 - 1) * HOUR_MS));
        assert!(!kept.demoted(&a, (26 + 7 * 24 + 1) * HOUR_MS));
        // a peer that stayed auditable for more than a week starts afresh
        let mut kept = Auditability::default();
        kept.demote(a, 0);
        kept.demote(a, 9 * 24 * HOUR_MS);
        assert!(!kept.demoted(&a, (9 * 24 + 25) * HOUR_MS));
    }

    #[test]
    fn a_peer_audited_for_the_model_that_turns_unauditable_changed_its_runtime() {
        let mut kept = Auditability::default();
        let a = peer(1);
        kept.audited(a, "m", 0);
        assert!(!kept.unauditable(a, "other model", HOUR_MS).audited_before);
        assert!(kept.unauditable(a, "m", 2 * HOUR_MS).audited_before);
        // an audit a day old no longer counts
        let mut kept = Auditability::default();
        kept.audited(a, "m", 0);
        assert!(!kept.unauditable(a, "m", 25 * HOUR_MS).audited_before);
    }

    #[test]
    fn what_is_kept_is_bounded() {
        let mut kept = Auditability::default();
        for n in 0..(MAX_KEPT as u64 + 10) {
            let seed = [(n % 251) as u8, (n / 251) as u8, 7];
            let mut bytes = [0u8; 32];
            bytes[..3].copy_from_slice(&seed);
            let p = EndpointId::from(iroh::SecretKey::from_bytes(&bytes).public());
            kept.audited(p, "m", n);
            kept.demote(p, n);
        }
        assert!(kept.audited.len() <= MAX_KEPT && kept.demoted.len() <= MAX_KEPT);
    }
}
