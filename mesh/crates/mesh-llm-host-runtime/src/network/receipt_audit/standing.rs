//! Which peers have shown this node that their audits settle, so that the
//! turns a plugin pays for may be sampled.
//!
//! A plugin that waits for the verdict of a turn, to pay for it, is told
//! `not_drawn` for a turn the VRF did not draw, and may pay for it: the
//! peer could not tell that the turn would not be audited. That is sound
//! only for a peer whose drawn turns do get a verdict. A peer can make a
//! drawn turn end without one: it holds this node's audit workers with
//! slow answers, says that a turn challenged late is gone, or names a
//! commit past this node's budget. Without a rule it would lose the pay of
//! its drawn turns alone and keep that of the others.
//!
//! So sampling is earned. A peer is proven once as many of its drawn turns
//! in a row passed as one draw stands for ([`proving_turns`]); until then
//! every turn of it that a plugin waits on is drawn. A drawn turn that
//! does not pass, whatever the cause, makes the peer unproven again. An
//! honest peer loses nothing by this: its turns are audited more often for
//! a while and pass. A peer that keeps its audits from settling is audited
//! on every turn, and leaves the `receipts` tier after a few
//! (`strikes::STRIKE_LIMIT` unsettled turns in a day, counted beside the
//! strikes).
//!
//! Standing is kept in memory, per peer and model, so passes on a cheap
//! model earn nothing for a costly one. After a restart every peer is
//! unproven. It is bounded, and a peer that is forgotten is unproven.

use std::collections::HashMap;

use iroh::EndpointId;

/// The most peers and models standing is kept for. Past it, the one with
/// the fewest passes is forgotten.
const MAX_PEERS: usize = 4096;
/// The most passes that proving a peer takes, however rare the draw.
const MAX_PROVING: u32 = 16;

#[derive(Default)]
pub(super) struct Standing {
    /// The drawn turns of a peer for a model that passed since one did not.
    passes: HashMap<(EndpointId, String), u32>,
}

/// The passes in a row that prove a peer at the draw rate `rate`: the
/// undrawn turns one draw stands for, nine at 1 in 10, and at most
/// [`MAX_PROVING`]. None when every turn is drawn anyway.
pub(super) fn proving_turns(rate: f64) -> u32 {
    if !(rate > 0.0 && rate < 1.0) {
        return 0;
    }
    let one_in = (1.0 / rate).round().min(f64::from(MAX_PROVING + 1));
    (one_in as u32).saturating_sub(1)
}

impl Standing {
    /// Whether `peer` passed `needed` drawn turns of `model` since one did
    /// not.
    pub(super) fn proven(&self, peer: &EndpointId, model: &str, needed: u32) -> bool {
        let passes = self.passes.get(&(*peer, model.to_owned()));
        passes.copied().unwrap_or(0) >= needed
    }

    /// A drawn turn of `peer` for `model` passed.
    pub(super) fn passed(&mut self, peer: EndpointId, model: &str) {
        let key = (peer, model.to_owned());
        if !self.passes.contains_key(&key) && self.passes.len() >= MAX_PEERS {
            self.forget_fewest();
        }
        let passes = self.passes.entry(key).or_default();
        *passes = passes.saturating_add(1).min(MAX_PROVING);
    }

    /// A drawn turn of `peer` for `model` did not pass: it is unproven
    /// again.
    pub(super) fn lapsed(&mut self, peer: &EndpointId, model: &str) {
        self.passes.remove(&(*peer, model.to_owned()));
    }

    fn forget_fewest(&mut self) {
        let fewest = self
            .passes
            .iter()
            .min_by_key(|(_, passes)| **passes)
            .map(|(key, _)| key.clone());
        if let Some(key) = fewest {
            self.passes.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(n: u64) -> EndpointId {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&(n + 1).to_le_bytes());
        iroh::SecretKey::from_bytes(&seed).public()
    }

    #[test]
    fn proving_takes_the_turns_one_draw_stands_for() {
        assert_eq!(proving_turns(0.1), 9);
        assert_eq!(proving_turns(1.0 / 3.0), 2);
        assert_eq!(proving_turns(1.0 / 10_000.0), MAX_PROVING);
        // every turn is drawn, or none: there is nothing to earn
        assert_eq!(proving_turns(1.0), 0);
        assert_eq!(proving_turns(0.0), 0);
        assert_eq!(proving_turns(f64::NAN), 0);
    }

    #[test]
    fn a_peer_is_proven_by_passes_in_a_row_and_unproven_by_one_miss() {
        let mut standing = Standing::default();
        assert!(!standing.proven(&peer(1), "m", 2), "a new peer is unproven");
        assert!(
            standing.proven(&peer(1), "m", 0),
            "when there is nothing to earn"
        );
        standing.passed(peer(1), "m");
        assert!(!standing.proven(&peer(1), "m", 2));
        standing.passed(peer(1), "m");
        assert!(standing.proven(&peer(1), "m", 2));
        assert!(
            !standing.proven(&peer(2), "m", 2),
            "another peer earns its own"
        );
        assert!(!standing.proven(&peer(1), "other", 1), "and another model");
        standing.lapsed(&peer(1), "m");
        assert!(!standing.proven(&peer(1), "m", 1));
        assert!(standing.passes.is_empty());
    }

    #[test]
    fn what_is_kept_is_bounded_and_a_forgotten_peer_is_unproven() {
        let mut standing = Standing::default();
        for _ in 0..2 * MAX_PROVING {
            standing.passed(peer(0), "m");
        }
        assert_eq!(standing.passes[&(peer(0), "m".to_owned())], MAX_PROVING);
        for n in 1..MAX_PEERS as u64 + 8 {
            standing.passed(peer(n), "m");
        }
        assert_eq!(standing.passes.len(), MAX_PEERS);
        // the peer with the most passes stays; the ones forgotten are unproven
        assert!(standing.proven(&peer(0), "m", MAX_PROVING));
        let kept = (1..MAX_PEERS as u64 + 8)
            .filter(|n| standing.proven(&peer(*n), "m", 1))
            .count();
        assert_eq!(kept, MAX_PEERS - 1);
    }
}
