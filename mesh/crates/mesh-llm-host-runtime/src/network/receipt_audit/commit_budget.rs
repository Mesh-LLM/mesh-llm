//! How many commits of a model a peer may make this node fetch.
//!
//! A request that names a model without a commit leaves the commit to the
//! peer's receipt (`requested_model`), and opening a commit this node has
//! not opened reads its GGUF from Hugging Face. A peer that names a fresh
//! commit on every turn would make every drawn turn a cold fetch, past the
//! challenge window, and so never audited, and would fill this node's
//! cache. So this node fetches at most [`COMMITS_PER_DAY`] commits per peer
//! and requested model within [`BUDGET_WINDOW`]. A drawn turn past that is
//! no audit, and the log says why. Whether a commit belongs to the repo's
//! history is checked elsewhere.
//!
//! The budget is kept in memory, for at most [`MAX_KEPT`] peer and model
//! pairs; past it, the stalest goes first.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use iroh::EndpointId;

/// Commits per peer and requested model this node fetches within
/// [`BUDGET_WINDOW`].
pub(super) const COMMITS_PER_DAY: usize = 2;
/// How long a fetched commit counts against its peer and model.
pub(super) const BUDGET_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);
/// The most peer and model pairs kept.
const MAX_KEPT: usize = 4096;

#[derive(Default)]
pub(super) struct CommitBudget {
    /// The commits each peer named for each requested model, with when
    /// each was first admitted, oldest first.
    by_pair: HashMap<(EndpointId, String), VecDeque<(String, u64)>>,
}

impl CommitBudget {
    /// Whether this node may open `commit` of `model` for an audit of
    /// `peer`'s turn at `now_ms`: a commit admitted within the window is
    /// free, a new one takes one of [`COMMITS_PER_DAY`].
    pub(super) fn admit(
        &mut self,
        peer: EndpointId,
        model: &str,
        commit: &str,
        now_ms: u64,
    ) -> bool {
        self.expire(now_ms);
        let key = (peer, model.to_owned());
        if !self.by_pair.contains_key(&key) && self.by_pair.len() >= MAX_KEPT {
            self.forget_stalest();
        }
        let commits = self.by_pair.entry(key).or_default();
        if commits.iter().any(|(named, _)| named == commit) {
            return true;
        }
        if commits.len() >= COMMITS_PER_DAY {
            return false;
        }
        commits.push_back((commit.to_owned(), now_ms));
        true
    }

    fn expire(&mut self, now_ms: u64) {
        let window = u64::try_from(BUDGET_WINDOW.as_millis()).unwrap_or(u64::MAX);
        self.by_pair.retain(|_, commits| {
            commits.retain(|(_, at)| now_ms.saturating_sub(*at) < window);
            !commits.is_empty()
        });
    }

    fn forget_stalest(&mut self) {
        let stalest = self
            .by_pair
            .iter()
            .min_by_key(|(_, commits)| commits.back().map_or(0, |(_, at)| *at))
            .map(|(key, _)| key.clone());
        if let Some(key) = stalest {
            self.by_pair.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR_MS: u64 = 3_600_000;

    fn peer(seed: u8) -> EndpointId {
        EndpointId::from(iroh::SecretKey::from_bytes(&[seed; 32]).public())
    }

    #[test]
    fn a_peer_makes_this_node_fetch_few_commits_a_day() {
        let mut budget = CommitBudget::default();
        let (a, b) = (peer(1), peer(2));
        assert!(budget.admit(a, "m", "c1", 0));
        assert!(budget.admit(a, "m", "c2", HOUR_MS));
        // a commit already fetched is free; a third is refused
        assert!(budget.admit(a, "m", "c1", 2 * HOUR_MS));
        assert!(!budget.admit(a, "m", "c3", 2 * HOUR_MS));
        // other models and other peers have their own budget
        assert!(budget.admit(a, "n", "c3", 2 * HOUR_MS));
        assert!(budget.admit(b, "m", "c3", 2 * HOUR_MS));
        // a day after the first, its place is free again
        assert!(budget.admit(a, "m", "c3", 24 * HOUR_MS));
        assert!(!budget.admit(a, "m", "c4", 24 * HOUR_MS));
    }
}
