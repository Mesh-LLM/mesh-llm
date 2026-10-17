//! Choosing a referee: who is eligible, in which tier, and which one is asked.
//!
//! **Eligible**, all required: the node serves the twins' model hash and
//! weights digest; is neither twin; announced a key (so it can sign); is not
//! blocked by this node; and is not barred for that model (`bar`).
//!
//! **Tiers.** Tier 1: eligible nodes with a corroboration for that model in
//! this node's own counts (after any lapsed bar, `bar`). Tier 2: eligible
//! nodes with none yet. The pick is random inside tier 1; tier 2 is used only
//! when tier 1 is empty. The tier is sealed in the verdict.
//!
//! These are yes/no rules applied to this node's own verified records.
//! Nothing is ordered by merit, weighted or combined into a number, and
//! nothing is sent to other nodes.

use chrono::{DateTime, Utc};

use super::bar::{barred, corroborated, Recorded};

/// "Not adjudicated" when nobody is eligible.
pub const NO_ELIGIBLE_REFEREE: &str = "no_eligible_referee";

/// A node this node could ask, as it knows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub node_id: String,
    /// The model and weights it serves, as this node last saw it serve them.
    pub model_hash: Option<String>,
    pub weights_digest: Option<String>,
    /// It announced a key, so it can sign a verdict.
    pub announced_key: bool,
    /// This node stopped routing to it.
    pub blocked: bool,
}

/// Who is asked about one pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// 1 or 2; `None` when nobody is eligible.
    pub tier: Option<u64>,
    /// The best non-empty tier, sorted by node id.
    pub pool: Vec<String>,
}

impl Selection {
    /// The node asked for a draw: `pool[draw(pool.len())]`. `None` when the
    /// pool is empty.
    pub fn pick(&self, draw: &mut dyn FnMut(usize) -> usize) -> Option<&str> {
        if self.pool.is_empty() {
            return None;
        }
        self.pool.get(draw(self.pool.len())).map(String::as_str)
    }
}

/// The pair's referee pool at `now`: the twins (`twins`) served `model_hash`
/// on `weights_digest`; `verdicts` are this node's counted verdicts.
pub fn select(
    now: DateTime<Utc>,
    bar_days: u32,
    model_hash: &str,
    weights_digest: &str,
    twins: [&str; 2],
    candidates: &[Candidate],
    verdicts: &[Recorded],
) -> Selection {
    let eligible: Vec<&str> = candidates
        .iter()
        .filter(|c| {
            c.model_hash.as_deref() == Some(model_hash)
                && c.weights_digest.as_deref() == Some(weights_digest)
                && !twins.contains(&c.node_id.as_str())
                && c.announced_key
                && (!c.blocked || cfg!(feature = "mutant-referee-select-ignores-blocked"))
                && !barred(&c.node_id, model_hash, verdicts, now, bar_days)
        })
        .map(|c| c.node_id.as_str())
        .collect();
    let mut tier1: Vec<String> = eligible
        .iter()
        .filter(|n| corroborated(n, model_hash, verdicts, bar_days))
        .map(|n| n.to_string())
        .collect();
    let mut tier2: Vec<String> = eligible
        .iter()
        .filter(|n| !tier1.iter().any(|t| t == *n))
        .map(|n| n.to_string())
        .collect();
    tier1.sort();
    tier2.sort();
    let (tier, pool) = if cfg!(feature = "mutant-referee-select-mixes-tiers") {
        let tier = if tier1.is_empty() { 2 } else { 1 };
        let mut all = [tier1, tier2].concat();
        all.sort();
        (tier, all)
    } else if !tier1.is_empty() {
        (1, tier1)
    } else {
        (2, tier2)
    };
    if pool.is_empty() {
        return Selection { tier: None, pool };
    }
    Selection {
        tier: Some(tier),
        pool,
    }
}

/// A uniform draw in `0..n` from the operating system's randomness (the same
/// source as a record's store nonce).
pub fn random_draw(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n = n as u64;
    // Rejection sampling: no bias toward the low indices.
    let zone = u64::MAX - (u64::MAX % n);
    loop {
        let nonce = crate::producer::capsule::fresh_store_nonce();
        let x = u64::from_str_radix(&nonce[..16], 16).expect("a store nonce is hex");
        if x < zone {
            return (x % n) as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_random_draw_stays_inside_the_pool_and_reaches_all_of_it() {
        let mut seen = [false; 3];
        for _ in 0..300 {
            let d = random_draw(3);
            assert!(d < 3);
            seen[d] = true;
        }
        assert_eq!(seen, [true; 3]);
        assert_eq!(random_draw(1), 0);
        assert_eq!(random_draw(0), 0);
    }
}
