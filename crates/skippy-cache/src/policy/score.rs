//! The benefit-per-exclusive-byte score (#1650 first slice).

use crate::policy::{CostSample, EntryKey, PolicyConfig, PolicyEntry, SharedSegmentLedger};

/// Raw inputs and the resulting score, for tests and opaque logging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreInputs {
    pub reuse_probability: f64,
    pub net_benefit: f64,
    pub exclusive_bytes: f64,
}

/// A computed score with its inputs. Higher is better; ordering by
/// `(value, entry_key)` is the canonical deterministic order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenefitScore {
    pub value: f64,
    pub inputs: ScoreInputs,
}

pub(crate) fn compute(
    config: &PolicyConfig,
    key: EntryKey,
    entry: &PolicyEntry,
    ledger: &SharedSegmentLedger,
) -> Option<BenefitScore> {
    let cost: CostSample = entry.last_cost?;
    if !cost.is_valid() {
        return None;
    }
    let reuse = entry.reuse_probability().max(config.min_reuse_probability);
    let shared = ledger.fractional_bytes(key, &entry.segments);
    let exclusive = entry.exclusive_bytes as f64 + shared;
    if exclusive <= 0.0 {
        return None;
    }
    let net = cost.net_benefit();
    let value = reuse * net / exclusive;
    if !value.is_finite() {
        return None;
    }
    Some(BenefitScore {
        value,
        inputs: ScoreInputs {
            reuse_probability: reuse,
            net_benefit: net,
            exclusive_bytes: exclusive,
        },
    })
}
