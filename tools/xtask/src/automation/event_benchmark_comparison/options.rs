//! Complete comparison policy, independent from command-line presentation.
use super::{
    manifest::{Manifest, PRIMARY},
    screening::Policy,
};
use crate::command::DynResult;

pub(super) struct Options {
    pub seed: u64,
    pub bootstrap_samples: usize,
    pub min_primary_pairs: usize,
    pub min_scenario_pairs: usize,
    pub max_degradation_percent: f64,
    pub max_mdd_percent: f64,
    pub report_holm: bool,
}

impl Options {
    pub fn validate(&self, production: &Manifest) -> DynResult<()> {
        let work = production
            .trials
            .len()
            .checked_mul(6)
            .and_then(|n| n.checked_mul(self.bootstrap_samples));
        if !(2..=1_000_000).contains(&self.bootstrap_samples)
            || !(20..=10_000).contains(&self.min_primary_pairs)
            || !(10..=10_000).contains(&self.min_scenario_pairs)
            || work.is_none_or(|n| n > 100_000_000)
            || [self.max_degradation_percent, self.max_mdd_percent]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("comparison policy requires frozen pair minima, finite limits and bounded bootstrap work".into());
        }
        Ok(())
    }

    pub fn policy(&self, group: &str) -> Policy {
        Policy {
            required_pairs: if group == PRIMARY {
                self.min_primary_pairs
            } else {
                self.min_scenario_pairs
            },
            bootstrap_samples: self.bootstrap_samples,
            seed: self.seed,
            degradation_limit: self.max_degradation_percent / 100.0,
            detectable_limit: self.max_mdd_percent / 100.0,
        }
    }
}
