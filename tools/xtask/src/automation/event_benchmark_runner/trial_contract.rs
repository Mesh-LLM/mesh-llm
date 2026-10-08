//! Benchmark trial-unit wording shared with the product's measurement contract.
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct Unit {
    pub trial: &'static str,
    pub pair: &'static str,
}

pub(super) fn unit() -> Unit {
    Unit {
        trial: "One trial is one fresh process launch, one readiness wait, one warmup request excluded from metrics, one measured streaming request, and one shutdown.",
        pair: "A pair is two trials, one per side, with the same prompt and seed, side order randomized per pair.",
    }
}

#[cfg(test)]
#[path = "trial_contract_tests.rs"]
mod tests;
