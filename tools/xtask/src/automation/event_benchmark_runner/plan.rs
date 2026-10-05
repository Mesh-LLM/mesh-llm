//! Versioned deterministic prompts and per-pair launch ordering.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::PathBuf};

pub(super) const ALGORITHM: &str = "sha256-paired-trial-plan-v1";
pub(super) const PRIMARY: &str = "__primary__";
const MAX_PAIRS: usize = 10_000;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Mode {
    Production,
    EventDisabled,
    Off,
}
impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::EventDisabled => "event-disabled",
            Self::Off => "off",
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(super) struct Side {
    pub binary: PathBuf,
    pub mode: Mode,
    pub side_id: String,
}

pub(super) fn sides(
    binary: PathBuf,
    baseline: Option<PathBuf>,
    modes: &[Mode],
) -> DynResult<[Side; 2]> {
    if binary.as_os_str().is_empty()
        || baseline
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
    {
        return Err("benchmark binary paths cannot be empty".into());
    }
    match (baseline,modes) {
        (Some(baseline),[mode])=>Ok([Side{binary,mode:*mode,side_id:"current".into()},Side{binary:baseline,mode:*mode,side_id:"baseline".into()}]),
        (None,[a,b]) if a!=b=>Ok([Side{binary:binary.clone(),mode:*a,side_id:a.label().into()},Side{binary,mode:*b,side_id:b.label().into()}]),
        _=>Err("comparison A requires two distinct modes; comparison B requires one mode and a baseline binary".into())
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Spec {
    pub seed: u64,
    pub pairs_primary: usize,
    pub pairs_scenario: usize,
    pub scenarios: Vec<String>,
}
impl Spec {
    pub fn validate(&self) -> DynResult<()> {
        let total = self
            .scenarios
            .len()
            .checked_mul(self.pairs_scenario)
            .and_then(|n| n.checked_add(self.pairs_primary));
        if self.pairs_primary == 0
            || self.pairs_scenario == 0
            || self.scenarios.is_empty()
            || self.scenarios.len() > 64
            || total.is_none_or(|n| n > MAX_PAIRS)
        {
            return Err("benchmark plans require positive primary/scenario counts, at least one scenario and at most10000 pairs".into());
        }
        let mut names = BTreeSet::new();
        for name in &self.scenarios {
            if name.trim().is_empty() || name.len() > 256 || name == PRIMARY || !names.insert(name)
            {
                return Err(
                    "scenario names must be unique bounded nonempty labels distinct from primary"
                        .into(),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(super) struct Entry {
    pub scenario: String,
    pub pair_index: u64,
    pub prompt_seed: u64,
    pub side_order_first: String,
}
impl Entry {
    pub fn prompt(&self) -> String {
        format!(
            "Respond with a short factual sentence. token={:016x}",
            self.prompt_seed
        )
    }
    pub fn prompt_sha256(&self) -> String {
        hex::encode(Sha256::digest(self.prompt().as_bytes()))
    }
    pub fn log_stem(&self, side_id: &str) -> String {
        let group = hex::encode(Sha256::digest(self.scenario.as_bytes()));
        format!("trial-{}-{}-{side_id}", group, self.pair_index)
    }
}

fn word(seed: u64, scenario: &str, index: usize, domain: &str) -> u64 {
    let mut hash = Sha256::new();
    hash.update(ALGORITHM.as_bytes());
    hash.update(seed.to_be_bytes());
    for value in [scenario.as_bytes(), domain.as_bytes()] {
        hash.update((value.len() as u64).to_be_bytes());
        hash.update(value);
    }
    hash.update((index as u64).to_be_bytes());
    let digest = hash.finalize();
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA256 has eight prefix bytes"),
    )
}

pub(super) fn build(spec: &Spec, sides: &[Side; 2]) -> DynResult<Vec<Entry>> {
    spec.validate()?;
    if sides[0].side_id == sides[1].side_id || sides.iter().any(|side| side.side_id.is_empty()) {
        return Err("paired trial sides require distinct nonempty identities".into());
    }
    let mut entries = Vec::new();
    for (scenario, count) in std::iter::once((PRIMARY, spec.pairs_primary)).chain(
        spec.scenarios
            .iter()
            .map(|name| (name.as_str(), spec.pairs_scenario)),
    ) {
        for pair_index in 0..count {
            let first = (word(spec.seed, scenario, pair_index, "order") & 1) as usize;
            entries.push(Entry {
                scenario: scenario.into(),
                pair_index: pair_index as u64,
                prompt_seed: word(spec.seed, scenario, pair_index, "prompt"),
                side_order_first: sides[first].side_id.clone(),
            });
        }
    }
    Ok(entries)
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
