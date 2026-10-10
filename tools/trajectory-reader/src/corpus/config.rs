use crate::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Configuration {
    pub schema_version: u32,
    pub seed: u64,
    pub tiers: BTreeMap<String, Tier>,
    pub sources: Vec<Source>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Tier {
    pub max_prompt_chars: Option<usize>,
    pub target_prompt_chars: Option<usize>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Source {
    pub name: String,
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub revision: String,
    pub family: String,
    pub adapter: String,
    pub routing_hint: Option<String>,
    pub quota: BTreeMap<String, usize>,
}
impl Configuration {
    pub fn validate(&self, tier: &str) -> DynResult<&Tier> {
        if self.schema_version != 1 || self.sources.is_empty() {
            return Err("corpus requires schema 1 and sources".into());
        }
        let selected = self.tiers.get(tier).ok_or("unknown corpus tier")?;
        let mut names = BTreeSet::new();
        for source in &self.sources {
            source.validate()?;
            if !names.insert(&source.name) {
                return Err("duplicate corpus source name".into());
            }
            if source.quota.values().any(|n| *n > 100_000) {
                return Err("corpus quota exceeds 100000".into());
            }
        }
        if !self
            .sources
            .iter()
            .any(|s| s.quota.get(tier).copied().unwrap_or_default() > 0)
        {
            return Err("tier has no requested rows".into());
        }
        Ok(selected)
    }
}
impl Source {
    fn validate(&self) -> DynResult<()> {
        for value in [&self.name, &self.config, &self.split, &self.family] {
            if !component(value) {
                return Err("invalid corpus source component".into());
            }
        }
        let parts = self.dataset.split('/').collect::<Vec<_>>();
        if parts.len() != 2 || !parts.iter().all(|part| component(part)) || !commit(&self.revision)
        {
            return Err(
                "dataset requires namespace/repository and immutable 40-hex revision".into(),
            );
        }
        if !super::projections::SUPPORTED.contains(&self.adapter.as_str()) {
            return Err("unknown corpus adapter".into());
        }
        Ok(())
    }
}
pub(super) fn component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
}
pub(super) fn commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
