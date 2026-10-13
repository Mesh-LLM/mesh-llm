#[path = "certification.rs"]
mod certification;

use super::ShardError;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct FamilyMap(BTreeMap<String, Vec<String>>);

impl FamilyMap {
    pub(super) fn parse(payload: &Value) -> Result<Self, ShardError> {
        let schema_one = match payload.get("schema_version") {
            Some(Value::Bool(flag)) => *flag,
            Some(Value::Number(number)) => number.as_f64() == Some(1.0),
            _ => false,
        };
        let families = payload
            .get("families")
            .and_then(Value::as_object)
            .filter(|_| schema_one)
            .ok_or(ShardError::FamilyMapSchema)?;
        let mut result = BTreeMap::new();
        for (family, sources) in families {
            let sources = sources
                .as_array()
                .filter(|sources| !sources.is_empty() && family_name(family))
                .ok_or_else(|| ShardError::FamilyMapping(family.clone()))?;
            let mut normalized = BTreeSet::new();
            for source in sources {
                let source = source
                    .as_str()
                    .filter(|source| source_name(source))
                    .ok_or_else(|| ShardError::SourceMapping(family.clone()))?;
                if !normalized.insert(source.to_owned()) {
                    return Err(ShardError::SourceMapping(family.clone()));
                }
            }
            result.insert(family.clone(), normalized.into_iter().collect());
        }
        Ok(Self(result))
    }

    pub(super) fn require_coverage(&self, manifest: &Value) -> Result<(), ShardError> {
        let families = certification::causal_families(manifest)?;
        let missing: Vec<_> = families
            .into_iter()
            .filter(|family| {
                !family
                    .as_str()
                    .is_some_and(|name| self.0.contains_key(name))
            })
            .map(|family| family.label())
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(ShardError::MissingCoverage(missing))
        }
    }

    pub(super) fn owners(&self, source: &str) -> Vec<String> {
        self.0
            .iter()
            .filter(|(_, sources)| sources.iter().any(|mapped| mapped == source))
            .map(|(family, _)| family.clone())
            .collect()
    }
}

fn family_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
}

fn source_name(source: &str) -> bool {
    source
        .strip_prefix("src/models/")
        .and_then(|source| source.strip_suffix(".cpp"))
        .is_some_and(|stem| {
            !stem.is_empty()
                && stem
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        })
}
