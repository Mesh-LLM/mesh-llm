use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Profile {
    /// Exact typed existing correctness frontend input, prior to plan overrides.
    pub correctness: Value,
    pub native: Option<Value>,
    pub old: Option<Value>,
    pub new: Option<Value>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    /// Existing cache-family-plan frontend input; its owner validates/catalogs it.
    pub plan: Value,
    pub profiles: BTreeMap<String, Profile>,
    pub execution_seconds: u64,
    pub request_timeout_ms: u64,
    pub cell_seconds: u64,
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || !(30..=86400).contains(&self.execution_seconds)
            || !(15..=3600).contains(&self.cell_seconds)
            || !(1..=600000).contains(&self.request_timeout_ms)
            || self.profiles.len() > 14
            || self.profiles.keys().any(|k| {
                k.is_empty()
                    || k.len() > 128
                    || !k
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
            })
        {
            return Err("invalid bounded cache matrix input".into());
        }
        Ok(())
    }
}
