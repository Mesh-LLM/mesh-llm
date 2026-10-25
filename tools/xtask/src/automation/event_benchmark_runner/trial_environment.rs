//! Snapshot the exact effective child environment without persisting secrets.
#[cfg(test)]
use super::plan::Mode;
use serde::Serialize;
use serde_json::Value;
use std::{collections::BTreeMap, ffi::OsString};

pub(super) const GATE: &str = "MESH_LLM_BENCHMARK_TUNE_TRIAL";
pub(super) const SELECTOR: &str = "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE";
const PARSER: &str = "MESH_LLM_LIFECYCLE_LOG_PARSER";
const REDACTED: &str = "<redacted:present>";
const SENSITIVE: &[&str] = &[
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
    "URL",
    "PATH",
];

#[derive(Debug, Serialize, PartialEq)]
pub(super) struct Entry {
    pub value: Value,
    pub redacted: bool,
}

#[cfg(test)]
pub(super) fn effective(
    mut inherited: BTreeMap<OsString, OsString>,
    mode: Mode,
) -> BTreeMap<OsString, OsString> {
    inherited.insert(GATE.into(), "1".into());
    inherited.insert(SELECTOR.into(), mode.label().into());
    inherited
}

fn sensitive(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    SENSITIVE.iter().any(|part| upper.contains(part))
}

pub(super) fn snapshot(environment: &BTreeMap<OsString, OsString>) -> BTreeMap<String, Entry> {
    environment
        .iter()
        .filter_map(|(name, raw)| {
            let name = name.to_str()?;
            if !name.starts_with("MESH_LLM_") {
                return None;
            }
            let raw = raw.to_str();
            let allowed = [GATE, SELECTOR, PARSER].contains(&name);
            let redacted = !allowed || sensitive(name) || raw.is_none();
            let value = if redacted {
                Value::String(REDACTED.into())
            } else if name == GATE {
                Value::Bool(raw == Some("1"))
            } else {
                Value::String(raw.unwrap_or_default().into())
            };
            Some((name.into(), Entry { value, redacted }))
        })
        .collect()
}

#[cfg(test)]
#[path = "trial_environment_tests.rs"]
mod tests;
