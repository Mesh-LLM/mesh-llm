//! Normalized environment identity with one explicit comparison-A selector.
use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::Value;

use crate::command::DynResult;

pub(super) const PLANNED_SELECTOR: &str = "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE";

#[derive(Clone, Deserialize)]
pub(super) struct Entry {
    pub value: Value,
    pub redacted: bool,
}

pub(super) type Environment = BTreeMap<String, Entry>;

pub(super) fn validate(environment: &Environment) -> DynResult<()> {
    for (name, entry) in environment {
        if name.trim().is_empty()
            || !matches!(
                entry.value,
                Value::String(_) | Value::Bool(_) | Value::Number(_)
            )
        {
            return Err("normalized environment names and scalar values are required".into());
        }
        if entry.redacted && entry.value.as_str() != Some("<redacted:present>") {
            return Err("redacted environment value must contain only its presence marker".into());
        }
    }
    Ok(())
}

pub(super) fn compare(
    a: &Environment,
    b: &Environment,
    planned_selector: bool,
) -> DynResult<Vec<String>> {
    validate(a)?;
    validate(b)?;
    let names = a.keys().chain(b.keys()).collect::<BTreeSet<_>>();
    let mut violations = Vec::new();
    for name in names {
        if planned_selector && name == PLANNED_SELECTOR {
            continue;
        }
        let reason = match a.get(name).zip(b.get(name)) {
            None => Some("present on one side only"),
            Some((a, b)) if a.redacted != b.redacted => Some("redacted-presence mismatch"),
            Some((a, b)) if !a.redacted && a.value != b.value => Some("normalized value mismatch"),
            _ => None,
        };
        if let Some(reason) = reason {
            violations.push(format!("{name}: {reason}"));
        }
    }
    Ok(violations)
}

#[cfg(test)]
#[path = "environment_identity_tests.rs"]
mod tests;
