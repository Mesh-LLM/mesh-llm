//! Narrow mutable model-size and append-only source classification policy.
use super::{model_boundaries, policy_document, process, producer_receipt::Context, source};
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
const FAMILY: &str = "ci/llama-canary/family-certified.json";
const PARITY: &str = "docs/skippy/llama-parity-candidates.json";
const RUNNABLE: [&str; 3] = ["candidate", "candidate_stateful", "candidate_multimodal"];
const NONRUNNABLE: [&str; 7] = [
    "implementation_base",
    "needs_boundary_registration",
    "needs_candidate",
    "needs_runtime_slice_support",
    "no_public_gguf_candidate",
    "non_causal_aux",
    "package_or_remote_only",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    context: Context,
    root: PathBuf,
    base: String,
}

pub(super) fn execute(input: &Input) -> DynResult<Value> {
    input.context.validate()?;
    let result = admit(&input.root, &input.base)?;
    input.context.validate()?;
    process::check()?;
    Ok(result)
}

pub(super) fn admit(root: &Path, base: &str) -> DynResult<Value> {
    source::revision(base)?;
    if !root.is_absolute() {
        return Err("manifest policy source must be absolute".into());
    }
    let root = root.canonicalize()?;
    let before = |path: &str| -> DynResult<Value> {
        let bytes = process::git(
            &root,
            &["show".into(), format!("{base}:{path}").into()],
            None,
        )?;
        Ok(serde_json::from_slice(&bytes)?)
    };
    let family_before = before(FAMILY)?;
    let parity_before = before(PARITY)?;
    let family_after: Value = policy_document::json(&root, FAMILY)?;
    let parity_after: Value = policy_document::json(&root, PARITY)?;
    source::prepared(&root)?;
    let (sources, boundaries) = model_boundaries::inventory(&root.join(".deps/llama.cpp"))?;
    family(&family_before, &family_after)?;
    parity(&parity_before, &parity_after, &sources, &boundaries)?;
    process::check()?;
    Ok(json!({"status":"agent_manifest_policy_admitted"}))
}
fn header(value: &Value, except: &str) -> DynResult<Value> {
    let mut value = value.as_object().ok_or("manifest must be object")?.clone();
    value.remove(except);
    Ok(Value::Object(value))
}
fn family(before: &Value, after: &Value) -> DynResult<()> {
    if header(before, "models")? != header(after, "models")? {
        return Err("family certification policy outside model rows changed".into());
    }
    let before = before["models"]
        .as_array()
        .ok_or("family manifest lacks model array")?;
    let after = after["models"]
        .as_array()
        .ok_or("candidate family manifest lacks model array")?;
    if before.len() != after.len() {
        return Err("family certification roster changed".into());
    }
    for (before, after) in before.iter().zip(after) {
        if !before.is_object() || !after.is_object() {
            return Err("family row must be object".into());
        }
        if after["family"] != before["family"] {
            return Err("family certification row identity changed".into());
        }
        let size = before["resources"]["estimated_model_bytes"]
            .as_u64()
            .filter(|value| *value > 0)
            .ok_or("base estimated_model_bytes invalid")?;
        after["resources"]["estimated_model_bytes"]
            .as_u64()
            .filter(|value| *value > 0)
            .ok_or("candidate estimated_model_bytes invalid")?;
        let mut normalized = after.clone();
        normalized["resources"]["estimated_model_bytes"] = json!(size);
        if &normalized != before {
            return Err("family row changed outside estimated_model_bytes".into());
        }
    }
    Ok(())
}
fn name(row: &Value) -> DynResult<&str> {
    row["llama_model"]
        .as_str()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "parity row has no llama_model".into())
}
fn parity(
    before: &Value,
    after: &Value,
    sources: &BTreeSet<String>,
    boundaries: &BTreeSet<String>,
) -> DynResult<()> {
    if header(before, "candidates")? != header(after, "candidates")? {
        return Err("parity policy outside candidate rows changed".into());
    }
    let before = before["candidates"]
        .as_array()
        .ok_or("base parity manifest lacks candidates")?;
    let after = after["candidates"]
        .as_array()
        .ok_or("candidate parity manifest lacks candidates")?;
    if after.get(..before.len()) != Some(before.as_slice()) {
        return Err("existing parity rows changed or reordered".into());
    }
    let existing = before
        .iter()
        .map(|row| name(row).map(str::to_owned))
        .collect::<DynResult<BTreeSet<_>>>()?;
    let expected = sources
        .difference(&existing)
        .cloned()
        .collect::<BTreeSet<_>>();
    let new = &after[before.len()..];
    let actual = new
        .iter()
        .map(|row| name(row).map(str::to_owned))
        .collect::<DynResult<BTreeSet<_>>>()?;
    if actual.len() != new.len() || actual != expected {
        return Err("new parity rows do not uniquely classify exact missing sources".into());
    }
    for row in new {
        classification(row, boundaries)?;
    }
    Ok(())
}
fn classification(row: &Value, boundaries: &BTreeSet<String>) -> DynResult<()> {
    let object = row.as_object().ok_or("new parity row must be object")?;
    if object.keys().any(|key| {
        ![
            "llama_model",
            "family",
            "status",
            "notes",
            "unsupported_reason",
        ]
        .contains(&key.as_str())
    }) {
        return Err("new parity row carries execution or artifact fields".into());
    }
    row["family"]
        .as_str()
        .filter(|family| !family.is_empty())
        .ok_or("new parity row has no family")?;
    let name = name(row)?;
    let status = row["status"]
        .as_str()
        .ok_or("new parity row lacks status")?;
    if !RUNNABLE.contains(&status) && !NONRUNNABLE.contains(&status) {
        return Err("new parity row has disallowed status".into());
    }
    for key in ["notes", "unsupported_reason"] {
        if object.get(key).is_some_and(|value| !value.is_string()) {
            return Err("parity classification text must be string".into());
        }
    }
    if boundaries.contains(name) {
        if !RUNNABLE.contains(&status)
            || row["unsupported_reason"]
                .as_str()
                .is_some_and(|reason| !reason.is_empty())
        {
            return Err(
                "boundary registered source must remain runnable without unsupported reason".into(),
            );
        }
    } else if RUNNABLE.contains(&status) {
        return Err("new runnable parity row lacks paired block registration".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "manifest_policy_tests.rs"]
mod tests;

#[cfg(test)]
use std::fs;
