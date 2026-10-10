//! Full prepared model-source classification, boundary and immutable pin admission.
use super::{
    candidate_view, model_boundaries, policy_document, process, producer_receipt::Context,
    runtime_slice, source,
};
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

#[path = "parity_inventory/boundary_selection.rs"]
mod boundary_selection;

const STATUSES: [&str; 12] = [
    "candidate",
    "candidate_stateful",
    "candidate_multimodal",
    "certified",
    "certified_package_only",
    "implementation_base",
    "needs_boundary_registration",
    "needs_candidate",
    "needs_runtime_slice_support",
    "no_public_gguf_candidate",
    "non_causal_aux",
    "package_or_remote_only",
];
const RUNNABLE: [&str; 3] = ["certified", "candidate", "candidate_stateful"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    context: Context,
    root: PathBuf,
    base: String,
    source_revision: String,
}

pub(super) fn execute(input: &Input) -> DynResult<Value> {
    input.context.validate()?;
    if !input.root.is_absolute() {
        return Err("parity source must be absolute".into());
    }
    let root = input.root.canonicalize()?;
    let base = if input.context.selected_source.is_empty() {
        &input.context.controller_revision
    } else {
        &input.context.selected_source
    };
    if &input.base != base
        || (!input.context.selected_source.is_empty() && &input.source_revision != base)
    {
        return Err("parity base/source differs from frozen controller/selected revision".into());
    }
    candidate_view::policy(&root, &input.base, &input.source_revision)?;
    if process::text(&root, &["rev-parse", "HEAD"])? != input.source_revision {
        return Err("parity source differs from frozen controller/selected revision".into());
    }
    let result = admit(&root, &input.source_revision)?;
    input.context.validate()?;
    process::check()?;
    Ok(result)
}

pub(super) fn admit(root: &Path, source_revision: &str) -> DynResult<Value> {
    let before = source::prepared(root)?;
    let native = root.join(".deps/llama.cpp");
    let (sources, boundaries) = model_boundaries::inventory(&native)?;
    runtime_slice::verify(&native)?;
    let parity = policy_document::json(root, source::parity_manifest_path(root)?)?;
    let family = policy_document::json(root, "ci/llama-canary/family-certified.json")?;
    validate(&parity, &family, &sources, &boundaries)?;
    let after = source::prepared(root)?;
    if before.head != after.head || before.markers != after.markers {
        return Err("prepared native identity changed during parity validation".into());
    }
    if policy_document::json(root, source::parity_manifest_path(root)?)? != parity
        || policy_document::json(root, "ci/llama-canary/family-certified.json")? != family
        || process::text(root, &["rev-parse", "HEAD"])? != source_revision
    {
        return Err("parity source/manifest changed during validation".into());
    }
    process::check()?;
    let classifications = classifications(&parity, &sources, &boundaries)?;
    let next_boundary_target = boundary_selection::next_target(&classifications, &native)?;
    Ok(
        json!({"status":"parity_inventory_admitted","model_sources":sources.len(),"paired_boundaries":boundaries.len(),"classifications":classifications,"next_boundary_target":next_boundary_target}),
    )
}

// Append before validate in the existing parity inventory owner.
/// Emit all admitted native-source classifications, preserving candidate fields.
/// Cache discovery, model-byte inspection and runnable candidate execution are separate.
fn classifications(
    parity: &Value,
    sources: &BTreeSet<String>,
    boundaries: &BTreeSet<String>,
) -> DynResult<Vec<Value>> {
    let rows = parity["candidates"]
        .as_array()
        .ok_or("parity manifest lacks candidates array")?;
    let mut output = Vec::new();
    for name in sources {
        process::check()?;
        for row in rows
            .iter()
            .filter(|row| row["llama_model"].as_str() == Some(name))
        {
            let mut classified = row
                .as_object()
                .ok_or("parity classification must be object")?
                .clone();
            let family = row["family"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| name.replace('-', "_"));
            classified.insert("family".into(), json!(family));
            classified
                .entry("status")
                .or_insert_with(|| json!("candidate"));
            let priority = support_priority(parity, "families", &family)
                .or_else(|| support_priority(parity, "llama_models", name))
                .unwrap_or("p2");
            classified.insert("priority".into(), json!(priority));
            classified.insert(
                "boundary_registered".into(),
                json!(boundaries.contains(name)),
            );
            output.push(Value::Object(classified));
        }
    }
    Ok(output)
}
fn support_priority(parity: &Value, group: &str, name: &str) -> Option<&'static str> {
    ["p0", "p1", "p2"].into_iter().rev().find(|priority| {
        parity["support_priority"][*priority][group]
            .as_array()
            .is_some_and(|names| names.iter().any(|value| value.as_str() == Some(name)))
    })
}

fn validate(
    parity: &Value,
    family: &Value,
    sources: &BTreeSet<String>,
    boundaries: &BTreeSet<String>,
) -> DynResult<()> {
    let rows = parity["candidates"]
        .as_array()
        .ok_or("parity manifest lacks candidates array")?;
    // Every declared pin is an immutable manifest claim, even for an orphan
    // source classification. Do not filter away stale or malformed claims.
    for row in rows {
        process::check()?;
        if let Some(pin) = row.get("model_pin").filter(|pin| !pin.is_null()) {
            validate_pin(pin, family)?;
        }
    }
    for name in sources {
        process::check()?;
        let mut matched = false;
        for row in rows
            .iter()
            .filter(|row| row["llama_model"].as_str() == Some(name))
        {
            matched = true;
            let status = match row.get("status") {
                None => "candidate",
                Some(status) => status.as_str().ok_or("parity status must be string")?,
            };
            if !STATUSES.contains(&status) {
                return Err(format!("unknown parity status for {name}: {status}").into());
            }
            let reason = row
                .get("unsupported_reason")
                .filter(|reason| !reason.is_null())
                .map(|reason| reason.as_str().ok_or("unsupported_reason must be string"))
                .transpose()?
                .unwrap_or("");
            if RUNNABLE.contains(&status) && (!reason.is_empty() || !boundaries.contains(name)) {
                let pending =
                    boundary_selection::pending_reclassification(rows, sources, boundaries);
                return Err(format!("runnable parity family {name} has unsupported reason or lacks paired boundary calls; pending_reclassification: {}", serde_json::to_string(&pending)?).into());
            }
        }
        if !matched {
            return Err(format!("native model source {name} has no parity classification").into());
        }
    }
    Ok(())
}

fn hex(value: &Value, width: usize) -> bool {
    value.as_str().is_some_and(|value| {
        value.len() == width
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}
fn validate_pin(pin: &Value, family: &Value) -> DynResult<()> {
    let repo = pin["repo"]
        .as_str()
        .filter(|repo| repo.contains('/'))
        .ok_or("parity pin repo must be org/name")?;
    let file = pin["file"]
        .as_str()
        .filter(|file| file.ends_with(".gguf"))
        .ok_or("parity pin must name GGUF")?;
    pin["selector"]
        .as_str()
        .filter(|selector| !selector.is_empty())
        .ok_or("parity pin selector missing")?;
    if !hex(&pin["revision"], 40)
        || !hex(&pin["blob_sha256"], 64)
        || pin["size_bytes"].as_u64().is_none_or(|size| size == 0)
    {
        return Err("parity model pin is not immutable revision/size/blob".into());
    }
    let models = family["models"]
        .as_array()
        .ok_or("family manifest lacks models array")?;
    let mut joined = false;
    for model in models {
        let artifact = &model["artifact"];
        let integrity = &artifact["file_integrity"][file];
        if artifact["repo"].as_str() != Some(repo)
            || artifact["revision"] != pin["revision"]
            || integrity.is_null()
        {
            continue;
        }
        if artifact["selector"] != pin["selector"]
            || integrity["size_bytes"] != pin["size_bytes"]
            || integrity["blob_id"] != pin["blob_sha256"]
        {
            return Err("parity model pin disagrees with certified artifact".into());
        }
        joined = true;
    }
    if !joined {
        return Err("parity model pin does not join identical certified artifact".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "parity_inventory_tests.rs"]
mod tests;

#[cfg(test)]
use std::fs;
