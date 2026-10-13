//! Qualified schema1 competitive artifacts, preserving their published cohort identity.
use super::{input::Input, records::Row};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};
pub(super) fn normalize(input: &mut Input<'_>, artifact: &Path) -> DynResult<Vec<Row>> {
    let provenance_files = input.files(&artifact.join("provenance"))?;
    let mut provenance = BTreeMap::new();
    for path in provenance_files {
        if path.extension().is_some_and(|e| e == "json") {
            let platform = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("invalid provenance platform")?
                .to_owned();
            provenance.insert(platform, input.json(&path)?);
        }
    }
    if provenance.is_empty() {
        return Err("benchmark artifact has no provenance JSON".into());
    }
    let gpu_path = artifact.join("runner-gpu.csv");
    if !gpu_path.exists() {
        return Err("missing runner GPU fingerprint".into());
    }
    let (hardware, observed) = super::gpu::fingerprint(&input.read(&gpu_path)?)?;
    let mut rows = Vec::new();
    for path in input.files(&artifact.join("trace"))? {
        if !selected(&path, artifact) {
            continue;
        }
        let marker_path = path.with_file_name("complete.json");
        match std::fs::symlink_metadata(&marker_path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e.into()),
            Ok(_) => {}
        }
        let marker = input.json(&marker_path)?;
        let cell = marker
            .get("cell")
            .filter(|c| c.is_object())
            .ok_or("invalid complete marker cell")?;
        let expected = crate::automation::cohort_identity::digest(cell)?;
        if marker["cell_sha256"].as_str() != Some(expected.as_str()) {
            return Err("invalid complete marker digest".into());
        }
        let result = input.json(&path)?;
        let platform = string(cell, "platform")?;
        let model = string(cell, "model")?;
        let arm = string(cell, "arm")?;
        let provenance = provenance
            .get(platform)
            .ok_or("missing platform provenance")?;
        let cohort = cohort(cell, provenance, &hardware, model, arm)?;
        let successful = count(&result, "successful_requests")?;
        let failed = count(&result, "failed_requests")?;
        let prompts = count(&result, "prompt_count")?;
        let row = Row {
            schema_version: 1,
            created_utc: optional(provenance, "created_utc")?,
            source_sha: optional(provenance, "mesh_head")?,
            cohort_key: crate::automation::cohort_identity::digest(&cohort)?,
            cohort,
            backend_binary_sha256: optional(cell, "binary_sha256")?,
            observed_gpu_state: observed.clone(),
            prompt_count: prompts,
            successful_requests: successful,
            failed_requests: failed,
            output_tokens: count(&result, "output_tokens")?,
            measured_wall_ms: metric(&result, "measured_wall_ms")?,
            output_tokens_per_second: metric(&result, "output_tokens_per_second")?,
            ttft_ms_mean: metric(&result, "ttft_ms_mean")?,
            complete: failed == 0 && successful == prompts,
            artifact_result: path
                .strip_prefix(artifact)?
                .to_string_lossy()
                .replace('\\', "/"),
        };
        row.validate()?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("benchmark artifact has no completed Thoughtworks cells".into());
    }
    Ok(rows)
}
fn selected(path: &Path, artifact: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(artifact.join("trace")) else {
        return false;
    };
    let parts = relative
        .components()
        .map(|part| part.as_os_str())
        .collect::<Vec<_>>();
    parts.len() == 5
        && parts[2].to_str().is_some_and(|s| s.starts_with("c-"))
        && parts[4] == "result.json"
}
fn cohort(
    cell: &Value,
    provenance: &Value,
    hardware: &Value,
    model: &str,
    arm: &str,
) -> DynResult<Value> {
    let backend = if matches!(arm, "mesh" | "mesh-adaptive") {
        Value::Null
    } else {
        cell["binary_sha256"].clone()
    };
    Ok(
        json!({"schema_version":1,"platform":string(cell,"platform")?,"platform_details":provenance["platform_details"],"hardware":hardware,"model":model,"model_sha256":provenance["models"][model],"arm":arm,"control_backend_binary_sha256":backend,"config_sha256":cell["config_sha256"],"prompt_manifest_sha256":cell["manifest_sha256"],"native_runtime_directory_sha256":provenance["native_runtime_directory_sha256"],"comparison_capacity_policy":cell["comparison_capacity_policy"],"concurrency":count(cell,"concurrency")?}),
    )
}
fn string<'a>(value: &'a Value, key: &str) -> DynResult<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("history {key} must be a string").into())
}
fn optional(value: &Value, key: &str) -> DynResult<Option<String>> {
    if value[key].is_null() {
        Ok(None)
    } else {
        Ok(Some(string(value, key)?.to_owned()))
    }
}
fn count(value: &Value, key: &str) -> DynResult<u64> {
    value[key]
        .as_u64()
        .ok_or_else(|| format!("history {key} must be an unsigned integer").into())
}
fn metric(value: &Value, key: &str) -> DynResult<f64> {
    let metric = value[key]
        .as_f64()
        .ok_or_else(|| format!("history {key} must be numeric"))?;
    if !metric.is_finite() || metric < 0.0 {
        return Err(format!("invalid history {key}").into());
    }
    Ok(metric)
}
