//! Exact local completion correlation; incomplete artifacts remain evidence, not results.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
fn read(path: &Path) -> DynResult<Value> {
    serde_json::from_slice(&super::competitive_cell::read(path, 8 * 1024 * 1024)?)
        .map_err(Into::into)
}
fn hash(path: &Path) -> DynResult<String> {
    crate::product::digest::file_sha256(path).map_err(|error| error.error.into())
}
pub(super) fn completed(
    directory: &Path,
    cell: &Value,
    config: &str,
    provenance: &Value,
) -> DynResult<bool> {
    if !directory.exists() {
        return Ok(false);
    }
    let marker = directory.join("complete.json");
    if !marker.exists() {
        return Ok(false);
    }
    let marker = read(&marker)?;
    if marker["schema_version"] != 2
        || marker["scope"] != "competitive_retained_cell"
        || marker["completed"] != true
        || marker["cell"] != *cell
        || marker["config_sha256"] != config
    {
        return Err("resumed completion identity differs from requested cell".into());
    }
    for (name, file) in [
        ("launch_sha256", "launch.json"),
        ("worker_summary_sha256", "worker/worker-summary.json"),
        ("lifecycle_sha256", "lifecycle.json"),
    ] {
        if marker[name].as_str() != Some(hash(&directory.join(file))?.as_str()) {
            return Err("resumed artifact bytes differ from completion marker".into());
        }
    }
    let launch = read(&directory.join("launch.json"))?;
    let summary = read(&directory.join("worker/worker-summary.json"))?;
    let lifecycle = read(&directory.join("lifecycle.json"))?;
    if launch["cell"] != *cell
        || launch["config_sha256"] != config
        || launch["launch_provenance"] != *provenance
        || summary["cell"] != *cell
        || summary["config_sha256"] != config
        || summary["launch_provenance"] != *provenance
        || summary["completed"] != true
        || summary["passed"] != true
        || lifecycle["infrastructure_clean"] != true
        || lifecycle["worker_status"] != 0
    {
        return Err(
            "resumed launch, worker or cleanup evidence differs from requested identity".into(),
        );
    }
    verify_consumed(directory, cell, &summary)?;
    Ok(true)
}
fn verify_consumed(directory: &Path, cell: &Value, summary: &Value) -> DynResult<()> {
    if cell["workload"] == "synthetic" {
        let result =
            super::competitive_cell::read(&directory.join("worker/result.json"), 8 * 1024 * 1024)?;
        let progress = super::competitive_cell::read(
            &directory.join("worker/progress.jsonl"),
            64 * 1024 * 1024,
        )?;
        if summary["result_sha256"].as_str()
            != Some(hash(&directory.join("worker/result.json"))?.as_str())
            || summary["progress_sha256"].as_str()
                != Some(hash(&directory.join("worker/progress.jsonl"))?.as_str())
        {
            return Err("resumed synthetic consumed artifacts differ".into());
        }
        super::competitive_synthetic::validate(
            &progress,
            &result,
            summary["completed_requests"]
                .as_u64()
                .ok_or("completed requests")?,
            cell["output_tokens"].as_u64().ok_or("output tokens")?,
        )?;
        let parity_path = directory.join("worker/parity.json");
        if summary["parity_sha256"].as_str() != Some(hash(&parity_path)?.as_str())
            || read(&parity_path)?["passed"] != true
        {
            return Err("resumed parity probe bytes or completion differ".into());
        }
    } else {
        let bytes = super::competitive_cell::read(
            &directory.join("worker/requests.jsonl"),
            64 * 1024 * 1024,
        )?;
        if summary["requests_sha256"].as_str()
            != Some(hash(&directory.join("worker/requests.jsonl"))?.as_str())
        {
            return Err("resumed trace request bytes differ".into());
        }
        let mut measured = 0;
        for line in std::str::from_utf8(&bytes)?
            .lines()
            .filter(|line| !line.trim().is_empty())
        {
            let row: Value = serde_json::from_str(line)?;
            if !row["error"].is_null() {
                return Err("resumed trace contains request failure".into());
            }
            if row["phase"] == "measured" {
                measured += 1;
                if row["completion_tokens"] != cell["output_tokens"] {
                    return Err("resumed trace output shorter than requested".into());
                }
            }
        }
        if Some(measured) != cell["prompt_count"].as_u64() {
            return Err("resumed trace missing measured requests".into());
        }
    }
    Ok(())
}
pub(super) fn provenance(
    config: &Value,
    cell: &Value,
    model: &super::competitive_roster::Model,
) -> DynResult<Value> {
    let name = cell["arm"].as_str().ok_or("arm")?;
    let backend = model.backends.get(name).ok_or("backend")?;
    let mut value = json!({"binary_sha256":super::competitive_launch::file(&backend.executable)?,"model_sha256":super::competitive_launch::file(&model.model)?,"backend_version_sha256":backend.version_sha256});
    if ["mesh", "mesh-adaptive"].contains(&name) {
        value["runtime_directory_sha256"] =
            super::competitive_launch::tree(backend.runtime.as_ref().ok_or("mesh runtime")?)?
                .into();
    }
    if let Some(alternate) = &backend.comparison_model {
        value["comparison_input_sha256"] = super::competitive_launch::tree(alternate)?.into();
        let source = config["models"]
            .as_array()
            .ok_or("models")?
            .iter()
            .find(|source| source["key"] == model.key)
            .ok_or("source model")?;
        value["tensor_equivalence_sha256"] =
            source["comparison_inputs"][name]["tensor_equivalence_sha256"].clone();
    }
    Ok(value)
}
pub(super) fn quarantine(root: &Path, directory: &Path) -> DynResult<PathBuf> {
    if !std::fs::symlink_metadata(directory)?.is_dir() {
        return Err("quarantine refuses a linked or special cell".into());
    }
    let relative = directory.strip_prefix(root)?;
    let storage = root.join("quarantine");
    if storage.exists() && !std::fs::symlink_metadata(&storage)?.is_dir() {
        return Err("quarantine must be an owned directory".into());
    }
    std::fs::create_dir_all(&storage)?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    for number in 0..1000 {
        let target = storage.join(format!("{stamp}-{number}"));
        match std::fs::create_dir(&target) {
            Ok(()) => {
                let destination = target.join(relative);
                std::fs::create_dir_all(destination.parent().ok_or("quarantine parent")?)?;
                std::fs::rename(directory, &destination)?;
                return Ok(destination);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(error) => return Err(error.into()),
        }
    }
    Err("quarantine name budget exhausted".into())
}
// Reporting an archived run uses immutable prepared-input declarations, not current
// executable paths and not the launch file being validated. This is correlation,
// not a claim of independent build/tensor authenticity.
pub(super) fn declared_provenance(
    config: &Value,
    cell: &Value,
    model: &super::competitive_roster::Model,
) -> DynResult<Value> {
    let name = cell["arm"].as_str().ok_or("arm")?;
    let backend = model.backends.get(name).ok_or("prepared backend")?;
    let source = config["models"]
        .as_array()
        .ok_or("source models")?
        .iter()
        .find(|source| source["key"] == model.key)
        .ok_or("source model")?;
    if source["sha256"].as_str() != Some(model.model.sha256.as_str()) {
        return Err("prepared report model pin differs from source snapshot".into());
    }
    let mut value = json!({"binary_sha256":backend.executable.sha256,"model_sha256":model.model.sha256,"backend_version_sha256":backend.version_sha256});
    if ["mesh", "mesh-adaptive"].contains(&name) {
        value["runtime_directory_sha256"] = backend
            .runtime
            .as_ref()
            .ok_or("prepared runtime")?
            .sha256
            .clone()
            .into();
    }
    if let Some(alternate) = &backend.comparison_model {
        let pin = &source["comparison_inputs"][name];
        if pin["sha256"].as_str() != Some(alternate.sha256.as_str()) {
            return Err("prepared comparison input pin differs from source snapshot".into());
        }
        value["comparison_input_sha256"] = alternate.sha256.clone().into();
        value["tensor_equivalence_sha256"] = pin["tensor_equivalence_sha256"].clone();
    }
    Ok(value)
}
