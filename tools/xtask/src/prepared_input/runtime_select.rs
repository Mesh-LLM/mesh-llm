//! `sdk-runtime-select RUNTIME_ROOT BACKEND REPORT SKIPPY_ABI`: the consumer
//! of `mesh-llm runtime list --available --json` in
//! `scripts/ci-prepare-native-runtime.sh` (see
//! `scripts/tests/test_ci_sdk_json_consumer.py`). It selects exactly one
//! prepared adjacent runtime, binds it to the expected Skippy ABI, and prints
//! its directory. A malformed or ambiguous report is rejected; nothing is
//! built as a fallback.

use super::{Checked, Rejected, positional, python_json, python_value};
use crate::ci_plan::document::Json;
use std::path::{Path, PathBuf};

const MALFORMED: &str =
    "native runtime compatibility output must be a JSON list or an object with a runtimes list";

pub(super) fn run(args: &[String]) -> Checked<String> {
    let [root, backend, report, abi] = positional(args, "RUNTIME_ROOT BACKEND REPORT SKIPPY_ABI")?;
    let root = absolute(Path::new(root))?;
    let backend = match backend {
        "cuda-blackwell" => "cuda",
        "hip" => "rocm",
        other => other,
    };
    let report = python_json::load(Path::new(report)).map_err(|error| {
        format!("native runtime compatibility output is not valid JSON: {error}")
    })?;
    let runtime_id = selected_id(&report, backend)?;
    let directory = adjacent_directory(&root, &runtime_id, abi)?;
    Ok(format!("{}\n", directory.display()))
}

fn selected_id(report: &Json, backend: &str) -> Checked<String> {
    let rows = match report {
        Json::Object(_) => report.get("runtimes"),
        other => Some(other),
    };
    let Some(rows) = rows.and_then(Json::as_array) else {
        return Err(MALFORMED.into());
    };
    if rows.iter().any(|row| row.as_object().is_none()) {
        return Err("native runtime compatibility rows must be JSON objects".into());
    }
    let supported: Vec<&Json> = rows
        .iter()
        .filter(|row| row.get("supported") == Some(&Json::Bool(true)))
        .collect();
    let preferred: Vec<&Json> = supported
        .iter()
        .copied()
        .filter(|row| row.get("backend").and_then(Json::as_str) == Some(backend))
        .collect();
    let selected = match (preferred.as_slice(), supported.as_slice()) {
        ([row], _) | (_, [row]) => *row,
        _ => {
            let rendered: Vec<String> = supported.iter().map(|row| label(row)).collect();
            let rendered = if rendered.is_empty() {
                "none".to_owned()
            } else {
                rendered.join(", ")
            };
            return Err(Rejected(format!(
                "expected exactly one compatible adjacent native runtime \
                 (preferred backend {backend}); found {rendered}"
            )));
        }
    };
    match selected.get("id").and_then(Json::as_str) {
        Some(id) if !crate::repository::python_text::strip(id).is_empty() => Ok(id.to_owned()),
        _ => Err("compatible native runtime is missing its id".into()),
    }
}

/// `f"{row.get('id', '<missing-id>')}:{row.get('backend', '<missing-backend>')}"`.
fn label(row: &Json) -> String {
    let part = |key: &str, missing: &str| match row.get(key) {
        Some(value) => python_value::display(Some(value)),
        None => missing.to_owned(),
    };
    format!(
        "{}:{}",
        part("id", "<missing-id>"),
        part("backend", "<missing-backend>")
    )
}

/// Manifests at the bundle root and one level below, sorted like `glob`.
fn manifest_paths(root: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if root.join("manifest.json").is_file() {
        paths.push(root.join("manifest.json"));
    }
    let mut nested: Vec<PathBuf> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path().join("manifest.json"))
        .filter(|path| path.exists())
        .collect();
    nested.sort();
    paths.extend(nested);
    paths
}

fn adjacent_directory(root: &Path, runtime_id: &str, abi: &str) -> Checked<PathBuf> {
    let mut matches = Vec::new();
    for manifest_path in manifest_paths(root) {
        let manifest = python_json::load(&manifest_path)
            .map_err(|error| format!("native runtime manifest is not valid JSON: {error}"))?;
        let runtime = manifest.get("runtime").filter(|value| truthy(value));
        let field = |key| runtime.and_then(|runtime| runtime.get(key));
        if field("id").and_then(Json::as_str) != Some(runtime_id) {
            continue;
        }
        if field("skippy_abi").and_then(Json::as_str) != Some(abi) {
            return Err(Rejected(format!(
                "adjacent native runtime {runtime_id} has Skippy ABI {}, expected {abi}",
                python_value::display(Some(field("skippy_abi").unwrap_or(&Json::Null)))
            )));
        }
        let parent = manifest_path.parent().unwrap_or(root);
        matches.push(
            parent
                .canonicalize()
                .unwrap_or_else(|_| parent.to_path_buf()),
        );
    }
    let [selected] = matches.as_slice() else {
        let rendered: Vec<String> = matches
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        let rendered = if rendered.is_empty() {
            "none".to_owned()
        } else {
            rendered.join(", ")
        };
        return Err(Rejected(format!(
            "expected one adjacent artifact directory for runtime {runtime_id}; found {rendered}"
        )));
    };
    if !selected.starts_with(root) {
        return Err(Rejected(format!(
            "selected native runtime escapes adjacent bundle root: {}",
            selected.display()
        )));
    }
    Ok(selected.clone())
}

fn truthy(value: &Json) -> bool {
    !matches!(value, Json::Null | Json::Bool(false))
        && value.as_object().is_none_or(|entries| !entries.is_empty())
}

/// `Path(root).resolve()`: canonical when the root exists.
fn absolute(root: &Path) -> Checked<PathBuf> {
    match root.canonicalize() {
        Ok(path) => Ok(path),
        Err(_) => std::path::absolute(root)
            .map_err(|error| Rejected(super::python_io::os_error(root, &error))),
    }
}
