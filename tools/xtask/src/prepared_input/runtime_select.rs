//! Select one adjacent runtime from the typed `runtime list --available --json` report.
use super::{Checked, Rejected, positional};
use serde::Deserialize;
use std::path::{Path, PathBuf};
mod input;

#[derive(Deserialize)]
struct RuntimeRow {
    id: String,
    backend: String,
    supported: bool,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum RuntimeReport {
    Rows(Vec<RuntimeRow>),
    Catalog { runtimes: Vec<RuntimeRow> },
}
#[derive(Deserialize)]
struct PackageManifest {
    runtime: PackageRuntime,
}
#[derive(Deserialize)]
struct PackageRuntime {
    id: String,
    skippy_abi: String,
}

pub(super) fn run(args: &[String]) -> Checked<String> {
    let [root, backend, report, abi] = positional(args, "RUNTIME_ROOT BACKEND REPORT SKIPPY_ABI")?;
    let root = Path::new(root)
        .canonicalize()
        .map_err(|error| format!("adjacent runtime root is unavailable: {error}"))?;
    let backend = match backend {
        "cuda-blackwell" => "cuda",
        "hip" => "rocm",
        other => other,
    };
    let report: RuntimeReport = read_json(Path::new(report)).map_err(|error| {
        format!(
            "native runtime compatibility output must be a JSON list or an object \
             with a runtimes list containing typed runtime rows: {}",
            error.0
        )
    })?;
    let rows = match report {
        RuntimeReport::Rows(rows) | RuntimeReport::Catalog { runtimes: rows } => rows,
    };
    let id = selected_id(&rows, backend)?;
    let directory = adjacent_directory(&root, id, abi)?;
    Ok(format!("{}\n", directory.display()))
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Checked<T> {
    let bytes = input::read(path)?;
    serde_json::from_slice(&bytes).map_err(|error| Rejected(error.to_string()))
}
fn selected_id<'a>(rows: &'a [RuntimeRow], backend: &str) -> Checked<&'a str> {
    if rows
        .iter()
        .any(|row| row.id.trim().is_empty() || row.backend.trim().is_empty())
    {
        return Err(
            "native runtime compatibility rows require nonempty id and backend strings".into(),
        );
    }
    let supported: Vec<_> = rows.iter().filter(|row| row.supported).collect();
    let preferred: Vec<_> = supported
        .iter()
        .copied()
        .filter(|row| row.backend == backend)
        .collect();
    match (preferred.as_slice(), supported.as_slice()) {
        ([row], _) | (_, [row]) => Ok(&row.id),
        _ => {
            let found = if supported.is_empty() {
                "none".to_owned()
            } else {
                supported
                    .iter()
                    .map(|row| format!("{}:{}", row.id, row.backend))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            Err(Rejected(format!(
                "expected exactly one compatible adjacent native runtime (preferred backend {backend}); found {found}"
            )))
        }
    }
}
fn manifest_paths(root: &Path) -> Checked<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if root.join("manifest.json").is_file() {
        paths.push(root.join("manifest.json"));
    }
    for entry in std::fs::read_dir(root).map_err(|error| error.to_string())? {
        let manifest = entry
            .map_err(|error| error.to_string())?
            .path()
            .join("manifest.json");
        if manifest.exists() {
            paths.push(manifest);
        }
    }
    paths.sort();
    Ok(paths)
}
fn adjacent_directory(root: &Path, runtime_id: &str, abi: &str) -> Checked<PathBuf> {
    let mut matches = Vec::new();
    for path in manifest_paths(root)? {
        let manifest: PackageManifest = read_json(&path)
            .map_err(|error| format!("native runtime manifest is invalid: {}", error.0))?;
        if manifest.runtime.id != runtime_id {
            continue;
        }
        if manifest.runtime.skippy_abi != abi {
            return Err(Rejected(format!(
                "adjacent native runtime {runtime_id} has Skippy ABI {}, expected {abi}",
                manifest.runtime.skippy_abi
            )));
        }
        let directory = path
            .parent()
            .ok_or("runtime manifest has no parent")?
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if !directory.starts_with(root) {
            return Err(Rejected(format!(
                "selected native runtime escapes adjacent bundle root: {}",
                directory.display()
            )));
        }
        matches.push(directory);
    }
    match matches.as_slice() {
        [selected] => Ok(selected.clone()),
        _ => Err(Rejected(format!(
            "expected one adjacent artifact directory for runtime {runtime_id}; found {}",
            matches.len()
        ))),
    }
}
