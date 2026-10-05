//! Installed model artifacts in the shared Hugging Face cache.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use serde::Serialize;
use skippy_model_artifact::ModelFormat;

use super::local;

#[derive(Clone, Debug, Serialize)]
pub struct InstalledModelArtifact {
    pub model_ref: String,
    pub path: PathBuf,
    pub load_path: PathBuf,
    pub format: ModelFormat,
    pub size_bytes: u64,
}

/// Scan named Hub snapshots and directly cached GGUFs without contacting the Hub.
pub fn scan_installed_artifacts_in(cache_root: &Path) -> Vec<InstalledModelArtifact> {
    let mut paths = local::direct_hf_cache_root_gguf_paths(cache_root);
    let Ok(entries) = std::fs::read_dir(cache_root) else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().starts_with("models--") {
            continue;
        }
        let snapshots = entry.path().join("snapshots");
        collect_model_files(&snapshots, &mut paths);
    }
    paths.sort();
    let mut installed = BTreeMap::new();
    for path in paths {
        let Some(format) = model_format_for_primary(&path) else {
            continue;
        };
        let Some(model_ref) = model_ref_for_path(&path, cache_root, format) else {
            continue;
        };
        let size_bytes = std::fs::metadata(&path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let load_path = if format == ModelFormat::Safetensors {
            path.parent().unwrap_or(&path).to_path_buf()
        } else {
            path.clone()
        };
        installed
            .entry(model_ref.clone())
            .or_insert(InstalledModelArtifact {
                model_ref,
                path,
                load_path,
                format,
                size_bytes,
            });
    }
    installed.into_values().collect()
}

pub fn scan_installed_artifacts() -> Vec<InstalledModelArtifact> {
    scan_installed_artifacts_in(&crate::huggingface_hub_cache_dir())
}

fn collect_model_files(dir: &Path, paths: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            collect_model_files(&path, paths);
        } else if path.is_file() {
            paths.push(path);
        }
    }
}

fn model_format_for_primary(path: &Path) -> Option<ModelFormat> {
    let name = path.file_name()?.to_str()?.to_ascii_lowercase();
    if name.starts_with("mmproj") {
        return None;
    }
    if name.ends_with(".gguf") {
        return skippy_model_ref::split_gguf_shard_info(&name)
            .is_none_or(|shard| shard.part == "00001")
            .then_some(ModelFormat::Gguf);
    }
    if !name.ends_with(".safetensors") || name.starts_with("mtp") {
        return None;
    }
    if let Some((_, part)) = name.rsplit_once("-of-")
        && part.len() == "00000.safetensors".len()
    {
        return name
            .contains("-00001-of-")
            .then_some(ModelFormat::Safetensors);
    }
    Some(ModelFormat::Safetensors)
}

fn model_ref_for_path(path: &Path, cache_root: &Path, format: ModelFormat) -> Option<String> {
    if format == ModelFormat::Gguf {
        return Some(local::model_ref_for_path(path));
    }
    let relative = path.strip_prefix(cache_root).ok()?;
    let mut parts = relative.components();
    let repo_folder = parts
        .next()?
        .as_os_str()
        .to_str()?
        .strip_prefix("models--")?;
    if parts.next()?.as_os_str() != "snapshots" {
        return None;
    }
    let revision = parts.next()?.as_os_str().to_str()?;
    let file = parts.as_path().to_str()?.replace('\\', "/");
    Some(format!(
        "{}@{revision}/{file}",
        repo_folder.replace("--", "/")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_safetensors_and_gguf_without_sidecars_or_extra_shards() {
        let temp = tempfile::tempdir().unwrap();
        let snapshot = temp.path().join("models--org--repo/snapshots/rev");
        std::fs::create_dir_all(&snapshot).unwrap();
        for file in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model-Q4_K_M.gguf",
            "mmproj.gguf",
        ] {
            std::fs::write(snapshot.join(file), b"weight").unwrap();
        }
        let found = scan_installed_artifacts_in(temp.path());
        assert_eq!(found.len(), 2);
        assert!(
            found
                .iter()
                .any(|entry| entry.format == ModelFormat::Safetensors)
        );
        assert!(found.iter().any(|entry| entry.format == ModelFormat::Gguf));
    }
}
