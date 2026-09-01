use std::{
    fs,
    path::{Path, PathBuf},
};

use skippy_protocol::{LoadMode, StageConfig};
use skippy_runtime::ModelInfo;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ModelKvCapability {
    KnownDense,
    KnownRecurrent,
    Unknown(String),
}

pub(super) fn inspect_model_kv_capability(
    config: &StageConfig,
    declared: Option<ModelKvCapability>,
) -> ModelKvCapability {
    let paths = kv_cache_inspection_paths(config);
    if paths.is_empty() {
        return declared.unwrap_or_else(|| {
            ModelKvCapability::Unknown(
                "model family is unknown and no model metadata is available".to_string(),
            )
        });
    }

    let mut recurrent = false;
    for path in paths {
        let info = match ModelInfo::open(&path) {
            Ok(info) => info,
            Err(error) => {
                return ModelKvCapability::Unknown(format!(
                    "failed to open KV capability metadata {}: {error}",
                    path.display()
                ));
            }
        };
        let tensors = match info.tensors() {
            Ok(tensors) => tensors,
            Err(error) => {
                return ModelKvCapability::Unknown(format!(
                    "failed to read KV capability tensors {}: {error}",
                    path.display()
                ));
            }
        };
        recurrent |= tensors
            .iter()
            .any(|tensor| tensor_name_requires_recurrent_state(&tensor.name));
    }

    let inspected = if recurrent {
        ModelKvCapability::KnownRecurrent
    } else {
        ModelKvCapability::KnownDense
    };
    reconcile_model_kv_capability(declared, inspected)
}

fn reconcile_model_kv_capability(
    declared: Option<ModelKvCapability>,
    inspected: ModelKvCapability,
) -> ModelKvCapability {
    match (declared, inspected) {
        (Some(ModelKvCapability::KnownDense), ModelKvCapability::KnownDense) => {
            ModelKvCapability::KnownDense
        }
        (Some(ModelKvCapability::KnownRecurrent), ModelKvCapability::KnownRecurrent)
        | (None, ModelKvCapability::KnownRecurrent) => ModelKvCapability::KnownRecurrent,
        (Some(declared), inspected) => ModelKvCapability::Unknown(format!(
            "declared model capability {declared:?} disagrees with inspected capability {inspected:?}"
        )),
        (None, ModelKvCapability::KnownDense) => ModelKvCapability::Unknown(
            "model family is unknown and dense tensor names alone are not conclusive".to_string(),
        ),
        (None, unknown @ ModelKvCapability::Unknown(_)) => unknown,
    }
}

pub(super) fn kv_cache_inspection_paths(config: &StageConfig) -> Vec<PathBuf> {
    let Some(path) = config.model_path.as_deref() else {
        return Vec::new();
    };
    match config.load_mode {
        LoadMode::LayerPackage => {
            let package_dir = Path::new(path);
            let mut paths =
                layer_package_inspection_paths(package_dir, config.layer_start, config.layer_end);
            if paths.is_empty() {
                if let Some(metadata) = layer_package_metadata_path(package_dir) {
                    paths.push(metadata);
                } else {
                    paths.push(PathBuf::from(path));
                }
            }
            paths
        }
        LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => vec![PathBuf::from(path)],
    }
}

pub(super) fn layer_package_inspection_paths(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
) -> Vec<PathBuf> {
    let Some(manifest_path) = package_inspection_file(package_dir, Path::new("model-package.json"))
    else {
        return Vec::new();
    };
    let Ok(manifest) =
        serde_json::from_slice::<serde_json::Value>(&fs::read(manifest_path).unwrap_or_default())
    else {
        return Vec::new();
    };
    let Some(layers) = manifest.get("layers").and_then(|value| value.as_array()) else {
        return Vec::new();
    };
    layers
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| {
            let layer_index = layer
                .get("layer_index")
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(index as u32);
            if layer_index < layer_start || layer_index >= layer_end {
                return None;
            }
            let path = layer.get("path")?.as_str()?;
            package_inspection_file(package_dir, Path::new(path))
        })
        .collect()
}

fn layer_package_metadata_path(package_dir: &Path) -> Option<PathBuf> {
    package_inspection_file(package_dir, Path::new("shared/metadata.gguf"))
}

fn package_inspection_file(package_dir: &Path, relative_path: &Path) -> Option<PathBuf> {
    if relative_path.as_os_str().is_empty()
        || !relative_path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        return None;
    }

    let canonical_package = fs::canonicalize(package_dir).ok()?;
    if !canonical_package.is_dir() {
        return None;
    }
    let canonical_candidate = fs::canonicalize(canonical_package.join(relative_path)).ok()?;
    if !canonical_candidate.is_file() {
        return None;
    }

    let containment_root = hugging_face_repo_cache_root(&canonical_package)
        .unwrap_or_else(|| canonical_package.clone());
    canonical_candidate
        .starts_with(containment_root)
        .then_some(canonical_candidate)
}

fn hugging_face_repo_cache_root(canonical_package: &Path) -> Option<PathBuf> {
    let snapshots_dir = canonical_package.parent()?;
    if snapshots_dir.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let repo_root = snapshots_dir.parent()?;
    let encoded_repo = repo_root.file_name()?.to_str()?.strip_prefix("models--")?;
    let mut repo_parts = encoded_repo.split("--");
    if !matches!(
        (repo_parts.next(), repo_parts.next(), repo_parts.next()),
        (Some(owner), Some(repo), None) if !owner.is_empty() && !repo.is_empty()
    ) {
        return None;
    }
    fs::canonicalize(repo_root).ok()
}

pub(super) fn tensor_name_requires_recurrent_state(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains(".ssm")
        || lower.contains("ssm_")
        || lower.contains("time_mix")
        || lower.contains("recurrent")
        || lower.contains("rwkv")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_family_with_only_negative_tensor_evidence_stays_unknown() {
        assert!(matches!(
            reconcile_model_kv_capability(None, ModelKvCapability::KnownDense),
            ModelKvCapability::Unknown(_)
        ));
    }

    #[test]
    fn positive_recurrent_tensor_evidence_classifies_an_unknown_family() {
        assert_eq!(
            reconcile_model_kv_capability(None, ModelKvCapability::KnownRecurrent),
            ModelKvCapability::KnownRecurrent
        );
    }

    #[test]
    fn declared_and_inspected_capability_disagreement_is_unknown() {
        for (declared, inspected) in [
            (
                ModelKvCapability::KnownDense,
                ModelKvCapability::KnownRecurrent,
            ),
            (
                ModelKvCapability::KnownRecurrent,
                ModelKvCapability::KnownDense,
            ),
        ] {
            assert!(matches!(
                reconcile_model_kv_capability(Some(declared), inspected),
                ModelKvCapability::Unknown(_)
            ));
        }
    }
}
