use super::contract::{CacheAdmission, CachePolicy};
use super::gguf::{self, Expected};
use crate::automation::canary_receipts::Digest;
use crate::automation::replay_matrix::model_preflight::dimensions::Dimensions;
use crate::command::DynResult;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
};

#[derive(Deserialize)]
struct Plan {
    selected_models: Vec<Model>,
}
#[derive(Deserialize)]
struct Model {
    architecture: String,
    execution: Execution,
    artifact: Artifact,
    draft_artifact: Option<Artifact>,
    mmproj_artifact: Option<Artifact>,
}
#[derive(Deserialize)]
struct Execution {
    layer_end: u64,
    activation_width: u64,
    mtp_layers: u64,
}
#[derive(Deserialize)]
struct Artifact {
    repo: String,
    revision: String,
    files: Vec<String>,
    file_integrity: BTreeMap<String, Integrity>,
}
#[derive(Deserialize)]
struct Integrity {
    size_bytes: u64,
    blob_id: Digest,
}

#[derive(Debug, thiserror::Error)]
enum CacheError {
    #[error("immutable cache root must exist and be absolute")]
    Root,
    #[error("artifact revision must be lowercase 40-hex")]
    Revision,
    #[error("unsafe artifact repository coordinate")]
    Repository,
    #[error("artifact blob directory escapes cache root")]
    BlobDirectory,
    #[error("unsafe snapshot relative path")]
    RelativePath,
    #[error("missing artifact integrity")]
    Integrity,
    #[error("snapshot entry is not a content-addressed symlink")]
    Symlink,
    #[error("snapshot entry differs from pinned blob identity or size")]
    Identity,
    #[error("immutable cache blob SHA-256 mismatch")]
    Digest,
}

pub(super) fn verify(bytes: &[u8], policy: &CachePolicy) -> DynResult<CacheAdmission> {
    match policy {
        CachePolicy::NotChecked => Ok(CacheAdmission::NotChecked),
        CachePolicy::BlobIdentity { root } | CachePolicy::GgufMetadata { root } => {
            if !root.is_absolute() || !root.is_dir() {
                return Err(CacheError::Root.into());
            }
            let hub = match root.file_name().and_then(|name| name.to_str()) {
                Some("hub") => root.clone(),
                _ => root.join("hub"),
            }
            .canonicalize()?;
            let plan: Plan = serde_json::from_slice(bytes)?;
            for model in plan.selected_models {
                let target = verify_artifact(&hub, &model.artifact)?;
                let draft = model
                    .draft_artifact
                    .as_ref()
                    .map(|artifact| verify_artifact(&hub, artifact))
                    .transpose()?;
                if let Some(projector) = &model.mmproj_artifact {
                    verify_artifact(&hub, projector)?;
                }
                match policy {
                    CachePolicy::GgufMetadata { .. } => {
                        let expected = Dimensions {
                            architecture: model.architecture,
                            block_count: model.execution.layer_end,
                            activation_width: model.execution.activation_width,
                            mtp_layers: model.execution.mtp_layers,
                        };
                        gguf::verify(&target, Expected::Target(&expected))?;
                        if let Some(paths) = draft {
                            gguf::verify(&paths, Expected::Draft)?;
                        }
                    }
                    CachePolicy::NotChecked | CachePolicy::BlobIdentity { .. } => {}
                }
            }
            match policy {
                CachePolicy::GgufMetadata { .. } => Ok(CacheAdmission::GgufMetadata),
                CachePolicy::BlobIdentity { .. } => Ok(CacheAdmission::BlobIdentity),
                CachePolicy::NotChecked => Ok(CacheAdmission::NotChecked),
            }
        }
    }
}

fn verify_artifact(hub: &Path, artifact: &Artifact) -> DynResult<Vec<PathBuf>> {
    if artifact.revision.len() != 40
        || !artifact
            .revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CacheError::Revision.into());
    }
    if artifact.repo.split('/').count() != 2
        || artifact
            .repo
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".." || part.contains('\\'))
    {
        return Err(CacheError::Repository.into());
    }
    let repository = hub.join(format!("models--{}", artifact.repo.replace('/', "--")));
    let blobs = repository.join("blobs").canonicalize()?;
    if !blobs.starts_with(hub) {
        return Err(CacheError::BlobDirectory.into());
    }
    let snapshot = repository.join("snapshots").join(&artifact.revision);
    let mut paths = Vec::new();
    for relative in &artifact.files {
        let relative_path = Path::new(relative);
        if relative_path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(CacheError::RelativePath.into());
        }
        let path = snapshot.join(relative_path);
        let expected = artifact
            .file_integrity
            .get(relative)
            .ok_or(CacheError::Integrity)?;
        if !fs::symlink_metadata(&path)?.file_type().is_symlink() {
            return Err(CacheError::Symlink.into());
        }
        let resolved = path.canonicalize()?;
        if resolved.parent() != Some(blobs.as_path())
            || resolved.file_name().and_then(|name| name.to_str())
                != Some(expected.blob_id.as_str())
            || fs::metadata(&resolved)?.len() != expected.size_bytes
        {
            return Err(CacheError::Identity.into());
        }
        if Digest::of_file(&resolved)? != expected.blob_id {
            return Err(CacheError::Digest.into());
        }
        paths.push(resolved);
    }
    Ok(paths)
}
