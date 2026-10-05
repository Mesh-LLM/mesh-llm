//! Download a complete SafeTensors checkpoint into its named Hub snapshot.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};
use hf_hub::{RepoTypeModel, progress::Progress};
use skippy_model_artifact::{
    ModelArtifactFile, ModelFormat, ModelRepository, ResolvedModelArtifact,
};

use crate::{HfModelRepository, huggingface_repo_folder_name};

pub struct DownloadedCheckpointFile {
    pub file: ModelArtifactFile,
    pub path: PathBuf,
}

impl HfModelRepository {
    /// Return named snapshot paths, not extensionless blob paths: native
    /// checkpoint loading needs the weight, index, config and tokenizer together.
    pub async fn download_checkpoint_with_progress(
        &self,
        artifact: &ResolvedModelArtifact,
        progress: Option<Progress>,
    ) -> Result<Vec<DownloadedCheckpointFile>> {
        ensure!(
            artifact.format == ModelFormat::Safetensors,
            "checkpoint download requires SafeTensors"
        );
        let siblings = self
            .list_files(&artifact.source_repo, &artifact.source_revision)
            .await?;
        let index_name = Path::new(&artifact.primary_file)
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join("model.safetensors.index.json")
            .to_string_lossy()
            .into_owned();
        let index_bytes = if siblings.iter().any(|file| file.path == index_name) {
            let path = self
                .download_file_with_progress(
                    &artifact.source_repo,
                    &artifact.source_revision,
                    &index_name,
                    progress.clone(),
                )
                .await?;
            Some(
                std::fs::read(&path)
                    .with_context(|| format!("read SafeTensors index {}", path.display()))?,
            )
        } else {
            None
        };
        let files = skippy_model_artifact::checkpoint::checkpoint_files(
            &artifact.primary_file,
            &siblings,
            index_bytes.as_deref(),
        )?;
        let mut downloaded = Vec::with_capacity(files.len());
        for file in files {
            let downloaded_path = self
                .download_file_with_progress(
                    &artifact.source_repo,
                    &artifact.source_revision,
                    &file.path,
                    progress.clone(),
                )
                .await?;
            let path = snapshot_file_path(&self.cache_dir, artifact, &file.path, &downloaded_path)?;
            downloaded.push(DownloadedCheckpointFile { file, path });
        }
        Ok(downloaded)
    }
}

fn snapshot_file_path(
    cache_dir: &Path,
    artifact: &ResolvedModelArtifact,
    file: &str,
    downloaded: &Path,
) -> Result<PathBuf> {
    ensure!(
        !file.is_empty()
            && Path::new(file)
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "checkpoint file must remain within its Hub snapshot: {file:?}"
    );
    ensure!(
        Path::new(&artifact.source_revision).components().count() == 1
            && Path::new(&artifact.source_revision)
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "invalid checkpoint revision: {}",
        artifact.source_revision
    );
    let path = cache_dir
        .join(huggingface_repo_folder_name(
            &artifact.source_repo,
            RepoTypeModel,
        ))
        .join("snapshots")
        .join(&artifact.source_revision)
        .join(file);
    let metadata = path.metadata().with_context(|| {
        format!(
            "Hub snapshot is missing downloaded checkpoint file {}",
            path.display()
        )
    })?;
    ensure!(
        metadata.is_file(),
        "Hub checkpoint entry is not a file: {}",
        path.display()
    );
    ensure!(
        metadata.len() == downloaded.metadata()?.len(),
        "Hub snapshot file size differs from downloaded blob: {}",
        path.display()
    );
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_named_snapshot_file_from_blob_download() {
        let temp = tempfile::tempdir().unwrap();
        let artifact = ResolvedModelArtifact {
            model_id: "owner/repo".into(),
            source_repo: "owner/repo".into(),
            source_revision: "0123456789012345678901234567890123456789".into(),
            selector: None,
            format: ModelFormat::Safetensors,
            files: vec![ModelArtifactFile::new("model.safetensors")],
            primary_file: "model.safetensors".into(),
            canonical_ref: "owner/repo@rev/model.safetensors".into(),
            distribution_id: "model".into(),
        };
        let blob = temp.path().join("blob");
        std::fs::write(&blob, b"weights").unwrap();
        let snapshot = temp
            .path()
            .join("models--owner--repo/snapshots")
            .join(&artifact.source_revision)
            .join("model.safetensors");
        std::fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        std::fs::write(&snapshot, b"weights").unwrap();
        assert_eq!(
            snapshot_file_path(temp.path(), &artifact, "model.safetensors", &blob).unwrap(),
            snapshot
        );
        assert!(
            snapshot_file_path(temp.path(), &artifact, "../escape.safetensors", &blob).is_err()
        );
    }
}
