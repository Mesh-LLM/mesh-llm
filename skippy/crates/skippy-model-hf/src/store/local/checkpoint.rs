//! Offline lookup of complete SafeTensors checkpoints in named Hub snapshots.

use std::{
    fs,
    path::{Component, Path, PathBuf},
    time::UNIX_EPOCH,
};

use skippy_model_artifact::{
    ModelArtifactFile, checkpoint::checkpoint_files, select_primary_artifact_file,
};
use skippy_model_ref::ModelRef;

use crate::huggingface_repo_folder_name;

pub(super) fn find_cached_safetensors_checkpoint(
    cache_root: &Path,
    model: &ModelRef,
) -> Option<PathBuf> {
    let (owner, repo) = model.repo.split_once('/')?;
    if !safe_repo_part(owner) || !safe_repo_part(repo) {
        return None;
    }
    if model.selector.is_none()
        && skippy_model_artifact::selection::repo_prefers_gguf_only(&model.repo)
    {
        return None;
    }
    let repo_root = cache_root.join(huggingface_repo_folder_name(
        &model.repo,
        hf_hub::RepoTypeModel,
    ));
    snapshot_candidates(&repo_root, model.revision.as_deref())?
        .into_iter()
        .find_map(|snapshot| checkpoint_in_snapshot(&snapshot, model.selector.as_deref()))
}

fn safe_repo_part(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.contains("--")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
}

fn safe_revision(value: &str) -> bool {
    !value.is_empty()
        && Path::new(value)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn snapshot_candidates(repo_root: &Path, revision: Option<&str>) -> Option<Vec<PathBuf>> {
    let snapshots = repo_root.join("snapshots");
    if let Some(revision) = revision {
        if !safe_revision(revision) {
            return None;
        }
        let commit = cached_ref(repo_root, revision).unwrap_or_else(|| revision.to_string());
        return Some(vec![snapshots.join(commit)]);
    }

    let mut candidates = fs::read_dir(&snapshots)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    candidates.sort_by_key(|path| {
        std::cmp::Reverse(
            path.metadata()
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok()),
        )
    });
    if let Some(main) = cached_ref(repo_root, "main") {
        let current = snapshots.join(main);
        if let Some(position) = candidates.iter().position(|path| path == &current) {
            candidates.swap(0, position);
        }
    }
    Some(candidates)
}

fn cached_ref(repo_root: &Path, revision: &str) -> Option<String> {
    let commit = fs::read_to_string(repo_root.join("refs").join(revision)).ok()?;
    let commit = commit.trim();
    safe_revision(commit).then(|| commit.to_string())
}

fn checkpoint_in_snapshot(snapshot: &Path, selector: Option<&str>) -> Option<PathBuf> {
    let files = snapshot_files(snapshot)?;
    let primary = select_primary_artifact_file(selector, &files).ok()?;
    if !primary.path.ends_with(".safetensors") {
        return None;
    }
    let primary_path = snapshot.join(&primary.path);
    let checkpoint_root = primary_path.parent()?;
    let index = checkpoint_root.join("model.safetensors.index.json");
    let index_bytes = if index.is_file() {
        Some(fs::read(index).ok()?)
    } else {
        None
    };
    let required = checkpoint_files(&primary.path, &files, index_bytes.as_deref()).ok()?;
    required
        .iter()
        .all(|file| {
            snapshot
                .join(&file.path)
                .metadata()
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
        })
        .then(|| checkpoint_root.to_path_buf())
}

fn snapshot_files(snapshot: &Path) -> Option<Vec<ModelArtifactFile>> {
    let mut pending = vec![snapshot.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            if entry.file_type().ok()?.is_dir() {
                pending.push(path);
            } else if path.is_file() {
                let relative = path
                    .strip_prefix(snapshot)
                    .ok()?
                    .to_str()?
                    .replace('\\', "/");
                files.push(ModelArtifactFile::new(relative));
            }
        }
    }
    Some(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(cache: &Path, revision: &str) -> PathBuf {
        let path = cache.join("models--org--repo/snapshots").join(revision);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn complete_checkpoint(path: &Path) {
        fs::write(path.join("config.json"), b"{}").unwrap();
        fs::write(path.join("tokenizer.json"), b"{}").unwrap();
        fs::write(path.join("model.safetensors"), b"weights").unwrap();
    }

    #[test]
    fn finds_complete_checkpoint_without_hub_ref_marker() {
        let cache = tempfile::tempdir().unwrap();
        let path = snapshot(cache.path(), "commit-a");
        complete_checkpoint(&path);
        let model = ModelRef::parse("org/repo").unwrap();
        assert_eq!(
            find_cached_safetensors_checkpoint(cache.path(), &model),
            Some(path)
        );
    }

    #[test]
    fn rejects_incomplete_shards_and_respects_revision_and_selector() {
        let cache = tempfile::tempdir().unwrap();
        let path = snapshot(cache.path(), "commit-a");
        complete_checkpoint(&path);
        fs::remove_file(path.join("model.safetensors")).unwrap();
        fs::write(path.join("model-00001-of-00002.safetensors"), b"shard").unwrap();
        let model = ModelRef::parse("org/repo").unwrap();
        assert_eq!(
            find_cached_safetensors_checkpoint(cache.path(), &model),
            None
        );

        fs::write(path.join("model-00002-of-00002.safetensors"), b"shard").unwrap();
        assert_eq!(
            find_cached_safetensors_checkpoint(cache.path(), &model),
            Some(path.clone())
        );
        let other_revision = ModelRef::parse("org/repo@commit-b").unwrap();
        assert_eq!(
            find_cached_safetensors_checkpoint(cache.path(), &other_revision),
            None
        );
        let gguf_selector = ModelRef::parse("org/repo:Q4_K_M").unwrap();
        assert_eq!(
            find_cached_safetensors_checkpoint(cache.path(), &gguf_selector),
            None
        );
    }

    #[test]
    fn indexed_checkpoint_requires_every_referenced_shard() {
        let cache = tempfile::tempdir().unwrap();
        let path = snapshot(cache.path(), "commit-a");
        fs::write(path.join("config.json"), b"{}").unwrap();
        fs::write(path.join("tokenizer.json"), b"{}").unwrap();
        fs::write(
            path.join("model.safetensors.index.json"),
            br#"{"weight_map":{"layer":"model.safetensors-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        let model = ModelRef::parse("org/repo").unwrap();
        assert_eq!(
            find_cached_safetensors_checkpoint(cache.path(), &model),
            None
        );
        fs::write(
            path.join("model.safetensors-00001-of-00001.safetensors"),
            b"weights",
        )
        .unwrap();
        assert_eq!(
            find_cached_safetensors_checkpoint(cache.path(), &model),
            Some(path)
        );
    }
}
