//! Exact local package snapshot binding. No acquisition or inference.
use anyhow::{Result, ensure};
use serde::Serialize;
use skippy_model_ref::package_reference::PackageReference;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize)]
pub struct CachedPackageSnapshot {
    repo: String,
    requested_revision: String,
    commit: String,
    snapshot_path: PathBuf,
}
impl CachedPackageSnapshot {
    pub fn repo(&self) -> &str {
        &self.repo
    }
    pub fn requested_revision(&self) -> &str {
        &self.requested_revision
    }
    pub fn commit(&self) -> &str {
        &self.commit
    }
    pub fn snapshot_path(&self) -> &Path {
        &self.snapshot_path
    }
}

pub fn resolve(reference: &PackageReference, cache_root: &Path) -> Result<CachedPackageSnapshot> {
    let root = cache_root.canonicalize()?;
    directory(&root)?;
    let repo = root.join(super::local_cache::model_folder(reference.repo())?);
    directory(&repo)?;
    if !is_commit(reference.revision()) {
        check_ref_path(&repo, reference.revision())?;
    }
    let commit = hf_hub::cache::resolve_cached_revision(
        &root,
        reference.repo(),
        hf_hub::RepoTypeModel,
        reference.revision(),
    )?
    .ok_or_else(|| anyhow::anyhow!("requested package revision is absent from cache"))?;
    ensure!(
        is_commit(&commit),
        "cache revision must resolve to an immutable commit"
    );
    let snapshots = repo.join("snapshots");
    directory(&snapshots)?;
    let expected = snapshots.join(&commit);
    directory(&expected)?;
    ensure!(
        expected.canonicalize()? == expected,
        "snapshot directory escaped exact repository"
    );
    check_manifest(&repo, &expected)?;
    let (owner, name) = reference
        .repo()
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("package repository has no namespace"))?;
    // Pass the captured commit, so a branch moving now cannot retarget lookup.
    let api = super::build_hf_sync_api_in(&root)?;
    let returned = api
        .model(owner, name)
        .snapshot_download()
        .revision(commit.clone())
        .local_files_only(true)
        .send()?;
    ensure!(
        returned == expected,
        "native cache resolver returned a foreign snapshot"
    );
    directory(&expected)?;
    check_manifest(&repo, &expected)?;
    Ok(CachedPackageSnapshot {
        repo: reference.repo().into(),
        requested_revision: reference.revision().into(),
        commit,
        snapshot_path: expected,
    })
}

fn directory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "cache directory must not redirect repository custody"
    );
    Ok(())
}
fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn check_ref_path(repo: &Path, revision: &str) -> Result<()> {
    let mut path = repo.join("refs");
    directory(&path)?;
    let mut parts = revision.split('/').peekable();
    while let Some(part) = parts.next() {
        path.push(part);
        if parts.peek().is_some() {
            directory(&path)?;
        } else {
            let metadata = fs::symlink_metadata(&path)?;
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "cache ref must be a regular file"
            );
        }
    }
    Ok(())
}
fn check_manifest(repo: &Path, snapshot: &Path) -> Result<()> {
    let manifest = snapshot.join("model-package.json");
    let metadata = fs::symlink_metadata(&manifest)?;
    if metadata.file_type().is_symlink() {
        let blobs = repo.join("blobs");
        directory(&blobs)?;
        ensure!(
            manifest.canonicalize()?.starts_with(&blobs),
            "manifest pointer is outside repository blobs"
        );
    }
    ensure!(
        fs::metadata(&manifest)?.is_file(),
        "cached package manifest must be a regular file"
    );
    Ok(())
}

#[cfg(test)]
#[path = "package_cache_tests.rs"]
mod tests;
