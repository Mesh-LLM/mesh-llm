//! Explicit-root local cache operations. These never contact the Hub.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
};

/// Serialize cooperating cache mutations. Other Hub clients do not use this lock.
/// The file remains after unlocking so waiting processes always share one inode.
pub fn lock_cache(root: &Path) -> Result<File> {
    std::fs::create_dir_all(root)?;
    let lock_path = root.join(".model-cache.lock");
    if let Ok(metadata) = std::fs::symlink_metadata(&lock_path) {
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "cache lock must be a regular file"
        );
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)?;
    file.try_lock()
        .context("model cache is busy; retry after the other cache operation finishes")?;
    Ok(file)
}

fn model_folder(repo: &str) -> Result<String> {
    let parts = repo.split('/').collect::<Vec<_>>();
    ensure!(
        (1..=2).contains(&parts.len()),
        "expected a model repository id, not a path or revision"
    );
    ensure!(
        parts.iter().all(|part| !part.is_empty()
            && *part != "."
            && *part != ".."
            && !part.contains("--")
            && !part.contains("..")
            && part
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))),
        "invalid model repository id"
    );
    Ok(format!("models--{}", parts.join("--")))
}

#[derive(Debug, Serialize)]
pub struct LocalRepositoryRemoval {
    pub repo: String,
    pub path: PathBuf,
    pub scope: &'static str,
    pub status: &'static str,
}

/// Remove all local revisions of one model repository, including its own blobs.
/// Symlinks inside the repository are unlinked, never followed. Other repository
/// directories and the shared download-lock directory are left untouched.
/// Callers must stop non-cooperating downloads and users of this repository first.
pub fn remove_repository(root: &Path, repo: &str, dry_run: bool) -> Result<LocalRepositoryRemoval> {
    let folder = model_folder(repo)?;
    let mut report = LocalRepositoryRemoval {
        repo: repo.into(),
        path: root.join(&folder),
        scope: "all-local-revisions",
        status: "not-found",
    };
    if !root.try_exists()? {
        return Ok(report);
    }
    let root = root.canonicalize()?;
    let _lock = if dry_run {
        None
    } else {
        Some(lock_cache(&root)?)
    };
    let path = root.join(folder);
    report.path = path.clone();
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(report),
        Err(error) => return Err(error.into()),
    };
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "cached repository root must be a directory, not a symlink"
    );
    if dry_run {
        report.status = "planned";
    } else {
        std::fs::remove_dir_all(&path)
            .with_context(|| format!("remove local model repository {}", path.display()))?;
        report.status = "removed";
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removal_preserves_other_repositories_and_dry_run_changes_nothing() {
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("models--org--first");
        let second = root.path().join("models--org--second");
        std::fs::create_dir_all(first.join("snapshots/a")).unwrap();
        std::fs::create_dir_all(first.join("snapshots/b")).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(second.join("blob"), b"preserved").unwrap();
        assert_eq!(
            remove_repository(root.path(), "org/first", true)
                .unwrap()
                .status,
            "planned"
        );
        assert!(first.join("snapshots/b").exists());
        assert!(!root.path().join(".model-cache.lock").exists());
        assert_eq!(
            remove_repository(root.path(), "org/first", false)
                .unwrap()
                .status,
            "removed"
        );
        assert_eq!(std::fs::read(second.join("blob")).unwrap(), b"preserved");
        assert!(!first.exists());
    }

    #[test]
    fn removal_rejects_paths_revisions_and_ambiguous_folder_ids() {
        for repo in [
            "../outside",
            "/tmp",
            "a/b/c",
            "org/repo@main",
            "org/repo:Q4",
            "a--b",
            "a/",
            "a/..",
        ] {
            assert!(model_folder(repo).is_err(), "{repo}");
        }
    }

    #[test]
    fn cooperating_mutation_is_exclusive() {
        let root = tempfile::tempdir().unwrap();
        let lock = lock_cache(root.path()).unwrap();
        assert!(lock_cache(root.path()).is_err());
        drop(lock);
        assert!(lock_cache(root.path()).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn removal_never_follows_repository_or_nested_symlinks() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("keep"), b"safe").unwrap();
        let repo = root.path().join("models--org--repo");
        symlink(outside.path(), &repo).unwrap();
        assert!(remove_repository(root.path(), "org/repo", false).is_err());
        std::fs::remove_file(&repo).unwrap();
        std::fs::create_dir(&repo).unwrap();
        symlink(outside.path(), repo.join("external")).unwrap();
        remove_repository(root.path(), "org/repo", false).unwrap();
        assert_eq!(std::fs::read(outside.path().join("keep")).unwrap(), b"safe");
    }
}
