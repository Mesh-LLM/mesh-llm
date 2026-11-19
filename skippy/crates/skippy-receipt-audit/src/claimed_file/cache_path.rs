//! Where the sparse copy of a claimed file lives: under the cache root, and
//! nowhere else.
//!
//! The path is `root/owner--name/commit/path/to/file.gguf`, built one
//! component at a time from a checked name. `Path::join` with an absolute
//! path or a `..` would leave the root, so nothing is joined that is not a
//! single plain component, and the result is checked again: it is under the
//! root as written, no component below the root is a symlink, and what
//! exists of it resolves to a place under the root.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::hf_file::HfFile;

/// The path of the sparse copy of `pinned`, a file at a commit, under
/// `cache_root`, which is created.
pub(super) fn sparse_copy_path(cache_root: &Path, pinned: &HfFile) -> Result<PathBuf> {
    pinned
        .check()
        .with_context(|| format!("{pinned} names no path in the cache"))?;
    if !pinned.is_pinned() {
        bail!(
            "the cache holds files by commit, and {} is not one",
            pinned.revision
        );
    }
    std::fs::create_dir_all(cache_root)
        .with_context(|| format!("create {}", cache_root.display()))?;
    // the root may be a symlink: what is under it may not
    let root = cache_root
        .canonicalize()
        .with_context(|| format!("resolve {}", cache_root.display()))?;
    let mut parts = vec![pinned.repo.replace('/', "--"), pinned.revision.clone()];
    parts.extend(pinned.file.split('/').map(str::to_owned));
    let mut path = cache_root.to_owned();
    let mut resolved = root.clone();
    for part in &parts {
        if !is_plain_component(part) {
            bail!("{part:?} is not a name of one file or directory");
        }
        path.push(part);
        resolved.push(part);
        refuse_symlink(&resolved)?;
    }
    let inside = resolved
        .strip_prefix(&root)
        .is_ok_and(|rest| rest.components().count() == parts.len());
    if !inside {
        bail!(
            "the sparse copy of {pinned} would be outside {}",
            cache_root.display()
        );
    }
    // what exists of the path, resolved by the system, is under the root
    let existing = resolved
        .ancestors()
        .find(|ancestor| ancestor.exists())
        .unwrap_or(&root);
    let real = existing
        .canonicalize()
        .with_context(|| format!("resolve {}", existing.display()))?;
    if !real.starts_with(&root) {
        bail!(
            "{} resolves to {}, outside {}",
            existing.display(),
            real.display(),
            root.display()
        );
    }
    Ok(path)
}

/// Whether `part` is one normal component of a path: no separator, no
/// prefix or root, and not `.` or `..`.
fn is_plain_component(part: &str) -> bool {
    let mut components = Path::new(part).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(name)), None) if name == part
    )
}

/// Refuses `path` when it is a symlink: a directory of the cache that
/// leads elsewhere, or a file that is another.
fn refuse_symlink(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => bail!(
            "{} is a symlink: the cache of claimed models holds none",
            path.display()
        ),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn pinned(file: &str) -> HfFile {
        HfFile {
            repo: "owner/model".into(),
            revision: COMMIT.into(),
            file: file.into(),
        }
    }

    #[test]
    fn the_copy_is_under_the_root_by_repo_commit_and_file() {
        let cache = tempfile::tempdir().unwrap();
        let root = cache.path().join("receipts").join("claimed");
        let path = sparse_copy_path(&root, &pinned("sub/model.gguf")).unwrap();
        assert_eq!(
            path,
            root.join("owner--model")
                .join(COMMIT)
                .join("sub")
                .join("model.gguf")
        );
        assert!(root.is_dir());
        // nothing below the root is made to name the path
        assert!(!root.join("owner--model").exists());
    }

    #[test]
    fn a_name_that_leaves_the_root_has_no_path() {
        let cache = tempfile::tempdir().unwrap();
        let outside = cache.path().join("outside.gguf");
        let absolute = outside.to_str().unwrap().to_owned();
        for file in [
            absolute.as_str(),
            "/etc/x.gguf",
            "../x.gguf",
            "a/../../../x.gguf",
            "..",
            "a//x.gguf",
            "a/./x.gguf",
            "a\\..\\x.gguf",
            "C:/x.gguf",
            "C:x.gguf",
            "",
        ] {
            let refused = sparse_copy_path(&cache.path().join("root"), &pinned(file));
            assert!(refused.is_err(), "{file:?} has the path {refused:?}");
        }
        for (repo, revision) in [
            ("owner/model", "main"),
            ("owner/model", ".."),
            ("owner/model", "../../../../etc"),
            ("../model", COMMIT),
            ("owner/..", COMMIT),
            ("/abs/model", COMMIT),
            ("owner", COMMIT),
        ] {
            let name = HfFile {
                repo: repo.into(),
                revision: revision.into(),
                file: "x.gguf".into(),
            };
            let refused = sparse_copy_path(&cache.path().join("root"), &name);
            assert!(refused.is_err(), "{name} has the path {refused:?}");
        }
        assert!(!outside.exists());
    }

    #[test]
    fn only_one_plain_component_is_joined() {
        for part in ["x.gguf", "owner--model", COMMIT] {
            assert!(is_plain_component(part), "{part}");
        }
        for part in ["", ".", "..", "/", "/x", "a/b", "a/", "./a"] {
            assert!(!is_plain_component(part), "{part:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_below_the_root_is_refused() {
        use std::os::unix::fs::symlink;
        let cache = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let root = cache.path().join("root");
        let name = pinned("sub/model.gguf");
        // the repo's directory leads out of the cache
        std::fs::create_dir_all(&root).unwrap();
        symlink(elsewhere.path(), root.join("owner--model")).unwrap();
        let refused = sparse_copy_path(&root, &name).unwrap_err().to_string();
        assert!(refused.contains("symlink"), "{refused}");
        std::fs::remove_file(root.join("owner--model")).unwrap();
        // a directory deeper down does
        let commit = root.join("owner--model").join(COMMIT);
        std::fs::create_dir_all(&commit).unwrap();
        symlink(elsewhere.path(), commit.join("sub")).unwrap();
        assert!(sparse_copy_path(&root, &name).is_err());
        std::fs::remove_file(commit.join("sub")).unwrap();
        // the file itself is a link to another, here or not yet anywhere
        std::fs::create_dir_all(commit.join("sub")).unwrap();
        let target = elsewhere.path().join("victim");
        symlink(&target, commit.join("sub").join("model.gguf")).unwrap();
        assert!(sparse_copy_path(&root, &name).is_err());
        std::fs::write(&target, b"kept").unwrap();
        assert!(sparse_copy_path(&root, &name).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"kept");
        std::fs::remove_file(commit.join("sub").join("model.gguf")).unwrap();
        // a link inside the cache to a place inside it is one too
        symlink(commit.join("sub"), commit.join("other")).unwrap();
        assert!(sparse_copy_path(&root, &pinned("other/model.gguf")).is_err());
        assert!(sparse_copy_path(&root, &name).is_ok());
        assert!(std::fs::read_dir(elsewhere.path()).unwrap().count() == 1);
    }

    #[cfg(unix)]
    #[test]
    fn a_root_that_is_a_symlink_still_holds_copies() {
        use std::os::unix::fs::symlink;
        let cache = tempfile::tempdir().unwrap();
        let real = cache.path().join("real");
        std::fs::create_dir_all(&real).unwrap();
        let root = cache.path().join("root");
        symlink(&real, &root).unwrap();
        let path = sparse_copy_path(&root, &pinned("model.gguf")).unwrap();
        assert_eq!(
            path,
            root.join("owner--model").join(COMMIT).join("model.gguf")
        );
    }
}
