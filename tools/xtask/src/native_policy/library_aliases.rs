//! Coalesce identical staged Linux libraries after relocation, before archiving.
use crate::command::DynResult;

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [directory] = args else {
        return Err("usage: native deduplicate-linux-libraries LIB_DIR".into());
    };
    deduplicate(std::path::Path::new(directory))
}

#[cfg(not(unix))]
fn deduplicate(_directory: &std::path::Path) -> DynResult<()> {
    Err("Linux library deduplication requires a Unix packaging host".into())
}

#[cfg(unix)]
fn deduplicate(directory: &std::path::Path) -> DynResult<()> {
    use std::{collections::BTreeMap, fs, os::unix::fs::PermissionsExt};

    if !fs::symlink_metadata(directory)?.is_dir() {
        return Err("library directory must be a real directory".into());
    }
    let mut entries = fs::read_dir(directory)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    let mut originals = BTreeMap::new();
    for entry in entries {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.ends_with(".so") || name.contains(".so.")) {
            continue;
        }
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_file() {
            return Err(
                format!("staged library must be a regular file: {}", path.display()).into(),
            );
        }
        let digest = crate::product::digest::file_sha256(&path).map_err(|failure| failure.error)?;
        let key = (metadata.len(), metadata.permissions().mode(), digest);
        if let Some(original) = originals.get(&key) {
            // Replace atomically: a failed link must leave the staged copy intact.
            let temporary = tempfile::tempdir_in(directory)?;
            let alias = temporary.path().join("alias");
            fs::hard_link(original, &alias)?;
            fs::rename(alias, path)?;
        } else {
            originals.insert(key, path);
        }
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{MetadataExt, PermissionsExt, symlink},
    };

    #[test]
    fn identical_aliases_share_bytes_but_distinct_bytes_and_modes_do_not() -> DynResult<()> {
        let root = tempfile::tempdir()?;
        for (name, bytes, mode) in [
            ("libcuda.so", "cuda", 0o755),
            ("libcuda.so.0", "cuda", 0o755),
            ("libcuda.so.0.29.0", "cuda", 0o755),
            ("libother.so", "else", 0o755),
            ("libmode.so", "cuda", 0o644),
            ("README", "cuda", 0o755),
        ] {
            let path = root.path().join(name);
            fs::write(&path, bytes)?;
            fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
        }
        deduplicate(root.path())?;
        deduplicate(root.path())?;
        let inode = fs::metadata(root.path().join("libcuda.so"))?.ino();
        for name in ["libcuda.so.0", "libcuda.so.0.29.0"] {
            assert_eq!(fs::metadata(root.path().join(name))?.ino(), inode);
            assert_eq!(fs::read(root.path().join(name))?, b"cuda");
        }
        for name in ["libother.so", "libmode.so", "README"] {
            assert_ne!(fs::metadata(root.path().join(name))?.ino(), inode);
        }
        Ok(())
    }

    #[test]
    fn refuses_symlink_inputs_without_modifying_their_targets() -> DynResult<()> {
        let root = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        let target = outside.path().join("outside");
        fs::write(&target, "unchanged")?;
        symlink(&target, root.path().join("libescape.so"))?;
        assert!(deduplicate(root.path()).is_err());
        assert_eq!(fs::read(target)?, b"unchanged");
        Ok(())
    }
}
