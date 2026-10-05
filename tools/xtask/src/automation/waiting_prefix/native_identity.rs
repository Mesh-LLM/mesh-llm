//! Pin the complete provided runtime artifact tree without silently skipped inputs.
use crate::command::DynResult;
use std::path::Path;

fn complete_tree(root: &Path) -> DynResult<()> {
    let canonical_root = root.canonicalize()?;
    let mut pending = vec![(canonical_root.clone(), 0_u32)];
    let mut entries = 0_u64;
    let mut files = 0_u64;
    while let Some((directory, depth)) = pending.pop() {
        if depth > 32 {
            return Err("native artifact tree exceeds 32 directory levels".into());
        }
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            entries += 1;
            if entries > 10000 {
                return Err("native artifact tree exceeds 10000 entries".into());
            }
            let kind = entry.file_type()?;
            if kind.is_dir() {
                pending.push((entry.path(), depth + 1));
            } else if kind.is_file() {
                files += 1;
            } else if kind.is_symlink() {
                let target = entry.path().canonicalize()?;
                if !target.is_file() || !target.starts_with(&canonical_root) {
                    return Err(
                        "native artifact links must resolve to regular files inside the tree"
                            .into(),
                    );
                }
                files += 1;
            } else {
                return Err("native artifact identity rejects special files".into());
            }
        }
    }
    if files == 0 {
        return Err("native artifact tree must contain pinned files".into());
    }
    Ok(())
}

pub(super) fn verify(root: &Path, expected: &str) -> DynResult<String> {
    if !root.is_absolute()
        || !root.is_dir()
        || expected.len() != 64
        || !expected.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("native artifact pin requires an absolute directory and SHA-256".into());
    }
    complete_tree(root)?;
    let actual = crate::product::digest::tree_sha256(root).map_err(|error| error.error)?;
    if actual != expected {
        return Err("provided native artifact tree SHA-256 mismatch".into());
    }
    Ok(actual)
}

#[cfg(test)]
#[path = "native_identity_tests.rs"]
mod tests;
