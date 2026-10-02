//! Bounded source-owned family/parity policy JSON, never workflow authority.
use super::process;
use crate::command::DynResult;
use serde_json::Value;
use std::{fs, io::Read, path::Path};
const MAXIMUM: u64 = 8 * 1024 * 1024;
pub(super) fn read(root: &Path, relative: &str) -> DynResult<Vec<u8>> {
    let path = root.join(relative);
    let before = fs::symlink_metadata(&path)?;
    if !before.is_file() || before.len() > MAXIMUM || !path.canonicalize()?.starts_with(root) {
        return Err("policy document must be a contained regular source at most 8 MiB".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options.open(&path)?;
    let opened = file.metadata()?;
    if !opened.is_file()
        || opened.len() > MAXIMUM
        || !path.canonicalize()?.starts_with(root)
        || !same_file(&before, &opened)
    {
        return Err("policy document changed during regular source admission".into());
    }
    read_opened(root, &path, &mut file, &opened)
}
fn read_opened(
    root: &Path,
    path: &Path,
    file: &mut fs::File,
    opened: &fs::Metadata,
) -> DynResult<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 65536];
    loop {
        process::check()?;
        let count = file.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        if count > (MAXIMUM as usize).saturating_sub(bytes.len()) {
            return Err("policy document exceeds 8 MiB".into());
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    let after = file.metadata()?;
    if bytes.len() as u64 != opened.len() || !same_file(opened, &after) {
        return Err("policy document changed during bounded read".into());
    }
    if !path.canonicalize()?.starts_with(root) || !same_file(opened, &fs::symlink_metadata(path)?) {
        return Err("policy document pathname changed during read".into());
    }
    process::check()?;
    Ok(bytes)
}
pub(super) fn json(root: &Path, relative: &str) -> DynResult<Value> {
    Ok(serde_json::from_slice(&read(root, relative)?)?)
}

/// Match the opened handle to the pathname witness; Unix identity includes device/inode.
pub(super) fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    if !right.is_file()
        || left.len() != right.len()
        || !matches!((left.modified(), right.modified()), (Ok(left), Ok(right)) if left == right)
    {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if left.dev() != right.dev()
            || left.ino() != right.ino()
            || left.ctime() != right.ctime()
            || left.ctime_nsec() != right.ctime_nsec()
        {
            return false;
        }
    }
    true
}
#[cfg(test)]
#[path = "policy_document_tests.rs"]
mod tests;
