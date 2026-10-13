//! Owned report leaves: reject links before output, replace bytes without following a leaf.
use crate::command::DynResult;
use std::{io::Write, path::Path};
pub(super) fn admit(root: &Path, path: &Path) -> DynResult<()> {
    let parent = path.parent().ok_or("report output parent")?;
    super::competitive_matrix::existing_parents(root, parent)?;
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(()),
        Ok(_) => Err("report output leaf refuses symlink/special file".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}
pub(super) fn replace(root: &Path, path: &Path, bytes: &[u8]) -> DynResult<()> {
    admit(root, path)?;
    let parent = path.parent().ok_or("report output parent")?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.flush()?;
    temporary.as_file().sync_all()?;
    // Recheck after constructing the exclusively created temporary file. Atomic rename
    // replaces a raced leaf rather than opening/following it; failed persist removes its
    // own temporary through RAII. Concurrent mutation of owned parent directories is not
    // a supported archive writer; the parent boundary is checked again before persist.
    admit(root, path)?;
    temporary.persist(path).map_err(|failure| failure.error)?;
    Ok(())
}
pub(super) fn json<T: serde::Serialize>(root: &Path, path: &Path, value: &T) -> DynResult<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    replace(root, path, &bytes)
}
