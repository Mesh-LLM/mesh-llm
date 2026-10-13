use std::{fs, io::Write, path::Path};

use serde::Serialize;

use crate::command::DynResult;

pub(super) fn bytes(document: &impl Serialize) -> DynResult<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(document)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(super) fn publish(bytes: &[u8], output: &Path) -> DynResult<()> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)?;
    staged.write_all(bytes)?;
    staged.as_file().sync_all()?;
    staged.persist(output)?;
    Ok(())
}
