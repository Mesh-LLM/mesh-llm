//! Bounded worker/manifest inputs and atomic evidence publication without replacement.
use crate::command::DynResult;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

pub(super) const INPUT_BYTES: usize = 128 * 1024;
pub(super) const RECEIPT_BYTES: usize = 64 * 1024;
pub(super) const MANIFEST_BYTES: usize = 16 * 1024 * 1024;

pub(super) fn read<T: DeserializeOwned>(path: &Path, limit: usize) -> DynResult<T> {
    if !path.is_absolute() || !std::fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("benchmark input must be an absolute regular file".into());
    }
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err("benchmark input changed file type before opening".into());
    }
    let cap = u64::try_from(
        limit
            .checked_add(1)
            .ok_or("benchmark input limit overflow")?,
    )?;
    let mut bytes = Vec::new();
    file.take(cap).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("benchmark input exceeds its declared byte limit".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

pub(super) fn publish<T: Serialize>(path: &Path, value: &T, limit: usize) -> DynResult<()> {
    if !path.is_absolute() {
        return Err("benchmark output must be absolute".into());
    }
    let parent = path.parent().ok_or("benchmark output has no parent")?;
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    if bytes.len() > limit {
        return Err("benchmark output exceeds its declared byte limit".into());
    }
    let mut staging = tempfile::NamedTempFile::new_in(parent)?;
    staging.write_all(&bytes)?;
    staging.as_file().sync_all()?;
    staging.persist_noclobber(path)?;
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
#[path = "evidence_io_tests.rs"]
mod tests;
