use crate::command::DynResult;
use serde::Serialize;
use std::{
    io::{Read, Write},
    path::Path,
};
pub(super) fn read(path: &Path, maximum: usize) -> DynResult<Vec<u8>> {
    if !path.is_absolute() || !std::fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("suffix input requires absolute regular file".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("suffix input changed type".into());
    }
    let mut bytes = Vec::new();
    file.take((maximum + 1) as u64).read_to_end(&mut bytes)?;
    if bytes.len() > maximum {
        return Err("suffix input exceeds bound".into());
    }
    Ok(bytes)
}
pub(super) fn publish(path: &Path, value: &impl Serialize) -> DynResult<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > 16777216 {
        return Err("suffix receipt too large".into());
    }
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("output parent")?)?;
    file.write_all(&bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)?;
    Ok(())
}
pub(super) fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
