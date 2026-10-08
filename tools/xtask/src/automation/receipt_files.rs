//! Bounded reads and no-clobber writes for automation input and receipt files.
use crate::command::DynResult;
use std::{
    io::{Read, Write},
    path::Path,
};

pub(in crate::automation) fn bounded(path: &Path, maximum: u64) -> DynResult<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() {
        return Err("adaptive input/receipt must be a regular file".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("opened adaptive input/receipt must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err("adaptive input/receipt exceeds bound".into());
    }
    Ok(bytes)
}
pub(in crate::automation) fn fresh(path: &Path, bytes: &[u8]) -> DynResult<()> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("adaptive receipt output must be fresh".into()),
    }
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("output parent absent")?)?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)?;
    Ok(())
}
