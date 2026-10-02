//! Bounded regular artifact reads for TTS native stamps and complete waveforms.
use crate::command::DynResult;
use std::{fs, io::Read, path::Path};
pub(super) fn read(path: &Path, maximum: u64, label: &str) -> DynResult<Vec<u8>> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(format!("{label} must be a regular file").into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        // Avoid blocking on a FIFO swapped after admission; reject swapped symlinks.
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.file_type().is_file() {
        return Err(format!("opened {label} must be a regular file").into());
    }
    let mut bytes = Vec::new();
    file.take(maximum.checked_add(1).ok_or("TTS input bound overflow")?)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > maximum {
        return Err(format!("{label} exceeds byte bound {maximum}").into());
    }
    Ok(bytes)
}
