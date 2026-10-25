//! Bounded regular JSON transport; domain owners retain schema and output semantics.
use super::process;
use crate::command::DynResult;
use std::{fs, io::Read, path::Path};
pub(super) const MAX_INPUT_BYTES: u64 = 65536;
fn same_file(before: &fs::Metadata, after: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return false;
        }
    }
    before.len() == after.len() && before.modified().ok() == after.modified().ok()
}
pub(super) fn read(path: &Path) -> DynResult<Vec<u8>> {
    process::check()?;
    let before = fs::symlink_metadata(path)?;
    if !before.is_file() || before.len() > MAX_INPUT_BYTES {
        return Err("canary package input must be a regular JSON file of at most 64 KiB".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    process::check()?;
    let mut file = options.open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file() || !same_file(&before, &opened) {
        return Err("canary package input changed during open".into());
    }
    let mut bytes = Vec::with_capacity(usize::try_from(before.len())?);
    let mut buffer = [0; 8192];
    loop {
        process::check()?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() as u64 > before.len() || bytes.len() as u64 > MAX_INPUT_BYTES {
            return Err("canary package input grew during reading".into());
        }
    }
    process::check()?;
    if bytes.len() as u64 != before.len() || !same_file(&opened, &file.metadata()?) {
        return Err("canary package input changed during reading".into());
    }
    Ok(bytes)
}
