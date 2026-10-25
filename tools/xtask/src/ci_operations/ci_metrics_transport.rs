//! Saved metrics document transport. JSON/report semantics belong to their owners.
use super::ci_metrics_normalize::{Failure, Outcome};
use crate::process::Cancellation;
use std::{
    fs::{self, File, Metadata, OpenOptions},
    io::Read,
    path::Path,
};

pub(super) const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;

pub(super) fn check(cancellation: &Cancellation) -> Outcome<()> {
    if cancellation.is_cancelled() {
        Err(Failure::Reported("metrics input cancelled".into()))
    } else {
        Ok(())
    }
}

fn refusal(message: &str) -> Failure {
    Failure::Reported(message.into())
}
fn io(error: std::io::Error, path: &Path) -> Failure {
    // Retain existing OS error formatting; transport refusals use native messages.
    Failure::Reported(super::build_cache_tree::io_text(&error, path))
}

fn regular(metadata: &Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes()
            & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
            != 0
        {
            return false;
        }
    }
    metadata.is_file()
}

fn same_file(before: &Metadata, after: &Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return false;
        }
    }
    before.len() == after.len() && before.modified().ok() == after.modified().ok()
}

pub(super) fn read_file(path: &Path, cancellation: &Cancellation) -> Outcome<Vec<u8>> {
    check(cancellation)?;
    let before = fs::symlink_metadata(path).map_err(|error| io(error, path))?;
    if !regular(&before) || before.len() > MAX_INPUT_BYTES {
        return Err(refusal(
            "metrics input must be a regular file of at most 64 MiB",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Open a reparse point itself, never its potentially blocking target.
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    check(cancellation)?;
    let mut file = options.open(path).map_err(|error| io(error, path))?;
    let opened = file.metadata().map_err(|error| io(error, path))?;
    if !regular(&opened) || !same_file(&before, &opened) {
        return Err(refusal("metrics input changed during open"));
    }
    read_opened(&mut file, path, &opened, cancellation)
}

fn read_opened(
    file: &mut File,
    path: &Path,
    opened: &Metadata,
    cancellation: &Cancellation,
) -> Outcome<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        check(cancellation)?;
        let count = file.read(&mut buffer).map_err(|error| io(error, path))?;
        if count == 0 {
            break;
        }
        if bytes.len() as u64 + count as u64 > opened.len()
            || bytes.len() as u64 + count as u64 > MAX_INPUT_BYTES
        {
            return Err(refusal("metrics input grew during reading"));
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    check(cancellation)?;
    if bytes.len() as u64 != opened.len()
        || !same_file(opened, &file.metadata().map_err(|error| io(error, path))?)
    {
        return Err(refusal("metrics input changed during reading"));
    }
    Ok(bytes)
}

#[cfg(test)]
#[path = "ci_metrics_transport_tests.rs"]
mod tests;
