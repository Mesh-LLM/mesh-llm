//! Bounded regular JSON transport; domain owners retain schema and output semantics.
use crate::command::DynResult;
use crate::process::Cancellation;
use std::{fs, io::Read, path::Path};
pub(super) const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;
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
pub(super) fn read(path: &Path, cancellation: &Cancellation) -> DynResult<Vec<u8>> {
    check(cancellation)?;
    let before = fs::symlink_metadata(path)?;
    if !before.is_file() || before.len() > MAX_INPUT_BYTES {
        return Err("canary timeout input must be a regular JSON file of at most 16 MiB".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    check(cancellation)?;
    let mut file = options.open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file() || !same_file(&before, &opened) {
        return Err("canary timeout input changed during open".into());
    }
    let mut bytes = Vec::with_capacity(usize::try_from(before.len())?);
    let mut buffer = [0; 8192];
    loop {
        check(cancellation)?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() as u64 > before.len() || bytes.len() as u64 > MAX_INPUT_BYTES {
            return Err("canary timeout input grew during reading".into());
        }
    }
    check(cancellation)?;
    if bytes.len() as u64 != before.len() || !same_file(&opened, &file.metadata()?) {
        return Err("canary timeout input changed during reading".into());
    }
    Ok(bytes)
}

fn check(cancellation: &Cancellation) -> DynResult<()> {
    if cancellation.is_cancelled() {
        return Err("canary timeout request admission cancelled".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_admission_stops_before_any_path_probe() {
        let root = tempfile::tempdir().unwrap();
        let token = Cancellation::default();
        token.cancel();
        let error = read(&root.path().join("absent-request"), &token).unwrap_err();
        assert!(error.to_string().contains("admission cancelled"));
    }
}
