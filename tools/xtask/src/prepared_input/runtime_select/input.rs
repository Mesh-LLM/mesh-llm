//! Bounded regular-file admission for adjacent runtime reports and manifests.
use super::super::Checked;
use std::{fs, io::Read, path::Path};

const LIMIT: u64 = 64 * 1024 * 1024;

fn unchanged(before: &fs::Metadata, after: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return false;
        }
    }
    before.len() == after.len() && before.modified().ok() == after.modified().ok()
}

pub(super) fn read(path: &Path) -> Checked<Vec<u8>> {
    let before = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !before.is_file() || before.len() > LIMIT {
        return Err("runtime JSON input must be a regular file of at most 64 MiB".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Open a replacement reparse point itself, rather than following it.
        options.custom_flags(0x0020_0000);
    }
    let mut file = options.open(path).map_err(|error| error.to_string())?;
    let opened = file.metadata().map_err(|error| error.to_string())?;
    if !opened.is_file() || !unchanged(&before, &opened) {
        return Err("runtime JSON input changed during open".into());
    }
    let mut bytes = Vec::new();
    let mut chunk = [0; 8192];
    loop {
        let count = file.read(&mut chunk).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if bytes.len() as u64 > before.len() || bytes.len() as u64 > LIMIT {
            return Err("runtime JSON input grew during reading".into());
        }
    }
    let after = file.metadata().map_err(|error| error.to_string())?;
    let named = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if bytes.len() as u64 != before.len()
        || !unchanged(&opened, &after)
        || !named.is_file()
        || !unchanged(&opened, &named)
    {
        return Err("runtime JSON input changed during reading".into());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_input_is_read_and_missing_directory_and_oversize_inputs_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("report.json");
        fs::write(&path, b"[]").unwrap();
        assert_eq!(
            read(&path).unwrap_or_else(|error| panic!("{}", error.0)),
            b"[]"
        );
        assert!(read(&root.path().join("missing")).is_err());
        assert!(read(root.path()).is_err());
        fs::File::create(&path).unwrap().set_len(LIMIT + 1).unwrap();
        assert!(read(&path).unwrap_err().0.contains("at most 64 MiB"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_and_fifos_are_rejected_without_waiting_for_a_writer() {
        use std::{
            ffi::CString,
            os::unix::{ffi::OsStrExt, fs::symlink},
        };
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("report.json");
        fs::write(&path, b"[]").unwrap();
        let link = root.path().join("link");
        symlink(&path, &link).unwrap();
        assert!(read(&link).is_err());
        let fifo = root.path().join("fifo");
        let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        // SAFETY: name is a NUL-terminated path retained for this synchronous call.
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        assert!(read(&fifo).is_err());
    }
}
