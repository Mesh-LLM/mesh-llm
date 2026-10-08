use anyhow::{Result, bail};
use std::{
    fs::File,
    io::{Read, Seek},
    path::Path,
};
pub(super) fn open(path: &Path, max: u64, credential: bool) -> Result<File> {
    if !path.is_absolute() {
        bail!("local input path refused");
    }
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.len() > max {
        bail!("local regular input refused");
    }
    #[cfg(unix)]
    {
        use std::os::unix::{
            fs::{MetadataExt, OpenOptionsExt},
            prelude::PermissionsExt,
        };
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
            .open(path)?;
        let opened = file.metadata()?;
        if !opened.is_file() || opened.len() > max {
            bail!("opened regular input refused");
        }
        // SAFETY: geteuid has no arguments and only reports the current effective owner.
        if credential
            && (opened.uid() != unsafe { libc::geteuid() }
                || opened.permissions().mode() & 0o077 != 0)
        {
            bail!("private credential owner/mode refused");
        }
        Ok(file)
    }
    #[cfg(not(unix))]
    {
        let _ = credential;
        bail!("publisher safe FD admission requires Unix");
    }
}
pub(super) fn read(file: &mut File, max: usize) -> Result<Vec<u8>> {
    file.rewind()?;
    let mut bytes = Vec::new();
    file.take(max as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > max {
        bail!("local input byte bound exceeded");
    }
    file.rewind()?;
    Ok(bytes)
}
