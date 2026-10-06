use anyhow::{Result, bail};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek},
    path::Path,
    time::Instant,
};
pub(crate) fn open(path: &Path, max: u64, credential: bool) -> Result<File> {
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
pub(crate) fn read(file: &mut File, max: usize) -> Result<Vec<u8>> {
    file.rewind()?;
    let mut bytes = Vec::new();
    file.take(max as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > max {
        bail!("local input byte bound exceeded");
    }
    file.rewind()?;
    Ok(bytes)
}
pub(super) fn hash(file: &mut File, expected: &str, size: u64, until: Instant) -> Result<()> {
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        bail!("local artifact SHA refused");
    }
    file.rewind()?;
    let mut total = 0u64;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        if Instant::now() >= until {
            bail!("local custody deadline expired");
        }
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        total = total
            .checked_add(n as u64)
            .ok_or_else(|| anyhow::anyhow!("local size overflow"))?;
        if total > size {
            bail!("local artifact grew");
        }
        hash.update(&buffer[..n]);
    }
    file.rewind()?;
    if total != size
        || hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
            != expected
    {
        bail!("local artifact identity mismatch");
    }
    Ok(())
}
