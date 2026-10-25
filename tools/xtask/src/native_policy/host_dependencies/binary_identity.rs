//! Opt-in identity of the actual regular host binary, streamed without following a leaf link.
use sha2::{Digest, Sha256};
use std::{fs::OpenOptions, io::Read, path::Path};

const MAX_BINARY_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub(super) fn digest(path: &Path) -> Result<String, String> {
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
        options.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let mut file = options
        .open(path)
        .map_err(|error| format!("cannot bind host binary identity: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("cannot inspect host binary identity: {error}"))?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err("host binary identity cannot follow a reparse point".into());
        }
    }
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_BINARY_BYTES {
        return Err("host binary identity requires a nonempty regular file at most 2 GiB".into());
    }
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot read host binary identity: {error}"))?;
        if count == 0 {
            break;
        }
        bytes += count as u64;
        if bytes > MAX_BINARY_BYTES {
            return Err("host binary identity exceeds 2 GiB".into());
        }
        hasher.update(&buffer[..count]);
    }
    if bytes != metadata.len() || file.metadata().map_err(|e| e.to_string())?.len() != bytes {
        return Err("host binary changed while reading its identity".into());
    }
    Ok(hex::encode(hasher.finalize()))
}
