use super::{Error, ErrorKind};
use std::{fs::File, io::Read, path::Path};

pub(super) const RECEIPT_LIMIT: u64 = 1024 * 1024;
pub(super) const RESULTS_LIMIT: u64 = 64 * 1024 * 1024;

pub(super) fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, Error> {
    if !std::fs::metadata(path)?.is_file() {
        return Err(Error::new(
            ErrorKind::Io,
            format!("not a regular file: {}", path.display()),
        ));
    }
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(Error::new(
            ErrorKind::Io,
            format!("not a regular file: {}", path.display()),
        ));
    }
    let mut bytes = Vec::new();
    file.take(limit.saturating_add(1)).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).map_or(true, |length| length > limit) {
        return Err(Error::new(
            ErrorKind::InputLimit,
            format!("input exceeds {limit} bytes: {}", path.display()),
        ));
    }
    Ok(bytes)
}
