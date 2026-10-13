//! Battery disk-headroom observations, independent of runtime qualification.
use crate::command::DynResult;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Serialize)]
struct Filesystem {
    label: &'static str,
    path: PathBuf,
    free_bytes: u64,
    minimum_free_bytes: u64,
    sufficient: bool,
}

#[derive(Serialize)]
struct Report {
    filesystems: Vec<Filesystem>,
    ports: Ports,
}

#[derive(Serialize)]
struct Ports {
    allocation: &'static str,
}

fn existing_ancestor(path: &Path) -> DynResult<PathBuf> {
    let mut current = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    loop {
        match fs::metadata(current) {
            Ok(_) => return Ok(current.to_owned()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                current = match current.parent() {
                    Some(parent) if !parent.as_os_str().is_empty() => parent,
                    Some(_) => Path::new("."),
                    None => return Err(error.into()),
                };
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn minimum_bytes(value: &str) -> DynResult<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("minimum free GiB must be an unsigned decimal integer".into());
    }
    value
        .parse::<u64>()?
        .checked_mul(1024 * 1024 * 1024)
        .ok_or_else(|| "minimum free GiB overflows byte count".into())
}

pub(super) fn run(artifact: &Path, models: &Path, minimum: &str, output: &Path) -> DynResult<()> {
    write_with(artifact, models, minimum, output, available_bytes)
}

fn write_with(
    artifact: &Path,
    models: &Path,
    minimum: &str,
    output: &Path,
    mut available: impl FnMut(&Path) -> DynResult<u64>,
) -> DynResult<()> {
    let minimum_free_bytes = minimum_bytes(minimum)?;
    let mut filesystems = Vec::new();
    for (label, requested) in [("artifacts", artifact), ("models", models)] {
        let path = existing_ancestor(requested)?;
        let free_bytes = available(&path)?;
        filesystems.push(Filesystem {
            label,
            path,
            free_bytes,
            minimum_free_bytes,
            sufficient: free_bytes >= minimum_free_bytes,
        });
    }
    let sufficient = filesystems.iter().all(|entry| entry.sufficient);
    let report = Report {
        filesystems,
        ports: Ports {
            allocation: "os-assigned-at-launch",
        },
    };
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    fs::write(output, bytes)?;
    if !sufficient {
        return Err("insufficient disk headroom; see environment receipt".into());
    }
    Ok(())
}

#[cfg(unix)]
fn available_bytes(path: &Path) -> DynResult<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes())?;
    let mut observation = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: path is NUL-terminated; observation points to writable statvfs storage.
    if unsafe { libc::statvfs(path.as_ptr(), observation.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: successful statvfs initialized the complete structure.
    let observation = unsafe { observation.assume_init() };
    let bytes = u128::from(observation.f_bavail)
        .checked_mul(u128::from(observation.f_frsize))
        .ok_or("filesystem available byte count overflow")?;
    Ok(u64::try_from(bytes)?)
}

#[cfg(not(unix))]
fn available_bytes(_path: &Path) -> DynResult<u64> {
    Err("family battery disk preflight requires a Unix filesystem".into())
}

#[cfg(test)]
#[path = "family_battery_environment_tests.rs"]
mod tests;
