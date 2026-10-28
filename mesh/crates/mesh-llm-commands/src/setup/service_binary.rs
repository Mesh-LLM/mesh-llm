//! The mesh-llm binary path a background service starts.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// The path mesh-llm was started by, else `current_exe`.
///
/// `current_exe` resolves a package manager's link or wrapper to the
/// versioned binary behind it, which an upgrade removes. A bare name is
/// looked up in `PATH`, a relative path joined to `cwd`, neither resolved.
pub(crate) fn service_binary_path(
    invoked: Option<&OsStr>,
    path_var: Option<&OsStr>,
    cwd: Option<&Path>,
    current_exe: PathBuf,
) -> PathBuf {
    invoked_binary_path(invoked, path_var, cwd).unwrap_or(current_exe)
}

/// The working directory as the shell names it: `$PWD` when it is absolute
/// and the same directory as `current_dir`, which resolves links.
pub(crate) fn working_directory(
    pwd: Option<&OsStr>,
    current_dir: Option<PathBuf>,
) -> Option<PathBuf> {
    let current_dir = current_dir?;
    let logical = pwd
        .map(PathBuf::from)
        .filter(|pwd| pwd.is_absolute() && same_directory(pwd, &current_dir));
    Some(logical.unwrap_or(current_dir))
}

#[cfg(unix)]
fn same_directory(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn same_directory(_a: &Path, _b: &Path) -> bool {
    false
}

#[cfg(unix)]
fn invoked_binary_path(
    invoked: Option<&OsStr>,
    path_var: Option<&OsStr>,
    cwd: Option<&Path>,
) -> Option<PathBuf> {
    let invoked = Path::new(invoked?);
    if invoked.components().count() > 1 {
        let path = if invoked.is_absolute() {
            invoked.to_path_buf()
        } else {
            cwd?.join(invoked)
        };
        return is_executable_file(&path).then_some(path);
    }
    std::env::split_paths(path_var?)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join(invoked))
        .find(|candidate| is_executable_file(candidate))
}

#[cfg(not(unix))]
fn invoked_binary_path(
    _invoked: Option<&OsStr>,
    _path_var: Option<&OsStr>,
    _cwd: Option<&Path>,
) -> Option<PathBuf> {
    None
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
}
