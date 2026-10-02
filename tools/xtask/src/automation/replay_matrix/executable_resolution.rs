//! Resolve replay process executables while preserving symlink spelling.
use std::path::{Path, PathBuf};

pub(super) fn tool(name: &str) -> crate::command::DynResult<PathBuf> {
    let search = std::env::var_os("PATH").ok_or("PATH is unavailable")?;
    for directory in std::env::split_paths(&search) {
        #[cfg(windows)]
        let path = directory.join(format!("{name}.exe"));
        #[cfg(not(windows))]
        let path = directory.join(name);
        if let Ok(path) = executable(&std::path::absolute(path)?) {
            return Ok(path);
        }
    }
    Err(format!("required replay tool {name} is unavailable on PATH").into())
}

pub(super) fn executable(path: &Path) -> Result<PathBuf, &'static str> {
    if !path.is_absolute() {
        return Err("--python must be an absolute executable path");
    }
    let path = path.to_path_buf();
    path.metadata()
        .map_err(|_| "--python must name an existing absolute executable")?;
    if !path.is_file() {
        return Err("--python must name an existing absolute executable");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if path
            .metadata()
            .map_err(|_| "cannot read executable metadata")?
            .permissions()
            .mode()
            & 0o111
            == 0
        {
            return Err("--python must be an executable file");
        }
    }
    #[cfg(windows)]
    if path
        .extension()
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("exe"))
    {
        return Err("--python must be an explicit .exe executable");
    }
    Ok(path)
}
