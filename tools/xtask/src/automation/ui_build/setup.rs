//! Native package-tool launchers and explicit child environment admission.
use super::{execution::Tool, options::Options};
use crate::command::DynResult;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

pub(super) fn prepare(options: &Options) -> DynResult<Tool> {
    let (executable, prefix) = if let Some(command) = &options.executable {
        let executable = admit_executable(command)?;
        let prefix = options
            .pnpm_script
            .as_ref()
            .map(|path| -> DynResult<Vec<OsString>> {
                let path = path.canonicalize()?;
                if !fs::metadata(&path)?.is_file() {
                    return Err("pnpm JavaScript entrypoint must be a regular file".into());
                }
                Ok(vec![path.into_os_string()])
            })
            .transpose()?
            .unwrap_or_default();
        (executable, prefix)
    } else {
        discover()?
    };
    Ok(Tool {
        executable,
        prefix,
        environment: package_environment(),
    })
}

fn admit_executable(path: &Path) -> DynResult<PathBuf> {
    let path = path.canonicalize()?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_file() {
        return Err("UI package launcher must be a regular executable file".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err("UI package launcher is not executable".into());
        }
    }
    #[cfg(windows)]
    if path
        .extension()
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("exe"))
    {
        return Err("Windows UI launcher must be an explicit .exe; use Node with --pnpm-script for JavaScript launchers".into());
    }
    Ok(path)
}

fn find(name: &str) -> DynResult<PathBuf> {
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let path = directory.join(name);
        if path.is_file() {
            return admit_executable(&path);
        }
    }
    Err(format!("UI package executable is missing from PATH: {name}").into())
}

#[cfg(unix)]
fn discover() -> DynResult<(PathBuf, Vec<OsString>)> {
    Ok((find("pnpm")?, vec![]))
}

#[cfg(windows)]
fn discover() -> DynResult<(PathBuf, Vec<OsString>)> {
    if let Ok(executable) = find("pnpm.exe") {
        return Ok((executable, vec![]));
    }
    let node = find("node.exe")?;
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        if !directory.join("pnpm.cmd").is_file() {
            continue;
        }
        // Known npm-global/Corepack entrypoint layouts; never parse or execute
        // a command shim as a shell string. Unknown layouts require explicit paths.
        for relative in [
            "node_modules/pnpm/bin/pnpm.cjs",
            "node_modules/corepack/dist/pnpm.js",
        ] {
            let script = directory.join(relative);
            if script.is_file() {
                return Ok((node, vec![script.canonicalize()?.into_os_string()]));
            }
        }
    }
    Err("Windows pnpm requires pnpm.exe or a native Node launcher with --pnpm-script".into())
}

fn package_environment() -> BTreeMap<OsString, (OsString, bool)> {
    std::env::vars_os()
        .filter_map(|(name, value)| {
            let label = name.to_str()?.to_ascii_uppercase();
            let admitted = matches!(
                label.as_str(),
                "PATH"
                    | "HOME"
                    | "USERPROFILE"
                    | "APPDATA"
                    | "LOCALAPPDATA"
                    | "SYSTEMROOT"
                    | "COMSPEC"
                    | "TEMP"
                    | "TMP"
                    | "TMPDIR"
                    | "CI"
                    | "XDG_CACHE_HOME"
                    | "HTTP_PROXY"
                    | "HTTPS_PROXY"
                    | "ALL_PROXY"
                    | "NO_PROXY"
                    | "SSL_CERT_FILE"
                    | "SSL_CERT_DIR"
                    | "NPM_TOKEN"
            ) || label.starts_with("NPM_CONFIG_")
                || label.starts_with("NODE_")
                || label.starts_with("PNPM_")
                || label.starts_with("COREPACK_");
            if !admitted {
                return None;
            }
            let sensitive = !value.is_empty()
                && (label.contains("TOKEN")
                    || label.contains("PASSWORD")
                    || label.contains("_AUTH")
                    || matches!(label.as_str(), "HTTP_PROXY" | "HTTPS_PROXY" | "ALL_PROXY"));
            Some((name, (value, sensitive)))
        })
        .collect()
}
