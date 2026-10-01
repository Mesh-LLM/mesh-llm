use super::Error;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/args.rs"]
mod tests;

pub(super) struct Options {
    pub(super) binary: PathBuf,
    pub(super) native_runtime_root: PathBuf,
    pub(super) state_parent: PathBuf,
    pub(super) ready_max_wait: Duration,
    pub(super) shutdown_max_wait: Duration,
}

impl Options {
    pub(super) fn parse(args: &[String]) -> Result<Self, Error> {
        let mut binary = None;
        let mut native_runtime_root = None;
        let mut state_parent = None;
        let mut ready = None;
        let mut shutdown = None;
        let mut seen = BTreeSet::new();
        let mut arguments = args.iter();
        while let Some(flag) = arguments.next() {
            if !seen.insert(flag) {
                return Err(Error::Invalid("duplicate option"));
            }
            let value = arguments
                .next()
                .filter(|value| !value.starts_with("--"))
                .ok_or(Error::Invalid("missing option value"))?;
            match flag.as_str() {
                "--binary" => binary = Some(PathBuf::from(value)),
                "--native-runtime-root" => native_runtime_root = Some(PathBuf::from(value)),
                "--state-parent" => state_parent = Some(PathBuf::from(value)),
                "--ready-max-wait" => ready = Some(seconds(value)?),
                "--shutdown-max-wait" => shutdown = Some(seconds(value)?),
                _ => return Err(Error::Invalid("unknown option")),
            }
        }
        let binary = executable(&binary.ok_or(Error::Invalid("--binary is required"))?)?;
        let native_runtime_root = directory(
            &native_runtime_root.ok_or(Error::Invalid("--native-runtime-root is required"))?,
        )?;
        let state_parent = directory(&state_parent.unwrap_or_else(|| {
            std::env::var_os("MESH_LLM_CLIENT_STATE_PARENT")
                .filter(|value| !value.is_empty())
                .map_or_else(std::env::temp_dir, PathBuf::from)
        }))?;
        Ok(Self {
            binary,
            native_runtime_root,
            state_parent,
            ready_max_wait: budget(ready, "MESH_LLM_CLIENT_READY_MAX_WAIT", 60)?,
            shutdown_max_wait: budget(shutdown, "MESH_LLM_CLIENT_SHUTDOWN_MAX_WAIT", 15)?,
        })
    }
}

fn seconds(value: &str) -> Result<Duration, Error> {
    if !value
        .as_bytes()
        .first()
        .is_some_and(|byte| (b'1'..=b'9').contains(byte))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(Error::Invalid(
            "wait budgets require integer seconds in 1..=86400",
        ));
    }
    match value.parse::<u64>() {
        Ok(seconds @ 1..=86400) => Ok(Duration::from_secs(seconds)),
        Ok(_) | Err(_) => Err(Error::Invalid(
            "wait budgets require integer seconds in 1..=86400",
        )),
    }
}

fn budget(explicit: Option<Duration>, key: &str, default: u64) -> Result<Duration, Error> {
    match explicit {
        Some(value) => Ok(value),
        None => match std::env::var_os(key).filter(|value| !value.is_empty()) {
            Some(value) => seconds(
                value
                    .to_str()
                    .ok_or(Error::Invalid("wait budget is not Unicode"))?,
            ),
            None => Ok(Duration::from_secs(default)),
        },
    }
}

fn canonical(path: &Path) -> Result<PathBuf, Error> {
    if !path.is_absolute() || path.as_os_str().as_encoded_bytes().contains(&0) {
        return Err(Error::Invalid("paths must be absolute and contain no NUL"));
    }
    path.canonicalize()
        .map_err(|error| Error::io("resolve input path", error))
}

fn directory(path: &Path) -> Result<PathBuf, Error> {
    let path = canonical(path)?;
    if !path.is_dir() {
        return Err(Error::Invalid("input directory is not a directory"));
    }
    Ok(path)
}

fn executable(path: &Path) -> Result<PathBuf, Error> {
    let path = canonical(path)?;
    let metadata = path
        .metadata()
        .map_err(|error| Error::io("inspect executable", error))?;
    if !metadata.is_file() {
        return Err(Error::Invalid("binary is not a regular executable file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(Error::Invalid("binary is not executable"));
        }
    }
    #[cfg(windows)]
    if path
        .extension()
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("exe"))
    {
        return Err(Error::Invalid("Windows requires an explicit .exe"));
    }
    Ok(path)
}
