use super::Error;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

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
        let mut native = None;
        let mut state = std::env::temp_dir();
        let mut ready = Duration::from_secs(60);
        let mut shutdown = Duration::from_secs(15);
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
                "--native-runtime-root" => native = Some(PathBuf::from(value)),
                "--state-parent" => state = PathBuf::from(value),
                "--ready-max-wait" => ready = seconds(value)?,
                "--shutdown-max-wait" => shutdown = seconds(value)?,
                _ => return Err(Error::Invalid("unknown option")),
            }
        }
        let binary = canonical(&binary.ok_or(Error::Invalid("--binary is required"))?)?;
        let metadata = binary
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
        if binary
            .extension()
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("exe"))
        {
            return Err(Error::Invalid("Windows requires an explicit .exe"));
        }
        Ok(Self {
            binary,
            native_runtime_root: directory(
                &native.ok_or(Error::Invalid("--native-runtime-root is required"))?,
            )?,
            state_parent: directory(&state)?,
            ready_max_wait: ready,
            shutdown_max_wait: shutdown,
        })
    }
}

fn seconds(value: &str) -> Result<Duration, Error> {
    let invalid = || Error::Invalid("wait budgets require integer seconds in 1..=86400");
    if !value
        .as_bytes()
        .first()
        .is_some_and(|byte| (b'1'..=b'9').contains(byte))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid());
    }
    match value.parse::<u64>() {
        Ok(seconds @ 1..=86400) => Ok(Duration::from_secs(seconds)),
        Ok(_) | Err(_) => Err(invalid()),
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
