use super::{Error, Mode, NativeLipo};
use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) const USAGE: &str = "release swift-xcframework <xcframework> [host-only|full] [--lipo <absolute-executable>] [--timeout-seconds <1..3600>] [--max-output-bytes <1..1048576>]";

pub(super) struct Options {
    pub(super) root: String,
    pub(super) mode: Option<Mode>,
    executable: Option<PathBuf>,
    timeout: Duration,
    max_bytes: NonZeroUsize,
}

pub(super) fn parse(args: &[String]) -> Result<Options, String> {
    let (root, mut remaining) = args
        .split_first()
        .ok_or_else(|| "missing xcframework".to_owned())?;
    let mut mode = None;
    if let Some(value) = remaining.first()
        && !value.starts_with("--")
    {
        mode = Some(match value.as_str() {
            "host-only" => Mode::HostOnly,
            "full" => Mode::Full,
            _ => return Err(format!("invalid mode: {value}")),
        });
        remaining = &remaining[1..];
    }
    let mut options = Options {
        root: root.clone(),
        mode,
        executable: None,
        timeout: Duration::from_secs(30),
        max_bytes: NonZeroUsize::new(1048576)
            .ok_or_else(|| "invalid default output bound".to_owned())?,
    };
    for pair in remaining.chunks(2) {
        let [flag, value] = pair else {
            return Err("missing option value".into());
        };
        match flag.as_str() {
            "--lipo" => {
                let path = Path::new(value);
                if !path.is_absolute() {
                    return Err("--lipo requires an absolute executable".into());
                }
                options.executable = Some(path.to_owned());
            }
            "--timeout-seconds" => options.timeout = Duration::from_secs(bounded(value, 3600)?),
            "--max-output-bytes" => {
                options.max_bytes = usize::try_from(bounded(value, 1048576)?)
                    .ok()
                    .and_then(NonZeroUsize::new)
                    .ok_or_else(|| "invalid output bound".to_owned())?
            }
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    Ok(options)
}

fn bounded(value: &str, maximum: u64) -> Result<u64, String> {
    value
        .parse::<u64>()
        .ok()
        .filter(|number| (1..=maximum).contains(number))
        .ok_or_else(|| format!("expected integer in 1..={maximum}: {value}"))
}

impl Options {
    pub(super) fn native(&self) -> Result<NativeLipo, Error> {
        let cwd = std::env::current_dir().map_err(|error| Error::io(Path::new("."), error))?;
        let executable = match &self.executable {
            Some(path) => path.clone(),
            None => {
                let name = std::env::var_os("LIPO").unwrap_or_else(|| "lipo".into());
                let path = Path::new(&name);
                if path.components().count() > 1 || path.is_absolute() {
                    std::path::absolute(path).map_err(|error| Error::io(path, error))?
                } else {
                    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
                        .map(|directory| cwd.join(directory).join(&name)).find(|candidate| executable(candidate))
                        .ok_or_else(|| Error::Contract(format!("failed to inspect XCFramework binary with lipo: executable {name:?} not found")))?
                }
            }
        };
        Ok(NativeLipo {
            executable,
            cwd,
            environment: std::env::vars_os().collect(),
            timeout: self.timeout,
            max_bytes: self.max_bytes,
        })
    }
}

fn executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
    }
    #[cfg(windows)]
    {
        path.is_file()
    }
}
