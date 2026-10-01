use crate::command::DynResult;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) fn read<T: DeserializeOwned>(path: &Path) -> DynResult<T> {
    if fs::metadata(path)?.len() > 16 * 1024 * 1024 {
        return Err("oversized evidence".into());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub(super) fn save(path: &Path, value: &impl Serialize) -> DynResult<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}

pub(super) fn environment(name: &str) -> DynResult<String> {
    Ok(std::env::var(name).map_err(|_| format!("missing environment: {name}"))?)
}

pub(super) fn empty(path: &Path) -> DynResult<bool> {
    if path.is_symlink() {
        return Ok(false);
    }
    Ok(!path.exists() || fs::read_dir(path)?.next().is_none())
}

pub(super) fn capture(program: &str, arguments: &[&str]) -> DynResult<Vec<u8>> {
    let executable = std::env::split_paths(&std::env::var_os("PATH").ok_or("missing PATH")?)
        .map(|directory| directory.join(program))
        .find(|path| path.is_file())
        .ok_or_else(|| format!("missing tool: {program}"))?
        .canonicalize()?;
    let environment: BTreeMap<_, _> = std::env::vars_os()
        .map(|(name, value)| {
            let secret = name.to_string_lossy().contains("TOKEN");
            (
                name,
                if secret {
                    Value::Secret(value)
                } else {
                    Value::Public(value)
                },
            )
        })
        .collect();
    let spec = ProcessSpec {
        executable,
        arguments: arguments
            .iter()
            .map(|value| {
                if value.starts_with("Authorization:") {
                    Value::Secret((*value).into())
                } else {
                    Value::Public((*value).into())
                }
            })
            .collect(),
        cwd: std::env::current_dir()?,
        environment,
    };
    let limits = Limits {
        execution: Duration::from_secs(30),
        graceful_shutdown: Duration::from_secs(2),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 0,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(16 * 1024 * 1024),
            stderr: None,
        },
    )?;
    if !report.process.success() {
        return Err(format!("{program}: bounded command failed").into());
    }
    Ok(report
        .stdout
        .ok_or("incomplete command output")?
        .as_bytes()
        .to_vec())
}

pub(super) fn files(directory: &Path, filename: Option<&str>) -> DynResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            paths.extend(files(&entry.path(), filename)?);
        } else if kind.is_file() && filename.is_none_or(|name| entry.file_name() == name) {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

pub(super) fn monotonic() -> DynResult<f64> {
    let text = fs::read_to_string("/proc/uptime")?;
    let seconds: f64 = text
        .split_whitespace()
        .next()
        .ok_or("missing uptime")?
        .parse()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err("invalid monotonic clock".into());
    }
    Ok(seconds)
}
