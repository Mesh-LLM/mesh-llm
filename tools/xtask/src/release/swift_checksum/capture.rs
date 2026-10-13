use super::error::Error;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

pub(super) struct NativeSwift {
    pub(super) executable: PathBuf,
    pub(super) cwd: PathBuf,
    pub(super) artifact: PathBuf,
    pub(super) timeout: Duration,
    pub(super) max_bytes: NonZeroUsize,
}

pub(super) fn compute(input: &NativeSwift, cancellation: &Cancellation) -> Result<String, Error> {
    let spec = ProcessSpec {
        executable: input.executable.clone(),
        arguments: vec![
            Value::Public("package".into()),
            Value::Public("compute-checksum".into()),
            Value::Public(input.artifact.as_os_str().to_owned()),
        ],
        cwd: input.cwd.clone(),
        environment: [
            "HOME",
            "PATH",
            "TMPDIR",
            "SDKROOT",
            "DEVELOPER_DIR",
            "TOOLCHAINS",
            "SYSTEMROOT",
            "WINDIR",
        ]
        .into_iter()
        .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
        .collect(),
    };
    let limits = Limits {
        execution: input.timeout,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        cancellation,
        RawCaptureOptions {
            stdout: Some(input.max_bytes),
            stderr: Some(input.max_bytes),
        },
    )?;
    if !report.process.success() || report.stdout.is_none() || report.stderr.is_none() {
        return Err(Error::Native(Box::new(report)));
    }
    let stdout = report.stdout.ok_or(Error::MissingPayload)?;
    let checksum = std::str::from_utf8(stdout.as_bytes())?;
    Ok(checksum.trim_end_matches('\n').to_owned())
}
