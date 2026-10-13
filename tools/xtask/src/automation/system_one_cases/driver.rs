//! Explicit local case executables use the same argument contract under owned cleanup.
use super::Options;
use crate::{
    command::DynResult,
    command_interrupt::Interrupt,
    process::{self, Completion, Limits, ProcessSpec, Readiness, Value},
    repository::check_report::CheckReport,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};
fn admit(path: &Path) -> DynResult<PathBuf> {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("py"))
    {
        return Err("Python case drivers are retired; use an absolute native executable with the System One case arguments".into());
    }
    let metadata = std::fs::symlink_metadata(path)?;
    if !path.is_absolute() || !metadata.file_type().is_file() {
        return Err(
            "System One case driver must be an absolute regular executable, not a symlink".into(),
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err("System One case driver is not executable".into());
        }
    }
    Ok(path.canonicalize()?)
}
pub(super) fn forwarded(options: &Options) -> Vec<Value> {
    options
        .driver_arguments
        .iter()
        .cloned()
        .map(|value| Value::Public(value.into()))
        .collect()
}
pub(super) fn run(options: &Options) -> DynResult<()> {
    let executable = match admit(options.driver.as_ref().expect("explicit driver")) {
        Ok(path) => path,
        Err(error) => {
            return CheckReport {
                stdout: String::new(),
                stderr: format!("{error}\n"),
                code: 2,
            }
            .emit();
        }
    };
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let environment: BTreeMap<_, _> = ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"]
        .into_iter()
        .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
        .collect();
    let deadline = std::time::Instant::now() + options.driver_timeout;
    let result = process::supervise(
        &ProcessSpec {
            executable,
            cwd: std::env::current_dir()?,
            arguments: forwarded(options),
            environment,
        },
        &Limits {
            execution: options.driver_timeout,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &cancellation,
        process::OutputFiles::default(),
    );
    let finished = interrupt.finish();
    let report = result?;
    let clean = report.outcome == process::Outcome::Exited
        && report.failure.is_none()
        && report.cleanup.complete
        && !report.cleanup.forced
        && !report.cleanup.graceful_signal_failed
        && report.cleanup.failure.is_none()
        && report.stdout.line_capture_complete
        && report.stderr.line_capture_complete
        && !report.stdout.truncated
        && !report.stderr.truncated;
    let code = if clean
        && finished.is_ok()
        && !cancellation.is_cancelled()
        && std::time::Instant::now() < deadline
    {
        report.status.and_then(|s| s.code()).unwrap_or(2)
    } else {
        2
    };
    let mut stderr = String::from_utf8_lossy(&report.stderr.bytes_retained).into_owned();
    if code == 2 && !clean {
        stderr.push_str(
            "System One custom driver did not finish with complete owned capture and cleanup\n",
        );
    }
    CheckReport {
        stdout: String::from_utf8_lossy(&report.stdout.bytes_retained).into_owned(),
        stderr,
        code,
    }
    .emit()
}
