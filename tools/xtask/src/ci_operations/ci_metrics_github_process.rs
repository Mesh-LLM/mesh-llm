//! Per-invocation bounded transport for the existing live metrics Gh owner.
use super::ci_metrics_github::{GhOutput, text_mode};
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

// A single response can carry run lists or a page of 100 jobs. These generous
// limits match saved metrics admission and avoid truncating legitimate JSON.
const JSON_LIMIT: usize = 64 * 1024 * 1024;
const STDERR_LIMIT: usize = 1024 * 1024;
type Environment = BTreeMap<OsString, OsString>;

pub(super) fn run(arguments: &[String], cancellation: &Cancellation) -> io::Result<GhOutput> {
    let cwd = std::env::current_dir()?;
    let environment = std::env::vars_os().collect();
    run_in(
        arguments,
        cancellation,
        &cwd,
        &environment,
        &limits(Duration::from_secs(120)),
        JSON_LIMIT,
    )
}

fn limits(execution: Duration) -> Limits {
    Limits {
        execution,
        graceful_shutdown: Duration::from_secs(2),
        forced_shutdown: Duration::from_secs(5),
        retained_bytes_per_stream: STDERR_LIMIT,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

fn run_in(
    arguments: &[String],
    cancellation: &Cancellation,
    cwd: &Path,
    environment: &Environment,
    budget: &Limits,
    json_limit: usize,
) -> io::Result<GhOutput> {
    if cancellation.is_cancelled() {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "gh collection cancelled",
        ));
    }
    let cwd = cwd.canonicalize()?;
    let spec = ProcessSpec {
        executable: executable(&cwd, environment.get(&OsString::from("PATH")).cloned())?,
        arguments: arguments
            .iter()
            .map(|arg| Value::Public(arg.clone().into()))
            .collect(),
        cwd,
        environment: environment
            .iter()
            .map(|(key, value)| {
                // Only gh's authentication variables need process redaction.
                // Unrelated inherited multiline keys retain ordinary gh behavior.
                let secret = !value.is_empty()
                    && matches!(
                        key.to_str(),
                        Some(
                            "GH_TOKEN"
                                | "GITHUB_TOKEN"
                                | "GH_ENTERPRISE_TOKEN"
                                | "GITHUB_ENTERPRISE_TOKEN"
                        )
                    );
                (
                    key.clone(),
                    if secret {
                        Value::Secret(value.clone())
                    } else {
                        Value::Public(value.clone())
                    },
                )
            })
            .collect(),
    };
    let report = process::supervise_raw(
        &spec,
        budget,
        cancellation,
        RawCaptureOptions {
            stdout: NonZeroUsize::new(json_limit),
            stderr: NonZeroUsize::new(STDERR_LIMIT),
        },
    )
    .map_err(io::Error::other)?;
    if !report.process.cleanup.complete || report.process.failure.is_some() {
        return Err(io::Error::other(format!(
            "gh supervision failed: {:?}",
            report.process.failure
        )));
    }
    if cancellation.is_cancelled() || report.process.outcome == process::Outcome::Cancelled {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "gh collection cancelled",
        ));
    }
    if report.process.outcome != process::Outcome::Exited {
        return Err(io::Error::other(format!(
            "gh collection stopped: {:?}",
            report.process.outcome
        )));
    }
    let stdout = report
        .stdout
        .ok_or_else(|| io::Error::other("gh stdout incomplete"))?;
    let stderr = report
        .stderr
        .ok_or_else(|| io::Error::other("gh stderr incomplete"))?;
    let success = report.process.success();
    // Successful API JSON stays lossless. On failing status, gh_json renders
    // diagnostics, so use the owner's retained credential-redacted streams.
    Ok(GhOutput {
        success,
        stdout: text_mode(if success {
            stdout.as_bytes()
        } else {
            &report.process.stdout.bytes_retained
        }),
        stderr: text_mode(if success {
            stderr.as_bytes()
        } else {
            &report.process.stderr.bytes_retained
        }),
    })
}

fn executable(cwd: &Path, path: Option<OsString>) -> io::Result<PathBuf> {
    #[cfg(unix)]
    let search = path.unwrap_or_else(|| "/usr/bin:/bin".into());
    #[cfg(windows)]
    let search = path.unwrap_or_default();
    #[cfg(unix)]
    let name = "gh";
    #[cfg(windows)]
    let name = "gh.exe";
    for directory in std::env::split_paths(&search) {
        let candidate = if directory.is_absolute() {
            directory.join(name)
        } else {
            cwd.join(directory).join(name)
        };
        let Ok(metadata) = candidate.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o111 == 0 {
                continue;
            }
        }
        // Keep PATH-selected spelling; do not canonicalize a selected tool symlink.
        return Ok(candidate);
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "gh executable unavailable on PATH",
    ))
}

#[cfg(all(test, unix))]
#[path = "ci_metrics_github_process_tests.rs"]
mod tests;
