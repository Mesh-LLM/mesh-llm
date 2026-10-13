//! Owned local Git and read-only GitHub evidence transport; no fetch or publication.
use super::{
    github::Gh,
    provenance::{Error, Git, Output, Result},
};
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
type Environment = BTreeMap<OsString, OsString>;
const EVIDENCE_LIMIT: usize = 128 * 1024 * 1024;
const STDERR_LIMIT: usize = 1024 * 1024;
pub(crate) struct Transport {
    cwd: PathBuf,
    environment: Environment,
    cancellation: Cancellation,
    deadline: Instant,
    stdout_limit: usize,
}
impl Transport {
    /// Explicit operator context is shared by every command in one evidence collection.
    // Finite integration projections own relative-budget and capture-limit qualification.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn new(
        cwd: &Path,
        environment: Environment,
        cancellation: Cancellation,
        budget: Duration,
    ) -> Result<Self> {
        let deadline = Instant::now()
            .checked_add(budget)
            .ok_or_else(|| Error("release inventory budget out of range".into()))?;
        Self::new_until(cwd, environment, cancellation, deadline)
    }
    pub(crate) fn new_until(
        cwd: &Path,
        environment: Environment,
        cancellation: Cancellation,
        deadline: Instant,
    ) -> Result<Self> {
        let cwd = cwd
            .canonicalize()
            .map_err(|_| Error("release inventory working directory unavailable".into()))?;
        Ok(Self {
            cwd,
            environment,
            cancellation,
            deadline,
            stdout_limit: EVIDENCE_LIMIT,
        })
    }
    #[cfg(all(test, unix))]
    #[allow(dead_code)]
    pub(crate) fn capture_limit(&mut self, bytes: usize) {
        self.stdout_limit = bytes;
    }
    fn run(&self, tool: &str, args: &[&str]) -> Result<Output> {
        if self.cancellation.is_cancelled() {
            return Err(Error("release inventory cancelled".into()));
        }
        let execution = self
            .deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| Error("release inventory deadline expired".into()))?;
        let spec = ProcessSpec {
            executable: executable(
                tool,
                &self.cwd,
                self.environment.get(&OsString::from("PATH")),
            )?,
            cwd: self.cwd.clone(),
            arguments: args.iter().map(|v| Value::Public((*v).into())).collect(),
            environment: self
                .environment
                .iter()
                .map(|(key, value)| {
                    let secret = matches!(
                        key.to_str(),
                        Some(
                            "GH_TOKEN"
                                | "GITHUB_TOKEN"
                                | "GH_ENTERPRISE_TOKEN"
                                | "GITHUB_ENTERPRISE_TOKEN"
                        )
                    ) && !value.is_empty();
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
            &Limits {
                execution,
                graceful_shutdown: Duration::from_secs(2),
                forced_shutdown: Duration::from_secs(5),
                retained_bytes_per_stream: STDERR_LIMIT,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &self.cancellation,
            RawCaptureOptions {
                stdout: NonZeroUsize::new(self.stdout_limit),
                stderr: NonZeroUsize::new(STDERR_LIMIT),
            },
        )
        .map_err(|failure| {
            Error(format!(
                "release inventory process admission failed: {failure}"
            ))
        })?;
        if !report.process.cleanup.complete {
            return Err(Error(
                "release inventory owned process cleanup incomplete".into(),
            ));
        }
        if self.cancellation.is_cancelled() || report.process.outcome == process::Outcome::Cancelled
        {
            return Err(Error("release inventory cancelled".into()));
        }
        if report.process.outcome == process::Outcome::Deadline {
            return Err(Error("release inventory process deadline exceeded".into()));
        }
        if let Some(failure) = report.process.failure {
            // Failure contains only supervisor operation/status metadata, never child output or argv.
            return Err(Error(format!(
                "release inventory supervision failed: {failure}"
            )));
        }
        if report.process.outcome != process::Outcome::Exited {
            return Err(Error("release inventory process lifecycle failed".into()));
        }
        let code = report
            .process
            .status
            .and_then(|s| s.code())
            .ok_or_else(|| Error("release inventory process has no native exit status".into()))?;
        // Never echo command output on failure: remote URLs and gh diagnostics can carry credentials.
        let stdout = report
            .stdout
            .ok_or_else(|| Error("release inventory output incomplete".into()))?
            .as_bytes()
            .to_vec();
        Ok(Output { code, stdout })
    }
}
impl Git for Transport {
    fn read(&mut self, args: &[&str]) -> Result<Output> {
        self.run("git", args)
    }
}
impl Gh for Transport {
    fn request(&mut self, args: &[&str]) -> Result<Output> {
        self.run("gh", args)
    }
}
fn executable(tool: &str, cwd: &Path, path: Option<&OsString>) -> Result<PathBuf> {
    #[cfg(unix)]
    let search = path.cloned().unwrap_or_else(|| "/usr/bin:/bin".into());
    #[cfg(windows)]
    let search = path.cloned().unwrap_or_default();
    #[cfg(unix)]
    let name = tool.to_owned();
    #[cfg(windows)]
    let name = format!("{tool}.exe");
    for directory in std::env::split_paths(&search) {
        let candidate = if directory.is_absolute() {
            directory.join(&name)
        } else {
            cwd.join(directory).join(&name)
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
        return Ok(candidate);
    }
    Err(Error(
        "release inventory native tool unavailable on operator PATH".into(),
    ))
}
