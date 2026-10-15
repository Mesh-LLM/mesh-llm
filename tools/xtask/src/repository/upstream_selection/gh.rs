//! Bounded gh-api transport for the upstream selection walk: one read-only
//! GitHub request per call, with the credential redacted from retained streams
//! and a per-call ceiling inside the command's own budget.

use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

/// A single check-run page can carry the complete run list of one commit.
const JSON_LIMIT: usize = 16 * 1024 * 1024;
const STDERR_LIMIT: usize = 64 * 1024;
/// One request may not consume the whole command budget.
const REQUEST_SECONDS: u64 = 60;

/// One authenticated GET against the GitHub REST API.
pub(super) trait Gh {
    /// The response body, or the failure detail.
    fn api(&self, path: &str, query: &[(&str, String)]) -> Result<String, String>;
}

/// Reads public GitHub metadata with the runner's gh and its ambient
/// GH_TOKEN/GITHUB_TOKEN.
pub(super) struct GhTransaction {
    executable: PathBuf,
    cwd: PathBuf,
    deadline: Instant,
    cancellation: Cancellation,
}

impl GhTransaction {
    pub(super) fn new(budget: Duration, cancellation: Cancellation) -> Result<Self, String> {
        Ok(Self {
            executable: executable("gh")?,
            cwd: std::env::current_dir().map_err(|error| {
                format!("upstream selection working directory unavailable: {error}")
            })?,
            deadline: Instant::now() + budget,
            cancellation,
        })
    }
}

impl Gh for GhTransaction {
    fn api(&self, path: &str, query: &[(&str, String)]) -> Result<String, String> {
        if self.cancellation.is_cancelled() {
            return Err("upstream selection cancelled".into());
        }
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or("upstream selection exceeded its budget")?;
        let mut arguments = vec![
            Value::Public("api".into()),
            Value::Public("--method".into()),
            Value::Public("GET".into()),
            Value::Public(path.into()),
        ];
        for (key, value) in query {
            arguments.push(Value::Public("-f".into()));
            arguments.push(Value::Public(format!("{key}={value}").into()));
        }
        let spec = ProcessSpec {
            executable: self.executable.clone(),
            arguments,
            cwd: self.cwd.clone(),
            environment: environment(),
        };
        let limits = Limits {
            execution: remaining.min(Duration::from_secs(REQUEST_SECONDS)),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: STDERR_LIMIT,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = process::supervise_raw(
            &spec,
            &limits,
            &self.cancellation,
            RawCaptureOptions {
                stdout: NonZeroUsize::new(JSON_LIMIT),
                stderr: NonZeroUsize::new(STDERR_LIMIT),
            },
        )
        .map_err(|error| format!("gh api {path} could not start: {error}"))?;
        if !admitted(&report) {
            let detail = report
                .stderr
                .as_ref()
                .map(|bytes| String::from_utf8_lossy(bytes.as_bytes()).trim().to_owned())
                .filter(|detail| !detail.is_empty())
                .unwrap_or_else(|| "no stderr detail".to_owned());
            return Err(format!("gh api {path} failed: {detail} ({})", diagnostic(&report)));
        }
        let stdout = report
            .stdout
            .ok_or_else(|| format!("gh api {path} returned no response body"))?;
        String::from_utf8(stdout.as_bytes().to_vec())
            .map_err(|_| format!("gh api {path} returned a non-UTF-8 response body"))
    }
}

/// The supervision state of one request, for a diagnosable failure.
fn diagnostic(report: &process::RawProcessReport) -> String {
    let stream = |name: &str, side: &process::StreamReport| {
        format!(
            "{name} {} bytes{}{}",
            side.bytes_seen,
            if side.truncated { " truncated" } else { "" },
            if side.line_capture_complete { "" } else { " incomplete" }
        )
    };
    format!(
        "outcome {:?}, status {:?}, {}, {}, cleanup complete {} forced {}, failure {:?}",
        report.process.outcome,
        report.process.status,
        stream("stdout", &report.process.stdout),
        stream("stderr", &report.process.stderr),
        report.process.cleanup.complete,
        report.process.cleanup.forced,
        report.process.failure,
    )
}

/// A request is admitted only when gh exited successfully and its response body
/// was captured whole. The per-line counters are diagnostic retention, not
/// transport health: one GitHub JSON body is a single line far past the line
/// limit, so an oversized line is expected and the raw capture is what matters.
fn admitted(report: &process::RawProcessReport) -> bool {
    report.process.outcome == process::Outcome::Exited
        && report.process.success()
        && report.process.failure.is_none()
        && report.stdout.is_some()
}

/// Only the variables gh needs, with the credential marked secret so it is
/// redacted from anything retained.
fn environment() -> BTreeMap<std::ffi::OsString, Value> {
    let mut environment = BTreeMap::from([
        ("GH_NO_UPDATE_NOTIFIER".into(), Value::Public("1".into())),
        ("GH_PAGER".into(), Value::Public("cat".into())),
    ]);
    for key in ["PATH", "HOME", "SYSTEMROOT"] {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(key.into(), Value::Public(value));
        }
    }
    for key in ["GH_TOKEN", "GITHUB_TOKEN"] {
        if let Some(value) = std::env::var_os(key).filter(|value| !value.is_empty()) {
            environment.insert(key.into(), Value::Secret(value));
        }
    }
    environment
}

fn executable(tool: &str) -> Result<PathBuf, String> {
    let path = std::env::var_os("PATH").ok_or(format!("{tool} PATH missing"))?;
    let candidate = std::env::split_paths(&path)
        .map(|directory| directory.join(tool))
        .find(|candidate| candidate.is_file())
        .ok_or(format!("{tool} executable missing"))?;
    std::path::absolute(Path::new(&candidate))
        .map_err(|error| format!("{tool} executable unavailable: {error}"))
}
