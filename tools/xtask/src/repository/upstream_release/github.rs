//! Bounded, read-only upstream GitHub API access through the installed gh CLI.
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

#[derive(Debug, PartialEq)]
pub(super) enum Request {
    Rest(String),
    ReleasePage(Option<String>),
}

impl Request {
    fn arguments(&self) -> Vec<Value> {
        let mut args = vec!["api".to_owned(), "--hostname".into(), "github.com".into()];
        match self {
            Self::Rest(endpoint) => {
                args.extend(["--method".into(), "GET".into(), endpoint.clone()])
            }
            Self::ReleasePage(cursor) => {
                // This fixed GraphQL query only reads release metadata; it cannot mutate GitHub.
                args.extend([
                    "--method".into(),
                    "POST".into(),
                    "graphql".into(),
                    "-f".into(),
                    format!("query={}", super::discovery::QUERY),
                ]);
                if let Some(cursor) = cursor {
                    args.extend(["-f".into(), format!("cursor={cursor}")]);
                }
            }
        }
        args.extend([
            "-H".into(),
            "Accept: application/vnd.github+json".into(),
            "-H".into(),
            "X-GitHub-Api-Version: 2022-11-28".into(),
        ]);
        args.into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect()
    }
}

pub(super) struct Client {
    deadline: Instant,
    cancellation: Cancellation,
}

impl Client {
    pub(super) fn new(cancellation: Cancellation) -> Self {
        Self {
            deadline: Instant::now() + Duration::from_secs(120),
            cancellation,
        }
    }

    pub(super) fn read(&mut self, request: &Request) -> DynResult<serde_json::Value> {
        let execution = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or("upstream release lookup exceeded its 120-second budget")?;
        if self.cancellation.is_cancelled() {
            return Err("upstream release lookup cancelled".into());
        }
        let executable = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
            .map(|path| path.join(if cfg!(windows) { "gh.exe" } else { "gh" }))
            .find(|path| path.is_file())
            .ok_or("upstream release lookup requires gh on PATH")?;
        let mut environment = BTreeMap::new();
        for key in [
            "PATH",
            "HOME",
            "SYSTEMROOT",
            "APPDATA",
            "GH_CONFIG_DIR",
            "GH_TOKEN",
            "GITHUB_TOKEN",
        ] {
            if let Some(value) = std::env::var_os(key) {
                let value = if matches!(key, "GH_TOKEN" | "GITHUB_TOKEN") && !value.is_empty() {
                    Value::Secret(value)
                } else {
                    Value::Public(value)
                };
                environment.insert(key.into(), value);
            }
        }
        let spec = ProcessSpec {
            executable: std::path::absolute(executable)?,
            arguments: request.arguments(),
            cwd: std::env::current_dir()?,
            environment,
        };
        let report = process::supervise_raw(
            &spec,
            &Limits {
                execution,
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 1024 * 1024,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &self.cancellation,
            RawCaptureOptions {
                stdout: NonZeroUsize::new(1024 * 1024),
                stderr: None,
            },
        )?;
        if report.process.outcome != process::Outcome::Exited
            || !report.process.success()
            || !report.process.cleanup.complete
            || report.process.failure.is_some()
            || report.process.stderr.truncated
        {
            return Err(format!(
                "upstream release API lookup failed; outcome={:?}, status={:?}, cleanup={}, failure={:?}, truncated={}/{}; no fallback selected",
                report.process.outcome, report.process.status, report.process.cleanup.complete,
                report.process.failure, report.process.stdout.truncated, report.process.stderr.truncated,
            )
            .into());
        }
        // API JSON can be one long line. The diagnostic stream may suppress
        // that line; only the independent bounded raw capture is our protocol.
        // Raw overflow/incomplete reads set process.failure and expose no bytes.
        let bytes = report
            .stdout
            .ok_or("upstream release response incomplete")?;
        Ok(serde_json::from_slice(bytes.as_bytes())?)
    }
}
