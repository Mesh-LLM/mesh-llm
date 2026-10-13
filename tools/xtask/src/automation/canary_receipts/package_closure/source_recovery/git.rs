//! Fixed credential-free Git probes under one recovery deadline.
use super::super::process;
use crate::{
    command::DynResult,
    process::{
        self as supervisor, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};

pub(super) struct Git {
    deadline: Instant,
}

impl Git {
    pub(super) fn new() -> Self {
        Self {
            deadline: Instant::now() + Duration::from_secs(120),
        }
    }
    pub(super) fn remaining(&self) -> DynResult<Duration> {
        process::check()?;
        self.deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| "recovery exceeded its 120-second deadline".into())
    }
    pub(super) fn run(&self, root: &Path, args: &[OsString]) -> DynResult<Vec<u8>> {
        let mut arguments: Vec<OsString> = [
            "--no-optional-locks",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
        ]
        .map(Into::into)
        .into();
        arguments.extend_from_slice(args);
        let environment: BTreeMap<_, _> = [
            ("PATH", "/usr/bin:/bin"),
            ("LC_ALL", "C"),
            ("GIT_MASTER", "1"),
            ("GIT_CONFIG_NOSYSTEM", "1"),
            ("GIT_CONFIG_GLOBAL", "/dev/null"),
            ("GIT_TERMINAL_PROMPT", "0"),
            ("GIT_ALLOW_PROTOCOL", "file"),
            ("GIT_NO_LAZY_FETCH", "1"),
            ("GIT_NO_REPLACE_OBJECTS", "1"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        let report = supervisor::supervise_raw(
            &ProcessSpec {
                executable: "/usr/bin/git".into(),
                cwd: root.to_path_buf(),
                environment,
                arguments: arguments.into_iter().map(Value::Public).collect(),
            },
            &Limits {
                execution: self.remaining()?.min(Duration::from_secs(60)),
                graceful_shutdown: Duration::from_secs(2),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &process::cancellation(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(8 * 1024 * 1024),
                stderr: None,
            },
        )?;
        self.remaining()?;
        if !report.process.success() {
            return Err("diagnostic recovery Git probe failed".into());
        }
        Ok(report
            .stdout
            .ok_or("recovery Git output missing")?
            .as_bytes()
            .to_vec())
    }
    pub(super) fn text(&self, root: &Path, args: &[&str]) -> DynResult<String> {
        let bytes = self.run(root, &args.iter().map(OsString::from).collect::<Vec<_>>())?;
        Ok(std::str::from_utf8(&bytes)?.trim().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expired_recovery_budget_refuses_before_git_launch() {
        let git = Git {
            deadline: Instant::now() - Duration::from_secs(1),
        };
        assert!(
            git.run(Path::new("/"), &["must-not-run".into()])
                .unwrap_err()
                .to_string()
                .contains("120-second deadline")
        );
    }
}
