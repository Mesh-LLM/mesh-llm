use super::{Failure, LineMatcher, Stream};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[cfg(test)]
#[path = "observed_spec_tests.rs"]
mod tests;

#[derive(Clone)]
pub enum Value {
    Public(OsString),
    Secret(OsString),
}

impl Value {
    pub(super) fn value(&self) -> &OsString {
        match self {
            Self::Public(value) | Self::Secret(value) => value,
        }
    }
}

pub struct ProcessSpec {
    pub executable: PathBuf,
    pub arguments: Vec<Value>,
    pub cwd: PathBuf,
    /// Complete child environment, never an overlay on the supervisor's secrets.
    pub environment: BTreeMap<OsString, Value>,
}

impl ProcessSpec {
    pub(super) fn command(&self) -> Result<Command, Failure> {
        if !self.executable.is_absolute() || !self.cwd.is_absolute() {
            return Err(Failure::InvalidSpec("executable and cwd must be absolute"));
        }
        #[cfg(windows)]
        if self
            .executable
            .extension()
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("exe"))
        {
            return Err(Failure::InvalidSpec(
                "Windows requires an explicit .exe, not a command script",
            ));
        }
        let mut command = Command::new(&self.executable);
        command.args(self.arguments.iter().map(Value::value));
        command.current_dir(&self.cwd).env_clear();
        command.envs(
            self.environment
                .iter()
                .map(|(key, value)| (key, value.value())),
        );
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Ok(command)
    }

    pub(super) fn secrets(&self) -> Result<Vec<Vec<u8>>, Failure> {
        self.arguments
            .iter()
            .chain(self.environment.values())
            .filter_map(|value| match value {
                Value::Public(_) => None,
                Value::Secret(value) => Some(value),
            })
            .map(|value| {
                if value.is_empty()
                    || value.as_encoded_bytes().len() > 4096
                    || value.as_encoded_bytes().contains(&b'\n')
                    || value.as_encoded_bytes().contains(&b'\r')
                {
                    return Err(Failure::InvalidSpec(
                        "secret values must contain 1..=4096 bytes without line breaks",
                    ));
                }
                #[cfg(unix)]
                let bytes = value.as_encoded_bytes().to_vec();
                #[cfg(windows)]
                let bytes = value
                    .to_str()
                    .ok_or(Failure::InvalidSpec(
                        "secret values must be Unicode on Windows",
                    ))?
                    .as_bytes()
                    .to_vec();
                Ok(bytes)
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct Cancellation(CancellationFlag);

#[derive(Clone)]
enum CancellationFlag {
    Shared(Arc<AtomicBool>),
    Static(&'static AtomicBool),
}

impl Default for Cancellation {
    fn default() -> Self {
        Self(CancellationFlag::Shared(Arc::default()))
    }
}

impl Cancellation {
    pub fn from_static(flag: &'static AtomicBool) -> Self {
        Self(CancellationFlag::Static(flag))
    }

    fn flag(&self) -> &AtomicBool {
        match &self.0 {
            CancellationFlag::Shared(flag) => flag,
            CancellationFlag::Static(flag) => flag,
        }
    }

    pub fn cancel(&self) {
        self.flag().store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.flag().load(Ordering::SeqCst)
    }
}

pub enum Readiness {
    None,
    /// Exact LF/EOF-terminated line, excluding LF and optional CR. Admission
    /// requires its bounded poll to finish before the deadline. A finite pipe
    /// snapshot taken before observing leader exit also remains eligible;
    /// cleanup/final drainage is diagnostic only. An exit observed first is
    /// EarlyExit; an expired budget observed first is a deadline outcome.
    Line {
        stream: Stream,
        bytes: Vec<u8>,
        deadline: Duration,
    },
    /// Both raw streams, independent of diagnostic retention/redaction.
    /// Only StopAfterReady is supported. Admission requires a live leader and
    /// completed observation/live checks strictly before the owner's deadline.
    /// Cleanup never invokes the matcher. Inspect the typed stop receipt, not
    /// just `ready`, to distinguish admission from a graceful request.
    ObservedLines {
        deadline: Duration,
        matcher: LineMatcher,
    },
}

#[derive(Clone, Copy)]
pub enum Completion {
    Exit,
    StopAfterReady,
}

pub struct Limits {
    pub execution: Duration,
    pub graceful_shutdown: Duration,
    pub forced_shutdown: Duration,
    pub retained_bytes_per_stream: usize,
    pub readiness: Readiness,
    pub completion: Completion,
}

impl Limits {
    pub(super) fn validate(&self) -> Result<(), Failure> {
        for budget in [self.execution, self.graceful_shutdown, self.forced_shutdown] {
            if budget.is_zero() || budget > Duration::from_secs(86400) {
                return Err(Failure::InvalidSpec(
                    "budgets must be positive and at most one day",
                ));
            }
        }
        if self.retained_bytes_per_stream > 16 * 1024 * 1024 {
            return Err(Failure::InvalidSpec(
                "retained output exceeds 16 MiB per stream",
            ));
        }
        match &self.readiness {
            Readiness::ObservedLines { deadline, .. } => {
                if deadline.is_zero() || *deadline > self.execution {
                    return Err(Failure::InvalidSpec("invalid readiness deadline"));
                }
                match self.completion {
                    Completion::StopAfterReady => Ok(()),
                    Completion::Exit => Err(Failure::InvalidSpec(
                        "observed lines require stop-after-ready",
                    )),
                }
            }
            Readiness::None => match self.completion {
                Completion::Exit => Ok(()),
                Completion::StopAfterReady => {
                    Err(Failure::InvalidSpec("stop-after-ready requires readiness"))
                }
            },
            Readiness::Line {
                bytes, deadline, ..
            } => {
                if bytes.is_empty()
                    || bytes.len() >= super::capture::LINE_LIMIT
                    || bytes.contains(&b'\n')
                    || deadline.is_zero()
                    || *deadline > self.execution
                {
                    return Err(Failure::InvalidSpec("invalid readiness marker or deadline"));
                }
                Ok(())
            }
        }
    }
}

#[derive(Default)]
pub struct OutputFiles {
    pub stdout: Option<PathBuf>,
    pub stderr: Option<PathBuf>,
}
