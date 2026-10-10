//! Bounded, executable-bound serving dialect admission shared by native benchmark owners.
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
        Value,
    },
};
use std::{
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) enum Role {
    Public,
    BinaryWorker,
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) enum Dialect {
    Current,
    Legacy,
}
impl Role {
    fn legacy(self) -> &'static str {
        match self {
            Self::Public => "serve-openai",
            Self::BinaryWorker => "serve-binary",
        }
    }
}
fn verify(path: &Path, expected: &str) -> DynResult<()> {
    if !path.is_absolute()
        || !std::fs::symlink_metadata(path)?.is_file()
        || crate::product::digest::file_sha256(path).map_err(|error| error.error)? != expected
    {
        return Err("Skippy CLI executable identity differs from admitted preimage".into());
    }
    Ok(())
}
pub(crate) fn admit(
    spec: &ProcessSpec,
    expected: &str,
    role: Role,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<Dialect> {
    for (command, dialect) in [
        ("serve", Dialect::Current),
        (role.legacy(), Dialect::Legacy),
    ] {
        verify(&spec.executable, expected)?;
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .checked_sub(Duration::from_secs(3))
            .filter(|value| !value.is_zero())
            .ok_or("Skippy CLI probe budget exhausted")?;
        if cancel.is_cancelled() {
            return Err("Skippy CLI probe cancelled".into());
        }
        let probe = ProcessSpec {
            executable: spec.executable.clone(),
            arguments: [command, "--help"]
                .into_iter()
                .map(|value| Value::Public(value.into()))
                .collect(),
            environment: spec.environment.clone(),
            cwd: spec.cwd.clone(),
        };
        let report = process::supervise_raw(
            &probe,
            &Limits {
                execution: remaining.min(Duration::from_secs(5)),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            cancel,
            RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )?;
        verify(&spec.executable, expected)?;
        for bytes in [&report.stdout, &report.stderr] {
            std::str::from_utf8(
                bytes
                    .as_ref()
                    .ok_or("Skippy help capture absent")?
                    .as_bytes(),
            )?;
        }
        if cancel.is_cancelled() || Instant::now() >= deadline {
            return Err("Skippy CLI probe cancelled or expired".into());
        }
        if report.process.success() {
            return Ok(dialect);
        }
        if report.process.outcome != Outcome::Exited
            || !report.process.cleanup.complete
            || report.process.cleanup.forced
            || report.process.cleanup.graceful_signal_failed
            || report.process.cleanup.failure.is_some()
            || report.process.failure.is_some()
            || report
                .process
                .status
                .is_none_or(|status| status.code().is_none())
        {
            return Err("Skippy CLI help did not complete cleanly".into());
        }
    }
    Err("Skippy executable supports neither current nor legacy serving role".into())
}
impl Dialect {
    /// Only apply to typed, owner-constructed legacy serving arguments.
    pub(crate) fn arguments(self, role: Role, args: &mut Vec<Value>) -> DynResult<()> {
        let Some(Value::Public(command)) = args.first() else {
            return Err("typed Skippy serving command absent".into());
        };
        if command != role.legacy() {
            return Err("typed Skippy serving role differs".into());
        }
        if matches!(self, Self::Legacy) {
            return Ok(());
        }
        args[0] = Value::Public("serve".into());
        let mut prefix = Vec::new();
        if !matches!(role, Role::Public) {
            prefix.extend([
                Value::Public("--stage-transport".into()),
                Value::Public("binary".into()),
            ]);
        }
        if matches!(role, Role::BinaryWorker) {
            prefix.push(Value::Public("--worker-only".into()));
        }
        args.splice(1..1, prefix);
        Ok(())
    }
}
pub(crate) fn prepare(
    spec: &mut ProcessSpec,
    expected: &str,
    role: Role,
    deadline: Instant,
    cancel: &Cancellation,
    receipt: &Path,
) -> DynResult<Dialect> {
    use std::io::Write as _;
    let dialect = admit(spec, expected, role, deadline, cancel)?;
    dialect.arguments(role, &mut spec.arguments)?;
    let bytes = serde_json::to_vec(
        &serde_json::json!({"schema_version":1,"binary":spec.executable,"binary_sha256":expected,"role":role,"dialect":dialect,"qualification":false,"evidence":"bounded exact executable help observation"}),
    )?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(receipt)?
        .write_all(&bytes)?;
    Ok(dialect)
}
#[cfg(test)]
#[path = "skippy_cli_admission/tests.rs"]
mod tests;
