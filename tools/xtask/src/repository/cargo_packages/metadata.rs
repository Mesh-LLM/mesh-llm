use super::PackageName;
use crate::command_interrupt::{Interrupt, Reason};
use crate::process::{self, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value};
use crate::repo_consistency::CargoMetadata;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(super) const MAX_BYTES: usize = 16 * 1024 * 1024;

pub(super) enum Source {
    Cargo {
        executable: PathBuf,
        timeout: Duration,
    },
    File(PathBuf),
}

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error(transparent)]
    Interrupt(#[from] Reason),
    #[error("Cargo metadata signal finalization failed: {reason}; preceding result: {preceding:?}")]
    Finalization {
        reason: Reason,
        preceding: Box<Result<process::RawProcessReport, process::Failure>>,
    },
    #[error(transparent)]
    Process(#[from] process::Failure),
    #[error(
        "Cargo metadata failed: outcome={outcome:?}, exit={exit:?}, cleanup_complete={cleanup_complete}, forced={forced}"
    )]
    Execution {
        outcome: process::Outcome,
        exit: Option<i32>,
        cleanup_complete: bool,
        forced: bool,
    },
    #[error("Cargo metadata returned no complete raw stdout")]
    MissingPayload,
    #[error("Cargo metadata exceeded 16 MiB input limit")]
    InputLimit,
    #[error("invalid Cargo metadata JSON at line {line}, column {column}")]
    Json { line: usize, column: usize },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Name(#[from] super::Error),
}

pub(super) fn discover(cwd: &Path, source: Source) -> Result<BTreeSet<PackageName>, Error> {
    match source {
        Source::File(path) => {
            let mut bytes = Vec::new();
            std::fs::File::open(path)?
                .take(16 * 1024 * 1024 + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() > MAX_BYTES {
                return Err(Error::InputLimit);
            }
            parse(&bytes)
        }
        Source::Cargo {
            executable,
            timeout,
        } => {
            let spec = ProcessSpec {
                executable,
                arguments: ["metadata", "--locked", "--no-deps", "--format-version=1"]
                    .into_iter()
                    .map(|value| Value::Public(value.into()))
                    .collect(),
                cwd: cwd.to_path_buf(),
                environment: environment(std::env::vars_os()),
            };
            let limits = Limits {
                execution: timeout,
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            };
            let interrupt = Interrupt::install()?;
            let result = process::supervise_raw(
                &spec,
                &limits,
                &interrupt.cancellation(),
                RawCaptureOptions {
                    stdout: std::num::NonZeroUsize::new(MAX_BYTES),
                    stderr: None,
                },
            );
            let report = finalize(result, interrupt.finish())?;
            if !report.process.success() {
                if let Some(error) = report.process.failure {
                    return Err(Error::Process(error));
                }
                return Err(Error::Execution {
                    outcome: report.process.outcome,
                    exit: report.process.status.and_then(|status| status.code()),
                    cleanup_complete: report.process.cleanup.complete,
                    forced: report.process.cleanup.forced,
                });
            }
            parse(report.stdout.ok_or(Error::MissingPayload)?.as_bytes())
        }
    }
}

fn finalize(
    preceding: Result<process::RawProcessReport, process::Failure>,
    finalization: Result<(), Reason>,
) -> Result<process::RawProcessReport, Error> {
    match finalization {
        Ok(()) => preceding.map_err(Error::Process),
        Err(reason) => Err(Error::Finalization {
            reason,
            preceding: Box::new(preceding),
        }),
    }
}

fn parse(bytes: &[u8]) -> Result<BTreeSet<PackageName>, Error> {
    let metadata: CargoMetadata = serde_json::from_slice(bytes).map_err(|error| Error::Json {
        line: error.line(),
        column: error.column(),
    })?;
    let members = metadata
        .workspace_members
        .into_iter()
        .collect::<BTreeSet<_>>();
    metadata
        .packages
        .into_iter()
        .filter(|package| members.contains(&package.id))
        .map(|package| PackageName::try_from(package.name).map_err(Error::Name))
        .collect()
}

fn environment(
    values: impl IntoIterator<Item = (OsString, OsString)>,
) -> BTreeMap<OsString, Value> {
    const ALLOWED: &[&str] = &[
        "PATH",
        "HOME",
        "USERPROFILE",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
        "CARGO_NET_OFFLINE",
        "SystemRoot",
        "WINDIR",
        "TMP",
        "TEMP",
        "TMPDIR",
    ];
    values
        .into_iter()
        .filter(|(key, _)| {
            ALLOWED
                .iter()
                .any(|allowed| key.as_os_str() == std::ffi::OsStr::new(allowed))
        })
        .map(|(key, value)| (key, Value::Public(value)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_keeps_offline_toolchain_when_secrets_and_git_overrides_are_present() {
        let input = [
            "PATH",
            "HOME",
            "CARGO_HOME",
            "RUSTUP_HOME",
            "RUSTUP_TOOLCHAIN",
            "CARGO_NET_OFFLINE",
            "GIT_DIR",
            "GIT_CONFIG_COUNT",
            "GH_TOKEN",
            "CARGO_REGISTRIES_PRIVATE_TOKEN",
            "UNRELATED",
        ];
        let result = environment(input.map(|key| (key.into(), "fixture".into())));
        assert_eq!(
            result
                .keys()
                .map(|key| key.to_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "CARGO_HOME",
                "CARGO_NET_OFFLINE",
                "HOME",
                "PATH",
                "RUSTUP_HOME",
                "RUSTUP_TOOLCHAIN"
            ]
        );
    }
}

#[cfg(test)]
#[path = "metadata_finalization_tests.rs"]
mod finalization_tests;
