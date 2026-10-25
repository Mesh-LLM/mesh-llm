//! Parent-owned preflight/revalidation. All heavy file hashing remains in owned workers.
use super::{
    admission, evidence_io, identity_worker, options, probes,
    worker_frontends::{self, ExpectedFile, IdentityData, IdentityRequest},
};
use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
pub(super) struct Prepared {
    pub metadata: admission::Metadata,
    pub thermal_state: serde_json::Value,
}
struct Deadline(Instant);
impl Deadline {
    fn new(remaining: Duration) -> DynResult<Self> {
        Ok(Self(
            Instant::now()
                .checked_add(remaining)
                .ok_or("preflight deadline overflow")?,
        ))
    }
    fn remaining(&self) -> Duration {
        self.0.saturating_duration_since(Instant::now())
    }
    fn execution(&self) -> DynResult<Duration> {
        // supervise also drains stdout/stderr for a separate forced-shutdown window.
        let budget = self.remaining().saturating_sub(Duration::from_millis(750));
        if budget.is_zero() {
            Err("preflight overall deadline expired".into())
        } else {
            Ok(budget)
        }
    }
}
fn run(
    tool: &Path,
    args: Vec<std::ffi::OsString>,
    directory: &Path,
    name: &str,
    deadline: &Deadline,
    cancel: &Cancellation,
) -> DynResult<()> {
    if cancel.is_cancelled() {
        return Err("preflight interrupted".into());
    }
    let spec = ProcessSpec {
        executable: tool.to_owned(),
        cwd: directory.to_owned(),
        arguments: args.into_iter().map(Value::Public).collect(),
        environment: BTreeMap::from([("PATH".into(), Value::Public("/usr/bin:/bin".into()))]),
    };
    let limits = Limits {
        execution: deadline.execution()?,
        graceful_shutdown: Duration::from_millis(250),
        forced_shutdown: Duration::from_millis(250),
        retained_bytes_per_stream: 64 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise(
        &spec,
        &limits,
        cancel,
        OutputFiles {
            stdout: Some(directory.join(format!("{name}.stdout.log"))),
            stderr: Some(directory.join(format!("{name}.stderr.log"))),
        },
    )?;
    if !report.cleanup.complete
        || !report.success()
        || report.stdout.truncated
        || report.stderr.truncated
        || report.stdout.suppressed_lines > 0
        || report.stderr.suppressed_lines > 0
    {
        return Err(format!("preflight owned step {name} failed or lost diagnostics").into());
    }
    Ok(())
}
fn worker(
    tool: &Path,
    request: &IdentityRequest,
    directory: &Path,
    deadline: &Deadline,
    cancel: &Cancellation,
) -> DynResult<IdentityData> {
    if !directory.is_absolute() {
        return Err("preflight directory must be absolute and parent-owned".into());
    }
    let input = directory.join("identity.input.json");
    let output = directory.join("identity.receipt.json");
    evidence_io::publish(&input, request, worker_frontends::IDENTITY_BYTES)?;
    run(
        tool,
        vec![
            "automation".into(),
            "event-benchmark-run".into(),
            "identity-worker".into(),
            "--input".into(),
            input.into_os_string(),
            "--output".into(),
            output.as_os_str().into(),
        ],
        directory,
        "identity-worker",
        deadline,
        cancel,
    )?;
    worker_frontends::correlated(&output, request)
}
fn absolute(path: &Path, cwd: &Path) -> PathBuf {
    if path.is_absolute() {
        path.into()
    } else {
        cwd.join(path)
    }
}
pub(super) fn prepare(
    command: &options::Command,
    directory: &Path,
    remaining: Duration,
    cancel: &Cancellation,
) -> DynResult<Prepared> {
    prepare_with_tool(
        command,
        directory,
        remaining,
        cancel,
        &std::env::current_exe()?,
        Path::new("/usr/bin/pmset"),
    )
}
fn prepare_with_tool(
    command: &options::Command,
    directory: &Path,
    remaining: Duration,
    cancel: &Cancellation,
    tool: &Path,
    pmset: &Path,
) -> DynResult<Prepared> {
    let deadline = Deadline::new(remaining)?;
    let cwd = std::env::current_dir()?;
    let mut sides = command.sides.clone();
    for side in &mut sides {
        side.binary = absolute(&side.binary, &cwd);
    }
    let input = identity_worker::Input {
        schema_version: 1,
        binaries: [sides[0].binary.clone(), sides[1].binary.clone()],
        model: absolute(&command.model, &cwd),
        minimum_context_tokens: command
            .max_tokens
            .checked_add(128)
            .ok_or("context request overflow")?,
    };
    let request = IdentityRequest::Admit { input };
    let IdentityData::Admitted {
        identity,
        runtime_packages,
        mut thermal_state,
    } = worker(tool, &request, directory, &deadline, cancel)?
    else {
        return Err("preflight worker returned wrong operation".into());
    };
    let mut validated = BTreeSet::new();
    for (index, packages) in runtime_packages.iter().enumerate() {
        if packages.is_empty() || packages.len() > 64 {
            return Err("preflight receipt lacks bounded native packages".into());
        }
        for package in packages {
            if package.parent() != Some(identity.binaries[index].adjacent_runtime_root.as_path()) {
                return Err("runtime package escaped admitted root".into());
            }
            if validated.insert(package.clone()) {
                run(
                    tool,
                    vec![
                        "native".into(),
                        "verify-runtime-package".into(),
                        "--portable".into(),
                        package.as_os_str().into(),
                    ],
                    directory,
                    &format!("runtime-package-{}", validated.len()),
                    &deadline,
                    cancel,
                )?;
            }
        }
    }
    let first = probes::version(
        &identity.binaries[0].file.path,
        deadline.remaining(),
        cancel,
    )?;
    let second = if identity.binaries[0].file.path == identity.binaries[1].file.path {
        first.clone()
    } else {
        probes::version(
            &identity.binaries[1].file.path,
            deadline.remaining(),
            cancel,
        )?
    };
    if std::env::consts::OS == "macos" {
        thermal_state = probes::darwin_thermal(pmset, deadline.remaining(), cancel)?;
    }
    let IdentityRequest::Admit { input } = request else {
        unreachable!()
    };
    let mut metadata = admission::bind(&input, &sides, *identity, [first, second])?;
    metadata.runtime_packages_verified = true;
    if cancel.is_cancelled() || deadline.remaining().is_zero() {
        return Err("preflight interrupted or overall deadline expired".into());
    }
    Ok(Prepared {
        metadata,
        thermal_state,
    })
}
pub(super) fn revalidate(
    metadata: &admission::Metadata,
    directory: &Path,
    remaining: Duration,
    cancel: &Cancellation,
) -> DynResult<()> {
    let deadline = Deadline::new(remaining)?;
    if !metadata.runtime_packages_verified {
        return Err("revalidation requires owning portable package validation".into());
    }
    let request = IdentityRequest::Revalidate {
        binaries: [
            ExpectedFile {
                path: metadata.binaries[0].path.clone(),
                sha256: metadata.binaries[0].sha256.clone(),
            },
            ExpectedFile {
                path: metadata.binaries[1].path.clone(),
                sha256: metadata.binaries[1].sha256.clone(),
            },
        ],
        model: ExpectedFile {
            path: metadata.model.clone(),
            sha256: metadata.source_model_sha256.clone(),
        },
    };
    let result = worker(
        &std::env::current_exe()?,
        &request,
        directory,
        &deadline,
        cancel,
    )?;
    if !matches!(result, IdentityData::Revalidated) {
        return Err("revalidation worker returned wrong operation".into());
    }
    if cancel.is_cancelled() || deadline.remaining().is_zero() {
        return Err("revalidation interrupted or deadline expired".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "preflight_tests.rs"]
mod tests;
