use super::contract::{Input, Resolved};
use crate::command::DynResult;
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) fn verify_revisions(input: &Input, resolved: &Resolved) -> DynResult<()> {
    verify_revisions_with_cancellation(input, resolved, &Cancellation::default())
}

pub(super) fn verify_revisions_with_cancellation(
    input: &Input,
    resolved: &Resolved,
    cancellation: &Cancellation,
) -> DynResult<()> {
    for (root, expected) in [
        (&resolved.controller, &input.controller_revision),
        (&resolved.source, &input.selected_revision),
    ] {
        let spec = ProcessSpec {
            executable: PathBuf::from("/usr/bin/git"),
            arguments: ["--no-optional-locks", "rev-parse", "HEAD"]
                .into_iter()
                .map(|value| Value::Public(value.into()))
                .collect(),
            cwd: root.clone(),
            environment: environment(),
        };
        let report = execute(&spec, Duration::from_secs(10), cancellation)?;
        if std::str::from_utf8(&report.stdout.bytes_retained)?.trim() != expected.as_str() {
            return Err("selected source or controller revision mismatch".into());
        }
    }
    Ok(())
}

pub(super) fn preflight_with_cancellation(
    resolved: &Resolved,
    plan: &Path,
    cancellation: &Cancellation,
) -> DynResult<()> {
    let battery = resolved
        .source
        .join("scripts/skippy-family-battery.sh")
        .canonicalize()?;
    if !battery.starts_with(&resolved.source) {
        return Err("selected battery escapes selected source".into());
    }
    let mut environment = environment();
    environment.insert(
        "FAMILY_BATTERY_ARTIFACT_ROOT".into(),
        Value::Public(resolved.output.join("battery-evidence").into()),
    );
    environment.insert(
        "FAMILY_BATTERY_RUN_ID".into(),
        Value::Public("contract".into()),
    );
    let spec = ProcessSpec {
        executable: PathBuf::from("/bin/bash"),
        arguments: [
            battery.into_os_string(),
            "--skip-build".into(),
            "--dry-run".into(),
            "--plan".into(),
            plan.as_os_str().to_owned(),
        ]
        .into_iter()
        .map(Value::Public)
        .collect(),
        cwd: resolved.source.clone(),
        environment,
    };
    execute(&spec, Duration::from_secs(120), cancellation)?;
    Ok(())
}

fn environment() -> BTreeMap<OsString, Value> {
    ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (name.into(), Value::Public(value))))
        .collect()
}

fn execute(
    spec: &ProcessSpec,
    execution: Duration,
    cancellation: &Cancellation,
) -> DynResult<process::ProcessReport> {
    let report = process::supervise(
        spec,
        &Limits {
            execution,
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        OutputFiles::default(),
    )?;
    if !report.success() || report.stdout.truncated || report.stderr.truncated {
        return Err(format!(
            "controller preflight command failed: {:?}, status {:?}, truncated stdout={}, stderr={}, cleanup={:?}, failure={:?}: {}",
            report.outcome,
            report.status,
            report.stdout.truncated,
            report.stderr.truncated,
            report.cleanup,
            report.failure,
            String::from_utf8_lossy(&report.stderr.bytes_retained)
        )
        .into());
    }
    Ok(report)
}
