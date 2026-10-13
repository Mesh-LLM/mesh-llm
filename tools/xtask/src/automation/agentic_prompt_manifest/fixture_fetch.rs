//! Optional Hugging Face CLI adapter; local validation/materialization never
//! starts this ecosystem tool or installs its dependencies.
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use serde_json::Value as Json;

use super::fixture_catalog as catalog;
use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};

pub(super) struct Adapter {
    pub executable: PathBuf,
    pub cwd: PathBuf,
    pub environment: BTreeMap<OsString, Value>,
    pub cache_dir: Option<PathBuf>,
    pub budget: Duration,
}

pub(super) fn fetch(
    dataset: &Json,
    adapter: &Adapter,
    cancellation: &Cancellation,
) -> DynResult<PathBuf> {
    if adapter.budget.is_zero() {
        return Err("fixture fetch budget must be positive".into());
    }
    let started = Instant::now();
    let download = arguments(dataset, adapter.cache_dir.as_deref(), false)?;
    let output = execute(adapter, download, started, cancellation, true)?;
    let snapshot = std::str::from_utf8(&output)?.trim();
    if snapshot.is_empty() || snapshot.contains(['\n', '\r']) {
        return Err("HF download must return one snapshot path".into());
    }
    execute(
        adapter,
        arguments(dataset, adapter.cache_dir.as_deref(), true)?,
        started,
        cancellation,
        false,
    )?;
    let snapshot = Path::new(snapshot);
    let snapshot = if snapshot.is_absolute() {
        snapshot.to_path_buf()
    } else {
        adapter.cwd.join(snapshot)
    };
    let parquet = snapshot.join(catalog::text(dataset, "parquet_file")?);
    if !parquet.is_file() {
        return Err("verified HF snapshot is missing the pinned parquet file".into());
    }
    Ok(parquet)
}

pub(super) fn arguments(
    dataset: &Json,
    cache: Option<&Path>,
    verify: bool,
) -> DynResult<Vec<OsString>> {
    let repo = catalog::text(dataset, "repo_id")?;
    if repo.starts_with('-') || repo.chars().any(char::is_whitespace) {
        return Err("invalid HF fixture repository".into());
    }
    let mut args: Vec<OsString> = if verify {
        vec!["cache".into(), "verify".into(), repo.into()]
    } else {
        vec!["download".into(), repo.into()]
    };
    if !verify {
        let files = dataset
            .get("files")
            .and_then(Json::as_array)
            .ok_or("HF fixture files must be an array")?;
        for file in files {
            let file = file
                .as_str()
                .filter(|file| !file.is_empty())
                .ok_or("HF fixture filename must be text")?;
            if file.starts_with('-')
                || file.contains('\\')
                || Path::new(file).is_absolute()
                || Path::new(file)
                    .components()
                    .any(|part| !matches!(part, std::path::Component::Normal(_)))
            {
                return Err("HF fixture filename must be a relative repository path".into());
            }
            args.push(file.into());
        }
        // The vendor CLI's automatic format may emit structured agent output.
        // Request the snapshot path explicitly for the native adapter.
        args.push("--quiet".into());
    }
    args.extend([
        "--repo-type".into(),
        catalog::text(dataset, "repo_type")?.into(),
        "--revision".into(),
        catalog::text(dataset, "revision")?.into(),
    ]);
    if verify {
        args.push("--fail-on-missing-files".into());
    }
    if let Some(cache) = cache {
        args.extend(["--cache-dir".into(), cache.as_os_str().to_owned()]);
    }
    Ok(args)
}

fn execute(
    adapter: &Adapter,
    arguments: Vec<OsString>,
    started: Instant,
    cancellation: &Cancellation,
    snapshot_output: bool,
) -> DynResult<Vec<u8>> {
    let remaining = adapter
        .budget
        .checked_sub(started.elapsed())
        .filter(|remaining| !remaining.is_zero())
        .ok_or("HF fixture fetch deadline expired")?;
    let spec = ProcessSpec {
        executable: adapter.executable.clone(),
        cwd: adapter.cwd.clone(),
        arguments: arguments.into_iter().map(Value::Public).collect(),
        environment: adapter
            .environment
            .iter()
            .map(|(key, value)| {
                let copied = match value {
                    Value::Public(value) => Value::Public(value.clone()),
                    Value::Secret(value) => Value::Secret(value.clone()),
                };
                (key.clone(), copied)
            })
            .collect(),
    };
    let report = process::supervise(
        &spec,
        &Limits {
            execution: remaining,
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        OutputFiles::default(),
    )?;
    if !report.success() || (snapshot_output && report.stdout.truncated) {
        return Err(format!(
            "HF fixture command failed: {:?}; {}",
            report.outcome,
            String::from_utf8_lossy(&report.stderr.bytes_retained)
        )
        .into());
    }
    Ok(report.stdout.bytes_retained)
}
