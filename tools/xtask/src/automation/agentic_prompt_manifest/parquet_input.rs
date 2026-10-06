//! Bounded process adapter to the separately built optional trajectory reader.
use super::selection::{Selection, Selector, Trajectory};
use crate::{
    automation::command_interrupt::Interrupt,
    command::DynResult,
    process::{self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};
use trajectory_reader::wire::{Request, Response, SCHEMA_VERSION};

pub(super) fn select(path: &Path, selection: &Selection) -> DynResult<Vec<Trajectory>> {
    selection.validate()?;
    let reader = reader_binary()?;
    let directory = tempfile::tempdir()?;
    let request_file = directory.path().join("request.json");
    let response_file = directory.path().join("response.json");
    let request = Request {
        schema_version: SCHEMA_VERSION,
        dataset_file: path.canonicalize()?,
        selection: selection.clone(),
    };
    std::fs::write(&request_file, serde_json::to_vec(&request)?)?;
    let spec = ProcessSpec {
        executable: reader,
        cwd: directory.path().to_path_buf(),
        arguments: vec![
            Value::Public("--request".into()),
            Value::Public(request_file.into_os_string()),
            Value::Public("--response".into()),
            Value::Public(response_file.clone().into_os_string()),
        ],
        environment: reader_environment(),
    };
    let interrupt = Interrupt::install()?;
    let result = process::supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(120),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &interrupt.cancellation(),
        OutputFiles::default(),
    );
    interrupt.finish()?;
    let report = result?;
    if !report.success() {
        return Err(format!(
            "trajectory reader failed: {:?}; {}",
            report.outcome,
            String::from_utf8_lossy(&report.stderr.bytes_retained)
        )
        .into());
    }
    let response: Response = serde_json::from_reader(std::fs::File::open(response_file)?)?;
    let rows = response.into_rows(selection.families)?;
    let mut selector = Selector::new(selection)?;
    for row in rows {
        selector.observe(row);
    }
    selector.finish()
}

pub(super) fn reader_binary() -> DynResult<PathBuf> {
    if let Some(path) = std::env::var_os("MESH_LLM_TRAJECTORY_READER_BIN") {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            return Err("MESH_LLM_TRAJECTORY_READER_BIN must name an absolute executable".into());
        }
        return Ok(path.canonicalize()?);
    }
    let executable = std::env::current_exe()?;
    let name = if cfg!(windows) {
        "trajectory-reader.exe"
    } else {
        "trajectory-reader"
    };
    let reader = executable.with_file_name(name);
    reader.canonicalize().map_err(|error| format!("optional trajectory reader missing at {}: {error}; build it with just with-lld cargo build --locked -p trajectory-reader --features parquet-input --bin trajectory-reader, or set MESH_LLM_TRAJECTORY_READER_BIN", reader.display()).into())
}

pub(super) fn reader_environment() -> BTreeMap<std::ffi::OsString, Value> {
    ["SYSTEMROOT", "WINDIR"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (name.into(), Value::Public(value))))
        .collect()
}
