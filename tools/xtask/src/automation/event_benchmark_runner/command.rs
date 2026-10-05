//! Production event benchmark matrix and bounded native worker dispatch.
use super::{evidence_io, matrix, options, preflight, worker_frontends};
use crate::{automation::command_interrupt::Interrupt, command::DynResult};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn regular(path: &Path) -> DynResult<PathBuf> {
    let absolute = std::path::absolute(path)?;
    if !std::fs::symlink_metadata(&absolute)?.file_type().is_file() {
        return Err("benchmark binary/model input must be a regular local file".into());
    }
    Ok(absolute.canonicalize()?)
}

fn output(path: &Path) -> DynResult<PathBuf> {
    let absolute = std::path::absolute(path)?;
    let parent = absolute
        .parent()
        .ok_or("benchmark output directory has no parent")?
        .canonicalize()?;
    let filename = absolute
        .file_name()
        .ok_or("benchmark output requires a fresh named directory")?;
    let result = parent.join(filename);
    // Existing evidence and symlinks are never reused or overwritten.
    std::fs::create_dir(&result)?;
    Ok(result)
}

fn run_matrix(args: &[String]) -> DynResult<()> {
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let result = drive(args, &cancellation, preflight::prepare, matrix::execute);
    let restored = interrupt.finish();
    let result = result?;
    restored?;
    println!(
        "{}",
        serde_json::to_string(&json!({"manifest_paths":result.manifest_paths}))?
    );
    if result.failed {
        Err(
            "benchmark matrix incomplete or contains failed trials; partial evidence retained"
                .into(),
        )
    } else {
        Ok(())
    }
}

pub(super) struct ResultReceipt {
    pub manifest_paths: std::collections::BTreeMap<String, String>,
    pub failed: bool,
}

pub(super) fn drive(
    args: &[String],
    cancellation: &crate::process::Cancellation,
    mut prepare: impl FnMut(
        &options::Command,
        &Path,
        Duration,
        &crate::process::Cancellation,
    ) -> DynResult<preflight::Prepared>,
    mut execute: impl FnMut(
        &options::Command,
        &super::admission::Metadata,
        Instant,
        &crate::process::Cancellation,
    ) -> DynResult<matrix::Execution>,
) -> DynResult<ResultReceipt> {
    let start = Instant::now();
    let mut command = options::Command::parse(args)?;
    command.model = regular(&command.model)?;
    for side in &mut command.sides {
        side.binary = regular(&side.binary)?;
    }
    command.output_dir = output(&command.output_dir)?;
    let directory = command.output_dir.join("preflight");
    std::fs::create_dir(&directory)?;
    let prepared = match prepare(
        &command,
        &directory,
        Duration::from_secs(command.execution_secs).saturating_sub(start.elapsed()),
        cancellation,
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            evidence_io::publish(
                &command.output_dir.join("preflight-error.json"),
                &json!({"schema_version":1,"error":error.to_string()}),
                evidence_io::RECEIPT_BYTES,
            )?;
            return Err(error);
        }
    };
    let execution = execute(&command, &prepared.metadata, start, cancellation)?;
    let generated_at = format!(
        "unix-ms:{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()
    );
    let manifest_paths = matrix::publish(&command, &prepared, &execution, &generated_at)?;
    let failed = execution.batch.interrupted.is_some()
        || execution
            .batch
            .trials
            .iter()
            .flatten()
            .any(|row| row.status != "succeeded");
    Ok(ResultReceipt {
        manifest_paths,
        failed,
    })
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [help] if help == "--help" => {
            println!("{}", options::USAGE);
            Ok(())
        }
        [verb, rest @ ..] if verb == "measurement-worker" || verb == "identity-worker" => {
            worker_frontends::run(verb, rest)
        }
        _ => run_matrix(args),
    }
}
