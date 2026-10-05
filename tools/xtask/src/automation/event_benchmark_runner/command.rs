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
    let start = Instant::now();
    let mut command = options::Command::parse(args)?;
    command.model = regular(&command.model)?;
    for side in &mut command.sides {
        side.binary = regular(&side.binary)?;
    }
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let result = (|| -> DynResult<()> {
        command.output_dir = output(&command.output_dir)?;
        let preflight_directory = command.output_dir.join("preflight");
        std::fs::create_dir(&preflight_directory)?;
        let prepared = match preflight::prepare(
            &command,
            &preflight_directory,
            Duration::from_secs(command.execution_secs).saturating_sub(start.elapsed()),
            &cancellation,
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
        let execution = matrix::execute(&command, &prepared.metadata, start, &cancellation)?;
        let generated_at = format!(
            "unix-ms:{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()
        );
        let manifest_paths = matrix::publish(&command, &prepared, &execution, &generated_at)?;
        println!(
            "{}",
            serde_json::to_string(&json!({"manifest_paths":manifest_paths}))?
        );
        if execution.batch.interrupted.is_some()
            || execution
                .batch
                .trials
                .iter()
                .flatten()
                .any(|row| row.status != "succeeded")
        {
            return Err(
                "benchmark matrix incomplete or contains failed trials; partial evidence retained"
                    .into(),
            );
        }
        Ok(())
    })();
    let restored = interrupt.finish();
    result?;
    restored?;
    Ok(())
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
