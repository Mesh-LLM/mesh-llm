//! Cache-specific retained serving cell. Baseline and old/new hosts share only
//! existing process/HTTP/model primitives, not fabricated competitive metadata.
#[path = "cache_family_cell/admission.rs"]
mod admission;
#[path = "cache_family_cell/contract.rs"]
mod contract;
#[path = "cache_family_cell/execution.rs"]
mod execution;
#[path = "cache_family_cell/owner.rs"]
mod owner;
#[path = "cache_family_cell/readiness.rs"]
mod readiness;
#[cfg(test)]
#[path = "cache_family_cell/tests.rs"]
mod tests;
use crate::command::DynResult;
use std::path::{Path, PathBuf};
fn paths(args: &[String]) -> DynResult<(PathBuf, PathBuf)> {
    let [a, input, b, output] = args else {
        return Err("cache-family-cell requires --input ABS --output ABS".into());
    };
    if a != "--input"
        || b != "--output"
        || !Path::new(input).is_absolute()
        || !Path::new(output).is_absolute()
    {
        return Err("cache-family-cell requires absolute ordered input/output".into());
    }
    Ok((input.into(), output.into()))
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!(
            "cargo xtool automation cache-family-cell [admit-worker|readiness-worker] --input ABS_JSON --output ABS_FRESH_DIRECTORY_OR_WORKER_FILE"
        );
        return Ok(());
    }
    if let [verb, rest @ ..] = args
        && ["admit-worker", "readiness-worker"].contains(&verb.as_str())
    {
        let (input, output) = paths(rest)?;
        return if verb == "admit-worker" {
            admission::run(&input, &output)
        } else {
            readiness::run(&input, &output)
        };
    }
    let (path, directory) = paths(args)?;
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(&path, 1024 * 1024)?;
    let input: contract::Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    if !std::fs::symlink_metadata(directory.parent().ok_or("cell output parent")?)?.is_dir() {
        return Err("cache cell output parent must be a regular directory".into());
    }
    std::fs::create_dir(&directory)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let result = execution::execute(&input, &directory, &cancel);
    let finish = interrupt.finish();
    let mut receipt = match result {
        Ok(value) => value,
        Err(error) => {
            serde_json::json!({"schema_version":1,"status":"failed","error":error.to_string()})
        }
    };
    receipt["request_sha256"] = serde_json::json!(admission::hash(&bytes));
    if finish.is_err() {
        receipt["status"] = serde_json::json!("incomplete");
        receipt["interrupt_observed"] = serde_json::json!(true);
    }
    crate::automation::waiting_prefix::adaptive_identity::fresh(
        &directory.join("cell.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    finish?;
    if receipt["status"] != "completed" {
        return Err("cache retained cell incomplete; partial evidence retained".into());
    }
    Ok(())
}
