//! Correlated bounded HTTP worker. Host identity/readiness custody belongs to its
//! retained parent; this command makes no claim that an endpoint is the pinned host.
#[path = "cache_family_measure/contract.rs"]
pub(super) mod contract;
#[path = "cache_family_measure/measurement.rs"]
mod measurement;
#[cfg(test)]
#[path = "cache_family_measure/tests.rs"]
mod tests;
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::path::Path;
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!(
            "cargo xtool automation cache-family-measure --input ABS_JSON --output ABS_FRESH_JSON"
        );
        return Ok(());
    }
    let [input_flag, input, output_flag, output] = args else {
        return Err("cache-family-measure requires --input ABS --output ABS".into());
    };
    if input_flag != "--input"
        || output_flag != "--output"
        || !Path::new(input).is_absolute()
        || !Path::new(output).is_absolute()
    {
        return Err("cache-family-measure requires absolute ordered input/output flags".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("cache measurement output must be fresh".into()),
    }
    let parent = Path::new(output)
        .parent()
        .ok_or("measurement output parent")?;
    if !std::fs::symlink_metadata(parent)?.is_dir() {
        return Err("measurement output parent must be a regular directory".into());
    }
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(
        Path::new(input),
        512 * 1024,
    )?;
    let input: contract::Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let receipt = runtime.block_on(measurement::execute(
        &input,
        hex::encode(Sha256::digest(&bytes)),
        &cancellation,
    ));
    let finish = interrupt.finish();
    let encoded = serde_json::to_vec_pretty(&receipt)?;
    if encoded.len() > 64 * 1024 * 1024 {
        return Err("cache measurement receipt exceeds 64 MiB".into());
    }
    crate::automation::waiting_prefix::adaptive_identity::fresh(Path::new(output), &encoded)?;
    finish?;
    if receipt["status"] != "completed" {
        return Err("cache measurement incomplete; partial receipt retained".into());
    }
    Ok(())
}
