//! Correlated bounded HTTP worker. Host identity/readiness custody belongs to its
//! retained parent; this command makes no claim that an endpoint is the pinned host.
#[path = "cache_family_measure/contract.rs"]
pub(super) mod contract;
#[path = "cache_family_measure/measurement.rs"]
mod measurement;
#[path = "cache_family_measure/sweep.rs"]
mod sweep;
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
#[cfg(test)]
#[path = "cache_family_measure/tests.rs"]
mod tests;
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::{io::Write as _, path::Path};
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(
            std::io::stdout().lock(),
            "cargo xtool automation cache-family-measure --input ABS_JSON --output ABS_FRESH_JSON"
        )?;
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
    let bytes = crate::automation::receipt_files::bounded(Path::new(input), 2 * 1024 * 1024)?;
    let input: serde_json::Value = serde_json::from_slice(&bytes)?;
    if input.get("stages").is_some() {
        let value: sweep::InputSweep = serde_json::from_value(input.clone())?;
        value.validate()?;
    } else {
        let value: contract::Input = serde_json::from_value(input.clone())?;
        value.validate()?;
    }
    let terminal_deadline = std::time::Instant::now()
        + std::time::Duration::from_millis(
            input["execution_timeout_ms"]
                .as_u64()
                .ok_or("cache measurement terminal budget absent")?,
        );
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut receipt = if input.get("stages").is_some() {
        let value: sweep::InputSweep = serde_json::from_value(input)?;
        runtime.block_on(sweep::execute(&value, hash(&bytes), &cancellation))
    } else {
        let value: contract::Input = serde_json::from_value(input)?;
        runtime.block_on(measurement::execute(&value, hash(&bytes), &cancellation))
    };
    let finish = interrupt.finish();
    finalize(
        &mut receipt,
        cancellation.is_cancelled(),
        std::time::Instant::now() >= terminal_deadline,
        finish.is_ok(),
    );
    let encoded = serde_json::to_vec_pretty(&receipt)?;
    if encoded.len() > 64 * 1024 * 1024 {
        return Err("cache measurement receipt exceeds 64 MiB".into());
    }
    crate::automation::receipt_files::fresh(Path::new(output), &encoded)?;
    finish?;
    if receipt["status"] != "completed" {
        return Err("cache measurement incomplete; partial receipt retained".into());
    }
    Ok(())
}

fn finalize(receipt: &mut serde_json::Value, cancelled: bool, expired: bool, finish_ok: bool) {
    if cancelled || expired || !finish_ok {
        receipt["status"] = serde_json::json!("incomplete");
        receipt["terminal_refusal"] = serde_json::json!({"cancelled":cancelled,"deadline_expired":expired,"interrupt_finish_failed":!finish_ok});
    }
}
