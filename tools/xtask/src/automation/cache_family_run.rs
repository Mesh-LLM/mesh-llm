//! Bounded composition of existing cache plan/correctness/retained serving owners.
#[path = "cache_family_run/child.rs"]
mod child;
#[path = "cache_family_run/contract.rs"]
mod contract;
#[path = "cache_family_run/metrics.rs"]
mod metrics;
#[path = "cache_family_run/pipeline.rs"]
mod pipeline;
#[path = "cache_family_run/reporting.rs"]
mod reporting;
#[path = "cache_family_run/sweeps.rs"]
mod sweeps;
#[cfg(test)]
#[path = "cache_family_run/tests.rs"]
mod tests;
use crate::command::DynResult;
use serde_json::Value;
use std::{io::Write as _, path::Path};
fn hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
fn publish(path: &Path, value: &Value) -> DynResult<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err("cache matrix projection exceeds64MiB".into());
    }
    crate::automation::waiting_prefix::adaptive_identity::fresh(path, &bytes)
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(
            std::io::stdout().lock(),
            "cargo xtool automation cache-family-run --input ABS_JSON --output ABS_FRESH_DIRECTORY"
        )?;
        return Ok(());
    }
    let [a, path, b, output] = args else {
        return Err("cache-family-run requires ordered input/output".into());
    };
    if a != "--input"
        || b != "--output"
        || !Path::new(path).is_absolute()
        || !Path::new(output).is_absolute()
    {
        return Err("cache-family-run requires absolute input/output".into());
    }
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(
        Path::new(path),
        1024 * 1024,
    )?;
    let input: contract::Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    if !std::fs::symlink_metadata(Path::new(output).parent().ok_or("cache matrix parent")?)?
        .is_dir()
    {
        return Err("cache matrix output parent must be regular directory".into());
    }
    std::fs::create_dir(output)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let terminal_deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(input.execution_seconds);
    let result = pipeline::execute(&input, Path::new(output), &cancel);
    let finish = interrupt.finish();
    let mut value=result.unwrap_or_else(|_|serde_json::json!({"schema_version":1,"status":"failed","reason":"owned_plan_or_matrix_execution_refused"}));
    value["request_sha256"] = serde_json::json!(hash(&bytes));
    finalize(
        &mut value,
        cancel.is_cancelled(),
        std::time::Instant::now() >= terminal_deadline,
        finish.is_ok(),
    );
    publish(&Path::new(output).join("cache-family-matrix.json"), &value)?;
    finish?;
    if value["status"] != "completed" {
        return Err("cache family producer incomplete; partial evidence retained".into());
    }
    Ok(())
}

fn finalize(value: &mut Value, cancelled: bool, expired: bool, finish_ok: bool) {
    if cancelled || expired || !finish_ok {
        value["status"] = serde_json::json!("incomplete");
        value["terminal_refusal"] = serde_json::json!({"cancelled":cancelled,"deadline_expired":expired,"interrupt_finish_failed":!finish_ok});
    }
}
