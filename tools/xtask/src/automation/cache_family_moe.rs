//! Native supplied-input MoE expert presence/cache smoke; no research interpreter.
#[path = "cache_family_moe/contract.rs"]
mod contract;
#[path = "cache_family_moe/evidence.rs"]
mod evidence;
#[path = "cache_family_moe/execution.rs"]
mod execution;
#[path = "cache_family_moe/processes.rs"]
mod processes;
#[path = "cache_family_moe/publication.rs"]
mod publication;
#[path = "cache_family_moe/render.rs"]
mod render;
#[cfg(test)]
#[path = "cache_family_moe/tests.rs"]
mod tests;
use crate::command::DynResult;
use serde_json::Value;
use std::{
    io::Write as _,
    path::Path,
    time::{Duration, Instant},
};
fn publish(path: &Path, value: &Value) -> DynResult<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > 32 * 1024 * 1024 {
        return Err("MoE evidence exceeds32MiB".into());
    }
    crate::automation::waiting_prefix::adaptive_identity::fresh(path, &bytes)
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(
            std::io::stdout().lock(),
            "cargo xtool automation cache-family-moe --input ABS_JSON --output ABS_FRESH_DIRECTORY"
        )?;
        return Ok(());
    }
    let [a, path, b, output] = args else {
        return Err("cache-family-moe requires ordered input/output".into());
    };
    if a != "--input"
        || b != "--output"
        || !Path::new(path).is_absolute()
        || !Path::new(output).is_absolute()
    {
        return Err("MoE paths must be absolute".into());
    }
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(
        Path::new(path),
        1024 * 1024,
    )?;
    let input: contract::Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    if !std::fs::symlink_metadata(Path::new(output).parent().ok_or("MoE output parent")?)?.is_dir()
    {
        return Err("MoE output parent must be regular directory".into());
    }
    std::fs::create_dir(output)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.execution_seconds);
    let result = execution::execute(&input, Path::new(output), &cancel);
    let mut receipt=result.unwrap_or_else(|_|serde_json::json!({"schema_version":1,"status":"failed","reason":"owned_MoE_execution_failed"}));
    {
        use sha2::{Digest, Sha256};
        receipt["request_sha256"] = serde_json::json!(hex::encode(Sha256::digest(&bytes)));
    }
    finalize(
        &mut receipt,
        cancel.is_cancelled(),
        Instant::now() >= deadline,
        true,
    );
    let reports = publication::publish(
        Path::new(output),
        &mut receipt,
        &mut || (cancel.is_cancelled(), Instant::now() >= deadline),
        &mut |_| {},
    );
    let finish = interrupt.finish();
    let mut reports = reports?;
    publication::finish(
        &mut reports,
        &mut receipt,
        cancel.is_cancelled(),
        Instant::now() >= deadline,
        finish.is_ok(),
    )?;
    finish?;
    if receipt["status"] != "completed" {
        return Err("MoE expert smoke incomplete; partial evidence retained".into());
    }
    Ok(())
}
fn finalize(receipt: &mut Value, cancelled: bool, expired: bool, finish_ok: bool) {
    if cancelled || expired || !finish_ok {
        receipt["status"] = serde_json::json!("incomplete");
        receipt["terminal_refusal"] = serde_json::json!({"cancelled":cancelled,"deadline_expired":expired,"interrupt_finish_failed":!finish_ok});
    }
}
