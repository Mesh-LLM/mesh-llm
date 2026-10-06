//! Final admission of retained restart observations after interrupt ownership closes.
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::time::Instant;

pub(super) fn wall_seconds() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

pub(super) fn finalize(
    receipt: &mut Value,
    finished: DynResult<()>,
    cancellation: &Cancellation,
    deadline: Instant,
) -> DynResult<()> {
    let terminal = if cancellation.is_cancelled() {
        Some("restart terminal cancellation")
    } else if finished.is_err() {
        Some("restart interrupt finalization failed")
    } else if Instant::now() >= deadline {
        Some("restart terminal deadline expired")
    } else {
        None
    };
    receipt["terminal_error"] = terminal.map_or(Value::Null, |reason| json!(reason));
    if receipt["error"].is_null()
        && let Some(reason) = terminal
    {
        receipt["error"] = json!(reason);
    }
    let complete = receipt["error"].is_null();
    receipt["terminal_complete"] = json!(complete);
    if complete {
        Ok(())
    } else {
        Err("restart terminal admission failed; partial observations retained".into())
    }
}

#[cfg(test)]
#[path = "kv_terminal_tests.rs"]
mod tests;
