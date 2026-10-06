//! Terminal admission for owned radix matrix and request receipts.
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::time::Instant;

pub(super) fn finalize(
    receipt: &mut Value,
    finished: DynResult<()>,
    cancellation: &Cancellation,
    deadline: Instant,
) -> DynResult<()> {
    let terminal = if cancellation.is_cancelled() {
        Some("radix terminal cancellation")
    } else if finished.is_err() {
        Some("radix interrupt finalization failed")
    } else if Instant::now() >= deadline {
        Some("radix terminal deadline expired")
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
        Err("radix terminal admission failed; partial observations retained".into())
    }
}

#[cfg(test)]
#[path = "radix_terminal_tests.rs"]
mod tests;
