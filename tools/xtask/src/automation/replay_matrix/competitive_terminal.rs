//! Final receipt admission for direct competitive workers; observations stay intact.
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::time::Instant;
pub(super) fn finalize(
    summary: &mut Value,
    finished: bool,
    cancellation: &Cancellation,
    deadline: Instant,
) -> DynResult<()> {
    let reason = if cancellation.is_cancelled() {
        Some("competitive terminal cancellation")
    } else if !finished {
        Some("competitive interrupt finalization failed")
    } else if Instant::now() >= deadline {
        Some("competitive terminal deadline exhausted")
    } else {
        None
    };
    summary["terminal_error"] = reason.map_or(Value::Null, |value| json!(value));
    summary["terminal_complete"] = json!(reason.is_none() && summary["error"].is_null());
    if let Some(reason) = reason {
        summary["completed"] = json!(false);
        summary["passed"] = json!(false);
        if summary["error"].is_null() {
            summary["error"] = json!(reason);
        }
        return Err("competitive terminal refusal; observed evidence retained".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "competitive_terminal_tests.rs"]
mod tests;
