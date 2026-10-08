//! Final ownership admission for direct mixed worker/cell receipts.
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::time::Instant;
#[derive(Clone, Copy)]
pub(super) enum Kind {
    Worker,
    Cell,
}
pub(super) fn finalize(
    output: &mut Value,
    finished: DynResult<()>,
    cancellation: &Cancellation,
    deadline: Instant,
    kind: Kind,
) -> DynResult<()> {
    let terminal = if cancellation.is_cancelled() {
        Some("mixed terminal cancellation")
    } else if finished.is_err() {
        Some("mixed interrupt finalization failed")
    } else if Instant::now() >= deadline {
        Some("mixed terminal deadline exhausted")
    } else {
        None
    };
    output["terminal_error"] = terminal.map_or(Value::Null, |reason| json!(reason));
    if output["error"].is_null()
        && let Some(reason) = terminal
    {
        output["error"] = json!(reason);
    }
    let complete = output["error"].is_null()
        && match kind {
            Kind::Worker => true,
            Kind::Cell => output["status"] == "mixed_cell_admitted",
        };
    output["terminal_complete"] = json!(complete);
    if !complete {
        output["status"] = json!(match kind {
            Kind::Worker => "mixed_worker_failed",
            Kind::Cell => "mixed_cell_failed",
        });
        return Err("mixed final admission failed; partial observations retained".into());
    }
    if matches!(kind, Kind::Worker) {
        output["status"] = json!("mixed_worker_completed");
    }
    Ok(())
}
#[cfg(test)]
#[path = "mixed_terminal_tests.rs"]
mod tests;
