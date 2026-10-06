//! Native source checkout and CPU Just bootstrap; acquisition/publication are separate.
pub(super) mod contract;
pub(super) mod execution;
use super::admission;
use crate::{automation::command_interrupt::Interrupt, command::DynResult};
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [a, path, b, output] = args else {
        return Err("hf-certify bootstrap --input FILE --output-directory FRESH_DIRECTORY".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("bootstrap closed flags".into());
    }
    let input: contract::Input =
        serde_json::from_slice(&admission::read(Path::new(path), 262144)?)?;
    input.validate()?;
    let requested = std::path::absolute(output)?;
    let parent = requested
        .parent()
        .ok_or("bootstrap output parent")?
        .canonicalize()?;
    let root = parent.join(requested.file_name().ok_or("bootstrap output leaf")?);
    std::fs::create_dir(&root)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_seconds);
    let mut rows = Vec::new();
    let result = execution::execute(&input, &root, deadline, &cancel, &mut rows);
    let finished = interrupt.finish();
    let terminal = terminal_decision(deadline, &cancel);
    let complete = result.is_ok() && finished.is_ok() && terminal.is_ok();
    let error = result
        .err()
        .map(|e| e.to_string())
        .into_iter()
        .chain(finished.err().map(|e| e.to_string()))
        .chain(terminal.err().map(|e| e.to_string()))
        .collect::<Vec<_>>()
        .join("; ");
    admission::publish(
        &root.join("bootstrap.json"),
        &json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(&input)?),"status":if complete{"BOOTSTRAP_COMPLETED"}else{"FAILED"},"declared":input,"phases":rows,"error":if error.is_empty(){None}else{Some(error)},"image_observed":false,"cpu_plan_and_rate_observed":false,"acquisition_completed":false,"publication_completed":false,"scope":"selected local source/tool bytes and fresh CPU Just output; container delivery and downstream native validation remain separate"}),
    )?;
    if complete {
        Ok(())
    } else {
        Err("bootstrap failed; source/logs/partial phase evidence retained".into())
    }
}
fn terminal_decision(deadline: Instant, cancel: &crate::process::Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() {
        return Err("bootstrap terminal cancellation".into());
    }
    if Instant::now() >= deadline {
        return Err("bootstrap terminal deadline expired".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "bootstrap/tests.rs"]
mod tests;
