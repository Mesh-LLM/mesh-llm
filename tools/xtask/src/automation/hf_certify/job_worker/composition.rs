//! Already-converted composition after actual bootstrap; no SafeTensors conversion claim.
use super::{bootstrap_phase, certification_window, check, contract, runner};
use crate::{
    automation::hf_mtp_compose::job_phase::Context, command::DynResult, process::Cancellation,
};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
pub(super) fn execute(
    input: &contract::CompositionInput,
    root: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<Value> {
    check(deadline, cancel)?;
    runner(&input.runner, deadline, cancel)?;
    let observed = bootstrap_phase(&input.bootstrap, root, deadline, cancel, evidence)?;
    let (phase_deadline, _) = certification_window(deadline, cancel)?;
    let phase = root.join("composition");
    std::fs::create_dir(&phase)?;
    evidence["composition"] = json!({"status":"FAILED","source_unchanged":false,"error":null});
    let context = Context {
        binary: &observed.binary,
        mesh_revision: &observed.mesh_commit,
        deadline: phase_deadline,
        cancellation: cancel,
    };
    let plan = input
        .composition
        .execute(&phase, &context, &mut evidence["composition"])?;
    runner(&input.runner, phase_deadline, cancel)?;
    check(phase_deadline, cancel)?;
    evidence["composition"]["status"] = json!("COMPOSE_VALIDATED");
    evidence["composition"]["pending_plan"] = plan.clone();
    Ok(plan)
}
