use super::{Digest, contract::PlanIdentity, create_file, placement::Matrix};
use crate::command::DynResult;
use std::path::Path;

pub(super) fn write(output: &Path, identity: &PlanIdentity, matrix: &Matrix) -> DynResult<()> {
    create_file(
        &output.join("scheduling-matrix.json"),
        &serde_json::to_vec_pretty(matrix)?,
    )?;
    let summary = serde_json::json!({
        "schema": 1,
        "status": "ready",
        "scope": "immutable_cache_metadata_scheduling_and_selected_consumer_contract",
        "families": matrix.include.len(),
        "matrix_jobs": matrix.include.len(),
        "controller_revision": identity.controller_revision,
        "selected_revision": identity.selected_revision,
        "plan_sha256": identity.plan_sha256,
        "source_plan_sha256": Digest::of_file(&output.join("source-plan.json"))?,
        "scheduling_matrix_sha256": Digest::of_file(&output.join("scheduling-matrix.json"))?,
        "certification": "pending",
        "selected_execution_preflight": "pending"
    });
    create_file(
        &output.join("summary.json"),
        &serde_json::to_vec_pretty(&summary)?,
    )
}
