use super::ledger::InvocationLedger;
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const LOADER: &str = "scripts/runner-image-identity.py";
const TARGET: &str = "scripts/plan-ci.py";
const BINDING: &str = "spec = importlib.util.spec_from_file_location(\"runner_identity_planner\", root / \"scripts/plan-ci.py\")";
const EXECUTION: &str = "spec.loader.exec_module(planner)";
const EVIDENCE_BINDING: &str = "_evidence_spec = importlib.util.spec_from_file_location(\"runner_image_evidence\", Path(__file__).with_name(\"runner-image-evidence.py\"))";
const EDGE_ID: &str = "scripts/runner-image-identity.py#dynamic-import:40fc0127cbc4dd32:1";

pub(super) fn check_runner_image_planner_loader(
    root: &Path,
    ledger: &InvocationLedger,
) -> DynResult<()> {
    let record = ledger
        .runner_image_planner_loader
        .as_ref()
        .ok_or("runner image planner loader: missing invocation contract")?;
    if record.caller != LOADER
        || record.source_block != BINDING
        || record.target != TARGET
        || record.target_sha256.len() != 64
        || !record
            .target_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || record.invocation.is_empty()
        || record.status_streams_effects.is_empty()
        || record.replacement_owner.is_empty()
        || record.deletion_phase == 0
        || record.deletion_condition.is_empty()
        || !ledger
            .source_verified_edges
            .iter()
            .any(|edge| edge.id == EDGE_ID && edge.target == TARGET)
    {
        return Err("runner image planner loader: incomplete or changed ownership".into());
    }
    let source = fs::read_to_string(root.join(LOADER))?;
    if source.lines().filter(|line| line.trim() == BINDING).count() != 1
        || source
            .lines()
            .filter(|line| line.trim() == EXECUTION)
            .count()
            != 1
        || source
            .lines()
            .filter(|line| line.trim() == EVIDENCE_BINDING)
            .count()
            != 1
        || source.matches("spec_from_file_location(").count() != 2
        || source.matches("exec_module(").count() != 2
    {
        return Err("runner image planner loader: changed or unowned dynamic import".into());
    }
    let digest = hex::encode(Sha256::digest(fs::read(root.join(TARGET))?));
    if digest != record.target_sha256 {
        return Err("runner image planner loader: changed planner target bytes".into());
    }
    Ok(())
}
