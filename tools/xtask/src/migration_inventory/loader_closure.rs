use super::ledger::InvocationLedger;
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
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

const FAMILY_LOADER: &str = "scripts/llama-canary-family-evidence.py";
const FAMILY_BINDINGS: [(&str, &str, &str); 2] = [
    (
        "scripts/llama-canary-family-evidence.py#dynamic-import:3c2c97de0b0aba79:1",
        "_MEMORY_SPEC = importlib.util.spec_from_file_location(",
        "scripts/lib/canary_family_memory.py",
    ),
    (
        "scripts/llama-canary-family-evidence.py#dynamic-import:e9b2f3ad7f9e3b53:1",
        "spec = importlib.util.spec_from_file_location(\"family_planner\", root / \"scripts/plan-family-battery.py\")",
        "scripts/plan-family-battery.py",
    ),
];

pub(super) fn check_family_canary_loaders(root: &Path, ledger: &InvocationLedger) -> DynResult<()> {
    if ledger.family_canary_loaders.len() != FAMILY_BINDINGS.len() {
        return Err("family canary loader: missing family loader invocation contract".into());
    }
    let source = fs::read_to_string(root.join(FAMILY_LOADER))?;
    let mut seen = BTreeSet::new();
    for record in &ledger.family_canary_loaders {
        let Some((_, binding, target)) = FAMILY_BINDINGS
            .iter()
            .find(|(id, _, _)| *id == record.edge_id)
        else {
            return Err("family canary loader: unreviewed loader record".into());
        };
        if !seen.insert(record.edge_id.as_str())
            || record.caller != FAMILY_LOADER
            || record.source_block != *binding
            || record.target != *target
            || record.target_sha256.len() != 64
            || !record
                .target_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || record.resolution.is_empty()
            || record.invocation.is_empty()
            || record.status_streams_effects.is_empty()
            || record.descendant_processes.is_empty()
            || record.replacement_owner.is_empty()
            || record.deletion_phase != 5
            || record.deletion_condition.is_empty()
            || !ledger.source_verified_edges.iter().any(|edge| {
                edge.id == record.edge_id && edge.target == *target && edge.replacement_task == 24
            })
        {
            return Err("family canary loader: incomplete or changed ownership".into());
        }
        let digest = hex::encode(Sha256::digest(fs::read(root.join(target))?));
        if digest != record.target_sha256 {
            return Err(format!("family canary loader: changed target bytes {target}").into());
        }
        if source
            .lines()
            .filter(|line| line.trim() == *binding)
            .count()
            != 1
        {
            return Err("family canary loader: changed or unowned dynamic import".into());
        }
    }
    if seen.len() != FAMILY_BINDINGS.len()
        || source.matches("spec_from_file_location(").count() != 2
        || source.matches("exec_module(").count() != 2
        || !source.contains("\"canary_family_memory\", Path(__file__).resolve().parent / \"lib/canary_family_memory.py\")")
        || source.lines().filter(|line| line.trim() == "_MEMORY_SPEC.loader.exec_module(MEMORY)").count() != 1
        || source.lines().filter(|line| line.trim() == "spec.loader.exec_module(module)").count() != 1
    {
        return Err("family canary loader: changed or unowned dynamic import".into());
    }
    Ok(())
}
