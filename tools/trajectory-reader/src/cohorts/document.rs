//! Public schema-one whole-session manifest and reproducible cohort identity.
use super::{Cohorts, Selection};
use crate::DynResult;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub fn manifest(
    cohorts: Cohorts,
    selection: &Selection,
    revision: &str,
    dataset_sha256: &str,
) -> DynResult<Value> {
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("dataset revision must be an immutable lowercase 40-hex commit".into());
    }
    let mut metadata = BTreeMap::new();
    for (name, rows) in &cohorts {
        let mut counts = BTreeMap::<&str, usize>::new();
        let mut turns = BTreeMap::<&str, usize>::new();
        for row in rows {
            *counts.entry(&row.agent_framework).or_default() += 1;
            *turns.entry(&row.agent_framework).or_default() += row.assistant_turns;
        }
        let ids = rows
            .iter()
            .map(|r| r.session_id.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        metadata.insert(name,json!({"trajectory_count":rows.len(),"assistant_turns":rows.iter().map(|r|r.assistant_turns).sum::<usize>(),
            "framework_trajectories":counts,"framework_assistant_turns":turns,"session_ids_sha256":Sha256::digest(ids.as_bytes()).iter().map(|byte|format!("{byte:02x}")).collect::<String>()}));
    }
    Ok(
        json!({"schema_version":1,"metadata":{"dataset":"thoughtworks/agentic-coding-trajectories","dataset_revision":revision,
        "dataset_sha256":dataset_sha256,"cohorts":metadata,"selection":{"sources":selection.sources,"frameworks":selection.frameworks,
        "trajectories_per_framework_per_cohort":selection.trajectories_per_framework,"sessions_per_cohort":selection.sessions_per_cohort,
        "allocation":"quotient/remainder in declared framework order","algorithm_version":"balanced-md5-v2",
        "min_isl":selection.min_isl,"max_isl_exclusive":selection.max_isl_exclusive,"min_assistant_turns":selection.min_turns,
        "order":"agent_framework, md5(session_id)","whole_trajectories":true,"cohorts_disjoint":true}},"cohorts":cohorts}),
    )
}
