//! Selected-profile resume verifies raw selected IDs and the retained measured summary.
use super::{manifest_preflight::Manifest, replay_profile::Mode};
use crate::command::DynResult;
use std::path::Path;
pub(super) fn verify(
    directory: &Path,
    manifest_path: &Path,
    cells: &[serde_json::Value],
    mode: Mode,
) -> DynResult<()> {
    if mode == Mode::All {
        return Ok(());
    }
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(manifest_path)?)?;
    for cell in cells {
        let level = cell["concurrency"]
            .as_u64()
            .ok_or("missing resumed cell concurrency")?;
        if cell["replay_mode"] != serde_json::to_value(mode)? {
            return Err("cannot resume: selected cell mode differs".into());
        }
        let trajectories = manifest
            .cohorts
            .get(&level.to_string())
            .ok_or("missing resumed cohort")?;
        let records = super::history_artifacts::records(
            &directory.join(format!("c-{level}-requests.jsonl")),
        )?;
        let expected = super::replay_profile::expected(trajectories, mode)?;
        let expected_set = expected
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let observed = records
            .iter()
            .map(|record| {
                record["request_id"]
                    .as_str()
                    .ok_or("missing retained request ID")
            })
            .collect::<Result<Vec<_>, _>>()?;
        if observed.len() != expected.len()
            || observed
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                != expected_set
        {
            return Err(
                "cannot resume: selected requests are missing, duplicated or foreign".into(),
            );
        }
        let summary = super::replay_profile::summarize(
            trajectories,
            &records,
            usize::try_from(level)?,
            mode,
        )?;
        for (name, value) in summary
            .as_object()
            .ok_or("invalid recomputed selected summary")?
        {
            if cell[name] != *value {
                return Err(
                    format!("cannot resume: selected raw evidence differs at {name}").into(),
                );
            }
        }
    }
    Ok(())
}
