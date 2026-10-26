//! Deterministic next source-bound coverage target and refused-row diagnostics.
use super::{DynResult, RUNNABLE, Value};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

#[derive(Serialize)]
pub(super) struct BoundaryTarget {
    llama_model: String,
    family: String,
    source_file: PathBuf,
    status: &'static str,
}

/// Called only after the owning source, manifest and pin admission succeeds.
pub(super) fn next_target(rows: &[Value], native: &Path) -> DynResult<Option<BoundaryTarget>> {
    let mut selected: Option<(&str, &str)> = None;
    for row in rows {
        super::process::check()?;
        if row["status"].as_str() != Some("needs_boundary_registration")
            || row["boundary_registered"].as_bool() != Some(false)
            || !no_unsupported_reason(row)
        {
            continue;
        }
        let model = row["llama_model"]
            .as_str()
            .ok_or("boundary target model must be string")?;
        let family = row["family"]
            .as_str()
            .ok_or("boundary target family must be string")?;
        let candidate = (model, family);
        if selected.is_none_or(|previous| candidate < previous) {
            selected = Some(candidate);
        }
    }
    Ok(selected.map(|(model, family)| BoundaryTarget {
        llama_model: model.into(),
        family: family.into(),
        source_file: native.join("src/models").join(format!("{model}.cpp")),
        status: "needs_boundary_registration",
    }))
}

fn no_unsupported_reason(row: &Value) -> bool {
    row.get("unsupported_reason")
        .is_none_or(|reason| reason.is_null() || reason.as_str() == Some(""))
}

/// A diagnostic for refused runnable rows, never an alternative admission path.
pub(super) fn pending_reclassification(
    rows: &[Value],
    sources: &BTreeSet<String>,
    boundaries: &BTreeSet<String>,
) -> BTreeSet<String> {
    rows.iter()
        .filter_map(|row| {
            let model = row["llama_model"].as_str()?;
            let status = row.get("status").map_or(Some("candidate"), Value::as_str)?;
            (sources.contains(model)
                && !boundaries.contains(model)
                && RUNNABLE.contains(&status)
                && no_unsupported_reason(row))
            .then(|| model.into())
        })
        .collect()
}
