use crate::command::DynResult;
use std::{collections::BTreeSet, path::Path};

pub(super) fn restore(
    current: &mut serde_json::Value,
    paths: (&Path, &Path),
    passes: u32,
) -> DynResult<BTreeSet<(u32, String)>> {
    let previous: serde_json::Value = serde_json::from_slice(&std::fs::read(paths.0)?)?;
    for field in ["plan_sha256", "config", "builds"] {
        if previous[field] != current[field] {
            return Err(format!("cannot resume: {field} differs").into());
        }
    }
    if previous["inputs"]["manifest_sha256"] != current["inputs"]["manifest_sha256"] {
        return Err("cannot resume: manifest identity differs".into());
    }
    for field in ["kind", "dataset", "dataset_file", "dataset_file_sha256"] {
        if previous["inputs"][field] != current["inputs"][field] {
            return Err("cannot resume: retained input provenance differs".into());
        }
    }
    let digest = crate::product::digest::file_sha256(paths.1).map_err(|error| error.error)?;
    if current["inputs"]["manifest_sha256"].as_str() != Some(&digest) {
        return Err("cannot resume: retained manifest bytes changed".into());
    }
    let completed = super::run_resume_pass::verify(&previous, current, paths, passes)?;
    super::run_resume_context::verify(&previous, current, paths.0)?;
    current["context_preflight"] = previous["context_preflight"].clone();
    current["results"] = previous["results"].clone();
    current["order"] = previous["order"].clone();
    Ok(completed)
}
