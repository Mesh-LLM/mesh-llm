use super::{run_budget::Budget, run_workload::Input};
use crate::command::DynResult;
use std::{collections::BTreeMap, path::Path};

pub(super) fn not_requested() -> serde_json::Value {
    serde_json::json!({"status":"not_requested","reason":"captured profile omits Mesh runtime/context qualification"})
}

pub(super) fn qualify(
    root: Option<&Path>,
    input: &Input,
    document: &mut serde_json::Value,
    manifest: &Path,
    run_path: &Path,
    budget: &Budget,
) -> DynResult<BTreeMap<String, serde_json::Value>> {
    let mut qualification = BTreeMap::new();
    if !input.context_qualification.is_mesh() {
        return Ok(qualification);
    }
    for build in &input.builds {
        if document["context_preflight"][build.label()]["passed"] == true {
            qualification.insert(
                build.label().to_owned(),
                tokens(&document["context_preflight"][build.label()])?,
            );
            continue;
        }
        let directory = input.output.join("context-preflight").join(build.label());
        let result = directory.join("eligibility.json");
        let request = super::run_transport::request((input, build), manifest, &directory, budget)?;
        let path = input
            .output
            .join(format!("preflight-{}.json", build.label()));
        crate::command::write_json_file(&path, &request)?;
        let execution = super::run_transport::invoke(root, "context-preflight", &path, &result);
        if result.try_exists()? {
            let evidence: serde_json::Value = serde_json::from_slice(&std::fs::read(&result)?)?;
            qualification.insert(build.label().to_owned(), tokens(&evidence)?);
            document["context_preflight"][build.label()] = evidence;
        }
        super::run_snapshot::write(run_path, document)?;
        execution?;
    }
    Ok(qualification)
}

fn tokens(evidence: &serde_json::Value) -> DynResult<serde_json::Value> {
    let mut tokens = evidence["prompt_tokens_by_cohort"].clone();
    tokens
        .as_object_mut()
        .ok_or("missing preflight token map")?
        .remove("warmup");
    Ok(tokens)
}
