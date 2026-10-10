use super::{
    run_budget::Budget,
    run_workload::{Build, Input},
};
use crate::command::DynResult;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

pub(super) fn measure(
    root: Option<&Path>,
    input: &Input,
    document: &mut serde_json::Value,
    paths: (&Path, &Path),
    completed: &BTreeSet<(u32, String)>,
    qualification: &BTreeMap<String, serde_json::Value>,
    budget: &Budget,
) -> DynResult<bool> {
    let mut passed = document["results"]
        .as_array()
        .ok_or("invalid results")?
        .iter()
        .all(|result| {
            result["passed"] == true
                && result["cells"].as_array().is_some_and(|cells| {
                    cells
                        .iter()
                        .all(|cell| cell["acceptance"]["passed"] == true)
                })
        });
    for pass in 1..=input.passes {
        let mut arms = input.builds.iter().collect::<Vec<_>>();
        if pass.is_multiple_of(2) {
            arms.reverse();
        }
        for build in arms {
            if completed.contains(&(pass, build.label().to_owned())) {
                continue;
            }
            super::run_transport::verify_build(build, budget)?;
            passed &= arm(
                root,
                (input, build),
                document,
                paths,
                pass,
                qualification,
                budget,
            )?;
        }
    }
    Ok(passed)
}

fn arm(
    root: Option<&Path>,
    workload: (&Input, &Build),
    document: &mut serde_json::Value,
    paths: (&Path, &Path),
    pass: u32,
    qualification: &BTreeMap<String, serde_json::Value>,
    budget: &Budget,
) -> DynResult<bool> {
    let (input, build) = workload;
    let directory = input
        .output
        .join(format!("data/pass-{pass}/{}", build.label()));
    let mut request = super::run_transport::request(workload, paths.0, &directory, budget)?;
    request["label"] = build.label().into();
    request["ref"] = build.reference().into();
    request["commit"] = build.commit().into();
    request["pass"] = pass.into();
    if input.context_qualification.is_mesh() {
        request["qualification"] = serde_json::json!({"model_sha256":input.model_sha256,
            "minimum_context_tokens":input.minimum_context_tokens,"minimum_session_prompt_tokens":input.minimum_session_prompt_tokens,
            "require_recurrent_restores":input.require_recurrent_restores,"prompt_tokens_by_cohort":qualification[build.label()]});
    }
    let path = input
        .output
        .join(format!("arm-{}-{pass}.json", build.label()));
    let result = input
        .output
        .join(format!("result-{}-{pass}.json", build.label()));
    crate::command::write_json_file(&path, &request)?;
    let execution = super::run_transport::invoke(root, "arm-pass", &path, &result);
    if !result.try_exists()? {
        execution?;
        return Err("arm execution produced no result".into());
    }
    let evidence: serde_json::Value = serde_json::from_slice(&std::fs::read(&result)?)?;
    let complete = evidence["cells"]
        .as_array()
        .is_some_and(|cells| cells.len() == input.requirements.concurrency.len());
    let acceptance_failed = evidence["acceptance_failed"] == true;
    document["order"]
        .as_array_mut()
        .ok_or("invalid run order")?
        .push(serde_json::json!({"pass":pass,"label":build.label()}));
    document["results"]
        .as_array_mut()
        .ok_or("invalid run results")?
        .push(evidence);
    super::run_snapshot::write(paths.1, document)?;
    if execution.is_err() && complete && acceptance_failed {
        Ok(false)
    } else {
        execution?;
        Ok(true)
    }
}
