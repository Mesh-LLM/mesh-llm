use super::{arm_pass::Input, manifest_preflight::Manifest};
use crate::command::DynResult;

pub(super) fn prepare(
    input: &Input,
    mut manifest: Manifest,
) -> DynResult<(serde_json::Value, Vec<usize>)> {
    let base_url = format!("http://127.0.0.1:{}/v1", input.port);
    let cell = |trajectories: Vec<super::recorded_requests::Trajectory>, concurrency: usize| {
        serde_json::json!({"trajectories":trajectories.into_iter().map(|trajectory|trajectory.original).collect::<Vec<_>>(),
            "model":"pending","base_url":base_url,"concurrency":concurrency,
            "max_output_tokens":input.max_output_tokens,"request_timeout_seconds":input.request_timeout_seconds})
    };
    let warmup = manifest
        .cohorts
        .remove("warmup")
        .ok_or("missing warmup cohort")?;
    let mut workload = cell(warmup, 1);
    workload["replay_mode"] = "all".into();
    workload["warmup_turns"] = input.requirements.warmup_turns.into();
    qualify_warmup(input, &mut workload)?;
    let mut levels = input.requirements.concurrency.clone();
    if input.pass.is_multiple_of(2) {
        levels.reverse();
    }
    let mut jobs = Vec::new();
    for level in &levels {
        let trajectories = manifest
            .cohorts
            .remove(&level.to_string())
            .ok_or("missing measured cohort")?;
        let mut measured = cell(trajectories, *level);
        measured["replay_mode"] = serde_json::to_value(input.replay_mode)?;
        if let Some(qualification) = &input.qualification {
            measured["eligibility"] = serde_json::json!({"context_tokens":qualification.minimum_context_tokens,
                "maximum_output_tokens":input.max_output_tokens,"minimum_session_prompt_tokens":qualification.minimum_session_prompt_tokens});
            measured["measured_prefix"] = serde_json::json!({"expected_prompt_tokens":qualification.prompt_tokens_by_cohort[&level.to_string()],
                "require_later_turn_reuse":true});
            if qualification.require_recurrent_restores {
                measured["minimum_recurrent_restored_tokens"] =
                    qualification.minimum_session_prompt_tokens.max(1).into();
            }
        }
        jobs.push(serde_json::json!({"workload":measured,"requests_output":input.output.join(format!("c-{level}-requests.jsonl")),
            "summary_output":input.output.join(format!("c-{level}.json"))}));
    }
    workload["following_cells"] = jobs.into();
    Ok((workload, levels))
}

fn qualify_warmup(input: &Input, workload: &mut serde_json::Value) -> DynResult<()> {
    let Some(qualification) = &input.qualification else {
        return Ok(());
    };
    if input.external.is_some() {
        return Err("runtime context qualification currently requires mesh arms".into());
    }
    workload["model_pin"] = serde_json::json!({"sha256":qualification.model_sha256,
        "minimum_context_tokens":qualification.minimum_context_tokens,"output":input.output.join("model-identity.json")});
    workload["runtime_context"] = serde_json::json!({"required_tokens":qualification.minimum_context_tokens,"output":input.output.join("runtime.json")});
    if qualification
        .prompt_tokens_by_cohort
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        != input
            .requirements
            .concurrency
            .iter()
            .map(ToString::to_string)
            .collect()
    {
        return Err("qualification must cover every measured concurrency cohort".into());
    }
    Ok(())
}
