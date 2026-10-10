use super::{config::Source, edit_loop, projections, prompt_budget};
use crate::DynResult;
use serde_json::{Value, json};

pub(super) fn normalize(
    source: &Source,
    tier: &str,
    index: usize,
    row: &Value,
    max: usize,
    target: Option<usize>,
) -> DynResult<Vec<Value>> {
    let id = format!("{}:{}:{index:05}", source.name, source.split);
    if source.adapter == "swe_smith_trajectory_loop" {
        if tier != "coding-loop" {
            return Err("trajectory loop adapter requires coding-loop tier".into());
        }
        let Some(projected) = edit_loop::project(row) else {
            return Ok(vec![]);
        };
        let count = projected.prompts.len();
        return projected
            .prompts
            .iter()
            .enumerate()
            .map(|(turn, prompt)| {
                let mut result = base(
                    source,
                    tier,
                    &format!("{id}:turn-{:02}", turn + 1),
                    &projected.group,
                    prompt_budget(prompt, max, target)?,
                    if turn == 0 {
                        projected.expected.clone()
                    } else {
                        Value::Null
                    },
                );
                result["family"] = json!("coding_edit_loop");
                let mut metadata = metadata(source, projected.metadata.clone(), target);
                metadata["routing_hint"] = json!("ngram");
                metadata["benchmark_shape"] = json!("repeated_edit_loop");
                metadata["original_family"] = json!(source.family);
                metadata["loop_turn"] = json!(turn + 1);
                metadata["loop_turns"] = json!(count);
                result["metadata"] = metadata;
                Ok(result)
            })
            .collect();
    }
    if tier == "coding-loop" {
        return Err("coding-loop tier requires repeated-edit adapter".into());
    }
    let Some(projected) = projections::project(&source.adapter, row) else {
        return Ok(vec![]);
    };
    let mut result = base(
        source,
        tier,
        &id,
        &projected.group,
        prompt_budget(&projected.prompt, max, target)?,
        projected.expected,
    );
    result["metadata"] = metadata(source, projected.metadata, target);
    Ok(vec![result])
}
fn base(
    source: &Source,
    tier: &str,
    id: &str,
    group: &str,
    prompt: String,
    expected: Value,
) -> Value {
    json!({"id":id,"tier":tier,"family":source.family,"source":source.dataset,"source_config":source.config,"source_revision":source.revision,"split":source.split,"session_group":group,"prompt":prompt,"expected_output":expected})
}
fn metadata(source: &Source, extra: Value, target: Option<usize>) -> Value {
    let mut metadata = json!({"source_name":source.name,"adapter":source.adapter,"routing_hint":source.routing_hint});
    if let Some(target) = target {
        metadata["long_context_expanded"] = json!(true);
        metadata["long_context_target_chars"] = json!(target);
    }
    if let Some(extra) = extra.as_object() {
        metadata
            .as_object_mut()
            .expect("metadata object")
            .extend(extra.clone());
    }
    metadata
}
