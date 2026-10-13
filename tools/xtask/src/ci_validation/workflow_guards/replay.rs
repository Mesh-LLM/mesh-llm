//! Nightly replay gates bind each model and evidence handoff explicitly.
use super::{Node, field};
use crate::command::DynResult;
use std::collections::BTreeMap;

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let document = workflows
        .get("agentic-replay-nightly.yml")
        .ok_or("missing replay workflow")?;
    let job = document
        .get("jobs")
        .and_then(|jobs| jobs.get("replay"))
        .ok_or("missing replay job")?;
    let Some(Node::Seq(steps)) = job.get("steps") else {
        return Err("missing replay steps".into());
    };
    model_bindings(steps)?;
    history_admission(steps)?;
    evidence_publication(steps)
}
fn by_id<'a>(steps: &'a [Node], id: &str) -> DynResult<(usize, &'a Node)> {
    steps
        .iter()
        .enumerate()
        .find(|(_, step)| field(step, "id") == Some(id))
        .ok_or_else(|| format!("missing replay step {id}").into())
}
fn model_bindings(steps: &[Node]) -> DynResult<()> {
    let (input_index, input) = by_id(steps, "inputs")?;
    require(
        field(input, "run"),
        "REPLAY_MODELS_LOCAL_FILE",
        "verified local model mapping",
    )?;
    for (id, family, model) in [
        ("model_0", "granite-3.1-2b", "$GRANITE_2B_MODEL_FILE"),
        ("model_1", "granite-3.1-3b-a800m", "$GRANITE_MOE_MODEL_FILE"),
        ("model_2", "qwen3.5-4b", "$QWEN_MODEL_FILE"),
    ] {
        let (index, step) = by_id(steps, id)?;
        if index <= input_index {
            return Err("replay model precedes pinned inputs".into());
        }
        let command = field(step, "run").ok_or("missing replay model command")?;
        require(
            Some(command),
            "cargo xtool automation replay-matrix run-family",
            id,
        )?;
        require(Some(command), &format!("--run-family {family}"), id)?;
        require(Some(command), &format!("--model-file \"{model}\""), id)?;
        require(Some(command), "--reader \"$REPLAY_READER\"", id)?;
    }
    Ok(())
}
fn history_admission(steps: &[Node]) -> DynResult<()> {
    let (_, history) = by_id(steps, "history")?;
    for condition in [
        "!cancelled()",
        "steps.benchmark.outcome == 'success'",
        "steps.baseline.outcome == 'success'",
    ] {
        require(field(history, "if"), condition, "history admission")?;
    }
    require(
        field(history, "run"),
        "cargo xtool automation replay-matrix history",
        "history owner",
    )?;
    require(
        field(history, "run"),
        "--github-output \"$GITHUB_OUTPUT\"",
        "history repair decision",
    )?;
    let (_, baseline) = by_id(steps, "baseline")?;
    super::handoffs::command(
        baseline,
        &[
            "cargo",
            "xtool",
            "automation",
            "replay-matrix",
            "history-fetch",
        ],
        &[
            ("--dataset-repo", "\"$DATASET_REPO\""),
            ("--output", "\"$HISTORY_LOCAL\""),
        ],
    )?;
    if field(baseline, "run").is_some_and(|body| body.contains("HF_TOKEN")) {
        return Err("anonymous history must not acquire publisher credentials".into());
    }
    if baseline
        .get("env")
        .and_then(|env| env.get("HF_TOKEN"))
        .is_some()
    {
        return Err("anonymous history receives publisher token".into());
    }
    let (_, repair) = by_id(steps, "repair")?;
    for condition in [
        "!cancelled()",
        "steps.history.outcome == 'failure'",
        "steps.history.outputs.repair_required == 'true'",
    ] {
        require(field(repair, "if"), condition, "repair admission")?;
    }
    Ok(())
}
fn evidence_publication(steps: &[Node]) -> DynResult<()> {
    let (_, upload) = by_id(steps, "upload_evidence")?;
    require(field(upload, "if"), "cancelled()", "cancelled-run evidence")?;
    if field(upload, "if").is_some_and(|condition| condition.contains("!cancelled()")) {
        return Err("cancelled replay evidence suppressed".into());
    }
    let publish = steps
        .iter()
        .find(|step| field(step, "name") == Some("Publish immutable run shard and card"))
        .ok_or("missing replay publication")?;
    require(
        field(publish, "if"),
        "steps.history.outcome == 'success'",
        "publication admission",
    )?;
    require(
        field(publish, "run"),
        "--kind shard",
        "immutable shard publication",
    )?;
    require(field(publish, "run"), "--kind card", "card publication")?;
    Ok(())
}
fn require(actual: Option<&str>, expected: &str, context: &str) -> DynResult<()> {
    if actual.is_some_and(|actual| {
        actual
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
            .contains(expected)
    }) {
        Ok(())
    } else {
        Err(format!("replay {context} must bind {expected}").into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const WORKFLOW: &str =
        include_str!("../../../../../.github/workflows/agentic-replay-nightly.yml");
    fn validate(source: &str) -> DynResult<()> {
        check(&BTreeMap::from([(
            "agentic-replay-nightly.yml".into(),
            super::super::workflow_yaml::parse(source)?,
        )]))
    }
    #[test]
    fn current_replay_handoffs_are_admitted() {
        validate(WORKFLOW).unwrap();
    }
    #[test]
    fn model_swap_and_missing_input_identity_are_rejected() {
        for (from, to) in [
            (
                "--model-file \"$GRANITE_2B_MODEL_FILE\"",
                "--model-file \"$GRANITE_MOE_MODEL_FILE\"",
            ),
            ("id: inputs", "id: obsolete_inputs"),
            ("--reader \"$REPLAY_READER\"", "--reader unspecified"),
        ] {
            assert!(validate(&WORKFLOW.replace(from, to)).is_err(), "{from}");
        }
    }
    #[test]
    fn history_fetch_requires_bound_dataset_output_and_anonymous_command() {
        for (from, to) in [
            (
                "--dataset-repo \"$DATASET_REPO\" --output \"$HISTORY_LOCAL\"",
                "--dataset-repo foreign/dataset --output \"$HISTORY_LOCAL\"",
            ),
            (
                "--dataset-repo \"$DATASET_REPO\" --output \"$HISTORY_LOCAL\"",
                "--dataset-repo \"$DATASET_REPO\" --output /tmp/unbound",
            ),
            (
                "cargo xtool automation replay-matrix history-fetch",
                "HF_TOKEN=publisher cargo xtool automation replay-matrix history-fetch",
            ),
        ] {
            assert!(WORKFLOW.contains(from));
            assert!(validate(&WORKFLOW.replace(from, to)).is_err(), "{from}");
        }
    }
    #[test]
    fn repair_baseline_and_cancelled_evidence_cannot_fail_open() {
        for (from, to) in [
            ("steps.baseline.outcome == 'success'", "true"),
            ("steps.history.outputs.repair_required == 'true'", "true"),
            (
                "cargo xtool automation replay-matrix history-fetch",
                "echo history-fetch",
            ),
            ("id: upload_evidence", "id: obsolete_upload"),
        ] {
            assert!(validate(&WORKFLOW.replace(from, to)).is_err(), "{from}");
        }
    }
}
