//! Replay shared-cache bindings and source-owned repair step contracts.
use super::{Node, field, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let document = workflows
        .get("agentic-replay-nightly.yml")
        .ok_or("missing replay workflow")?;
    let replay = h::job(document, "replay")?;
    let steps = h::steps(replay)?;
    shared_cache(replay, steps)?;
    repair(steps)
}
fn statements(step: &Node) -> DynResult<Vec<String>> {
    Ok(field(step, "run")
        .ok_or("missing replay shell scalar")?
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("\\\n", " ")
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect())
}
fn direct(lines: &[String], statement: &str) -> DynResult<()> {
    if lines
        .iter()
        .filter(|line| line.as_str() == statement)
        .count()
        != 1
    {
        return Err(format!("replay requires one direct binding: {statement}").into());
    }
    Ok(())
}
fn shared_cache(replay: &Node, steps: &[Node]) -> DynResult<()> {
    let env = h::member(replay, "env")?;
    h::binding(env, "HF_HUB_OFFLINE", "0")?;
    h::binding(env, "HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")?;
    let (tool_index, toolchain) = h::step(steps, "name", "Verify runner toolchain")?;
    let (input_index, inputs) = h::step(steps, "id", "inputs")?;
    h::before(tool_index, input_index)?;
    let lines = statements(toolchain)?;
    for statement in [
        "export HF_HOME=\"$HF_CACHE\"",
        "export HF_HUB_CACHE=\"$HF_CACHE/hub\"",
        "if [[ \"${HF_CACHE:-}\" != \"/Users/lab/models/huggingface\" ]]; then",
        "if [[ ! -w \"$HF_CACHE\" || ! -w \"$HF_CACHE/hub\" ]]; then",
        "if [[ \"$missing\" == \"1\" ]]; then exit 1; fi",
        "echo \"HF_HOME=$HF_HOME\"",
        "echo \"HF_HUB_CACHE=$HF_HUB_CACHE\"",
        "echo \"SCCACHE_SERVER_UDS=$RUNNER_TEMP/agentic-replay-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.sock\"",
    ] {
        direct(&lines, statement)?;
    }
    if lines
        .iter()
        .filter(|line| line.starts_with("export HF_HOME="))
        .count()
        != 1
        || lines
            .iter()
            .filter(|line| line.starts_with("export HF_HUB_CACHE="))
            .count()
            != 1
        || !lines.iter().any(|line| line == "} >> \"$GITHUB_ENV\"")
    {
        return Err("shared cache exports must bind once and reach GITHUB_ENV".into());
    }
    downloads(inputs)?;
    let (_, baseline) = h::step(steps, "id", "baseline")?;
    if baseline.get("env").is_some() {
        return Err("anonymous history download must not acquire step-specific credentials".into());
    }
    Ok(())
}
fn downloads(inputs: &Node) -> DynResult<()> {
    let lines = statements(inputs)?;
    for line in &lines {
        let words = line.split_whitespace().collect::<Vec<_>>();
        if words.contains(&"--cache-dir") || line.contains("import hf_hub_download") {
            return Err("pinned replay downloads must reuse the approved CLI shared cache".into());
        }
    }
    for expected in [
        "model_path=$(hf download \"$repo\" \"$file\" --revision \"$revision\" --format quiet)",
        "dataset_path=$(HF_HUB_OFFLINE=0 hf download \"$dataset_repo\" \"$dataset_file\" --repo-type dataset --revision \"$dataset_revision\" --format quiet)",
    ] {
        direct(&lines, expected)?;
    }
    h::command(
        inputs,
        &[
            "cargo",
            "xtool",
            "automation",
            "replay-matrix",
            "verify-digest",
        ],
        &[
            ("--file", "\"$model_path\""),
            ("--sha256", "\"$expected_sha\""),
        ],
    )?;
    h::command(
        inputs,
        &[
            "cargo",
            "xtool",
            "automation",
            "replay-matrix",
            "verify-digest",
        ],
        &[
            ("--file", "\"$dataset_path\""),
            ("--sha256", "\"$dataset_sha\""),
        ],
    )
}
fn repair(steps: &[Node]) -> DynResult<()> {
    let (_, step) = h::step(steps, "id", "repair")?;
    h::condition(
        step,
        "${{ !cancelled() && steps.history.outcome == 'failure' && steps.history.outputs.repair_required == 'true' }}",
    )?;
    let env = h::member(step, "env")?;
    for (key, expected) in [
        (
            "REPLAY_AGENT_PROVIDER",
            "${{ vars.LLAMA_CANARY_GOOSE_PROVIDER || 'zai_coding_plan' }}",
        ),
        (
            "REPLAY_AGENT_MODEL",
            "${{ vars.LLAMA_CANARY_GOOSE_MODEL || 'glm-5.3-flash' }}",
        ),
        ("HISTORY_LOCAL", "${{ runner.temp }}/agentic-replay-history"),
    ] {
        h::binding(env, key, expected)?;
    }
    let lines = statements(step)?;
    direct(
        &lines,
        "if scripts/agentic-replay-repair.sh \"$RUNNER_TEMP/agentic-replay-artifacts\" 2>&1 | tee \"$RUNNER_TEMP/agentic-replay-artifacts/repair.log\"; then",
    )?;
    direct(&lines, "echo \"prepared=true\" >> \"$GITHUB_OUTPUT\"")?;
    direct(&lines, "echo \"prepared=false\" >> \"$GITHUB_OUTPUT\"")?;
    if lines.first().map(String::as_str) != Some("set -euo pipefail") {
        return Err("repair logging must preserve pipeline failure semantics".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "replay_environment_tests.rs"]
mod tests;
