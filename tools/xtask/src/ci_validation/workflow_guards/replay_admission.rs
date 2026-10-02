//! Required nightly admission, trajectory-reader setup and execution budgets.
//! This checks the declared workflow domain; it does not evaluate expressions.
use super::{Node, field};
use crate::command::DynResult;
use std::{collections::BTreeMap, path::Path};

const WORKFLOW: &str = "agentic-replay-nightly.yml";
const READER: &str = "Prepare pinned replay Python environment";
const REPOSITORY: &str = "github.repository=='Mesh-LLM/mesh-llm'";
const MAIN: &str = "github.ref=='refs/heads/main'";
const EVENTS: [&str; 2] = [
    "(github.event_name=='workflow_dispatch'||github.event_name=='schedule')",
    "(github.event_name=='schedule'||github.event_name=='workflow_dispatch')",
];

pub(super) fn check(root: &Path, workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let document = workflows.get(WORKFLOW).ok_or("missing replay workflow")?;
    let jobs = document.get("jobs").ok_or("missing replay jobs")?;
    let replay = jobs.get("replay").ok_or("missing replay job")?;
    admission(document, replay)?;
    let Some(Node::Seq(steps)) = replay.get("steps") else {
        return Err("replay requires a step sequence".into());
    };
    runner_order(replay, steps)?;
    reader_setup(root, replay, steps)?;
    budgets(jobs, replay, steps)
}

fn compact_condition(value: &str) -> DynResult<String> {
    let mut quoted = false;
    let mut compact = String::new();
    for character in value.chars() {
        if character == '\'' {
            quoted = !quoted;
        }
        if quoted || !character.is_whitespace() {
            compact.push(character);
        }
    }
    if quoted {
        return Err("unclosed replay admission literal".into());
    }
    Ok(compact)
}

fn admission(document: &Node, replay: &Node) -> DynResult<()> {
    let compact = compact_condition(field(replay, "if").unwrap_or(""))?;
    let condition = compact
        .strip_prefix("${{")
        .and_then(|value| value.strip_suffix("}}"))
        .unwrap_or(&compact);
    // This closed policy has exactly three conjuncts. Extra disjunctions,
    // disabled guards and arbitrary expression substitutions are not admitted.
    let admitted = EVENTS.iter().any(|event| {
        [
            [REPOSITORY, MAIN, *event],
            [MAIN, REPOSITORY, *event],
            [*event, REPOSITORY, MAIN],
            [*event, MAIN, REPOSITORY],
            [REPOSITORY, *event, MAIN],
            [MAIN, *event, REPOSITORY],
        ]
        .iter()
        .any(|terms| terms.join("&&") == condition)
    });
    if !admitted {
        return Err(
            "replay must require canonical repository, main and manual/scheduled event".into(),
        );
    }
    let triggers = document.get("on").ok_or("missing replay triggers")?;
    if triggers.get("workflow_dispatch").is_none()
        || triggers
            .entries()
            .iter()
            .any(|(key, _)| !matches!(key.as_str(), "workflow_dispatch" | "schedule"))
    {
        return Err(
            "replay triggers must remain manual with only optional reviewed schedule".into(),
        );
    }
    if document
        .get("concurrency")
        .and_then(|node| field(node, "cancel-in-progress"))
        != Some("false")
    {
        return Err("replay cannot cancel a previous full-session run".into());
    }
    Ok(())
}

fn named<'a>(steps: &'a [Node], name: &str) -> DynResult<(usize, &'a Node)> {
    let matches: Vec<_> = steps
        .iter()
        .enumerate()
        .filter(|(_, step)| field(step, "name") == Some(name))
        .collect();
    match matches.as_slice() {
        [matched] => Ok(*matched),
        _ => Err(format!("replay requires exactly one {name} step").into()),
    }
}

fn runner_order(replay: &Node, steps: &[Node]) -> DynResult<()> {
    let (position, guard) = named(steps, "Verify pinned replay runner")?;
    if position != 0
        || field(guard, "shell") != Some("bash")
        || field(guard, "run").is_none_or(|run| run.trim().is_empty())
        || guard.get("if").is_some()
        || guard.get("continue-on-error").is_some()
    {
        return Err("unconditional Bash runner guard must execute first".into());
    }
    let checkout = steps.get(1).ok_or("missing guarded replay checkout")?;
    if !field(checkout, "uses").is_some_and(|value| value.starts_with("actions/checkout@"))
        || checkout.get("with").and_then(|node| field(node, "ref")) != Some("${{ github.sha }}")
        || checkout
            .get("with")
            .and_then(|node| field(node, "persist-credentials"))
            != Some("false")
    {
        return Err("runner guard must precede immutable credential-free checkout".into());
    }
    let env = replay.get("env").ok_or("missing replay environment")?;
    if field(env, "EXPECTED_REPLAY_RUNNER_NAME") != Some("micstudio") {
        return Err("replay requires its pinned runner identity".into());
    }
    Ok(())
}

fn reader_setup(root: &Path, replay: &Node, steps: &[Node]) -> DynResult<()> {
    let (prepare_index, prepare) = named(steps, READER)?;
    let input_index = steps
        .iter()
        .position(|step| field(step, "id") == Some("inputs"))
        .ok_or("missing replay input admission")?;
    if prepare_index >= input_index
        || prepare.get("if").is_some()
        || prepare.get("continue-on-error").is_some()
    {
        return Err("locked trajectory reader must prepare before pinned inputs".into());
    }
    if !root.join("ci/agentic-replay-nightly/uv.lock").is_file() {
        return Err("trajectory reader requires its actual dependency lock".into());
    }
    let run = field(prepare, "run").ok_or("missing reader setup command")?;
    let lines: Vec<_> = run
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let direct_sync = lines
        .iter()
        .position(|line| *line == "uv sync --locked --project ci/agentic-replay-nightly")
        .ok_or("reader requires direct locked component-local sync")?;
    if lines.first() != Some(&"set -euo pipefail")
        || direct_sync != 1
        || lines.len() != 6
        || lines[2] != "REPLAY_PYTHON_BIN=\"$GITHUB_WORKSPACE/ci/agentic-replay-nightly/.venv/bin\""
        || lines[3] != "echo \"REPLAY_PYTHON=$REPLAY_PYTHON_BIN/python3\" >> \"$GITHUB_ENV\""
        || !lines[4].starts_with("\"$REPLAY_PYTHON_BIN/python3\" -c 'import duckdb;")
        || !lines[4].ends_with('\'')
        || lines[5] != "echo \"$REPLAY_PYTHON_BIN\" >> \"$GITHUB_PATH\""
    {
        return Err("reader setup must sync, export its own interpreter, import DuckDB and export PATH directly".into());
    }
    for node in [replay, prepare] {
        if node
            .get("env")
            .and_then(|env| env.get("HF_TOKEN"))
            .is_some()
        {
            return Err("trajectory reader preparation must not receive publisher token".into());
        }
    }
    Ok(())
}

fn minutes(node: &Node) -> DynResult<u64> {
    let text = field(node, "timeout-minutes").ok_or("missing replay timeout budget")?;
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("replay timeout budget must be a positive integer".into());
    }
    let value: u64 = text.parse()?;
    if value == 0 {
        return Err("replay timeout budget must be positive".into());
    }
    Ok(value)
}

fn budgets(jobs: &Node, replay: &Node, steps: &[Node]) -> DynResult<()> {
    for (_, job) in jobs.entries() {
        if let Some(Node::Seq(job_steps)) = job.get("steps") {
            for step in job_steps {
                if step.get("timeout-minutes").is_some() && minutes(step)? > 360 {
                    return Err("GitHub step timeout cannot exceed360 minutes".into());
                }
            }
        }
    }
    let mut execution = 0_u64;
    for id in ["model_0", "model_1", "model_2", "repair"] {
        let step = steps
            .iter()
            .find(|step| field(step, "id") == Some(id))
            .ok_or("missing budgeted replay or repair step")?;
        execution = execution
            .checked_add(minutes(step)?)
            .ok_or("replay budget overflow")?;
        let run = field(step, "run").ok_or("missing budgeted replay command")?;
        if !run.lines().any(|line| line.trim() == "ulimit -n 65536") {
            return Err("full-session replay and repair require descriptor headroom".into());
        }
    }
    if minutes(replay)? <= execution {
        return Err(
            "replay job must retain preflight and evidence headroom beyond all execution steps"
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "replay_admission_tests.rs"]
mod tests;
