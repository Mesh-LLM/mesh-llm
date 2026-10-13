//! Startup budget for the compute-changes composite action's run scalars.
//! Only its declared event and immutable SHA inputs are admitted substitutions.
use super::{Node, workflow_yaml};
use crate::command::DynResult;
use std::{fs, path::Path};

const ACTION: &str = ".github/actions/compute-changes/action.yml";
const DERIVE: &str = ".github/actions/compute-changes/derive-outputs.sh";
const RUN_LIMIT: usize = 21_000;
const LONGEST_EVENT: &str = "workflow_dispatch";
const SHA_LENGTH: usize = 40;

pub(super) fn check(root: &Path) -> DynResult<()> {
    let source = fs::read_to_string(root.join(ACTION))?;
    let action = workflow_yaml::parse(&source)?;
    check_document(&action, &root.join(DERIVE))
}

fn check_document(action: &Node, derive: &Path) -> DynResult<()> {
    let steps = match action.get("runs").and_then(|runs| runs.get("steps")) {
        Some(Node::Seq(steps)) => steps,
        _ => return Err("compute-changes requires a composite step sequence".into()),
    };
    if !steps
        .iter()
        .any(|step| step.get("id").and_then(Node::text) == Some("derive"))
    {
        return Err("compute-changes must retain its derive step".into());
    }
    if !derive.is_file() {
        return Err("compute-changes requires its sibling derive-outputs.sh file".into());
    }
    for step in steps {
        let Some(run) = step.get("run") else {
            continue;
        };
        let run = run.text().ok_or("compute-changes run must be a scalar")?;
        let label = step
            .get("name")
            .or_else(|| step.get("id"))
            .and_then(Node::text)
            .unwrap_or("unnamed step");
        let expanded =
            expanded_length(run).map_err(|error| format!("compute-changes {label}: {error}"))?;
        if expanded >= RUN_LIMIT {
            return Err(format!("compute-changes {label}: expanded run is {expanded} characters; it must stay below {RUN_LIMIT}").into());
        }
    }
    Ok(())
}

fn expanded_length(run: &str) -> Result<usize, String> {
    let mut remainder = run;
    let mut length = 0usize;
    while let Some((literal, expression)) = remainder.split_once("${{") {
        length = length
            .checked_add(literal.chars().count())
            .ok_or("run length overflow")?;
        let (input, rest) = expression
            .split_once("}}")
            .ok_or("unterminated run expression")?;
        let replacement = match input.trim() {
            "inputs.event_name" => LONGEST_EVENT.len(),
            "inputs.base_sha" | "inputs.head_sha" => SHA_LENGTH,
            _ => return Err(format!("unsupported run expression: {}", input.trim())),
        };
        length = length
            .checked_add(replacement)
            .ok_or("run length overflow")?;
        remainder = rest;
    }
    length
        .checked_add(remainder.chars().count())
        .ok_or_else(|| "run length overflow".into())
}

#[cfg(test)]
#[path = "compute_changes_budget_tests.rs"]
mod tests;
