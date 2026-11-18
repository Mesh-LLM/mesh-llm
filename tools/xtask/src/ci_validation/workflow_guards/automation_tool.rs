//! The protected authority producers compile only the narrow
//! `authority-automation` executable; every other producer keeps full xtask.
use super::{Node, field};
use crate::command::DynResult;
use std::path::Path;

const TOOL_BINDING: &str =
    "${{ inputs.source-profile == 'protected-clean' && 'authority-automation' || 'xtask' }}";
const CLOSED_TOOLS: &str = "case \"$AUTOMATION_TOOL\" in\n  xtask|authority-automation) ;;\n  *) echo \"unsupported automation tool\" >&2; exit 1 ;;\nesac\n";
const IMAGE_FALLBACK: &str = "  case \"$AUTOMATION_TOOL\" in\n    xtask) recipe=automation-bootstrap ;;\n    authority-automation) recipe=authority-automation-bootstrap ;;\n    *) exit 1 ;;\n  esac\n  report=\"$(just \"$recipe\")\"\n";

pub(super) fn check(root: &Path) -> DynResult<()> {
    let read = |name: &str| -> DynResult<Node> {
        let path = root.join(format!(".github/actions/{name}/action.yml"));
        Ok(super::workflow_yaml::parse(&std::fs::read_to_string(
            path,
        )?)?)
    };
    prepare(&read("prepare-automation")?)?;
    upload(&read("upload-automation")?)?;
    Ok(())
}

fn steps(action: &Node) -> Result<&[Node], String> {
    match action.get("runs").and_then(|runs| runs.get("steps")) {
        Some(Node::Seq(steps)) => Ok(steps),
        _ => Err("automation action steps absent".into()),
    }
}

fn step<'a>(action: &'a Node, key: &str, value: &str) -> Result<&'a Node, String> {
    let found: Vec<_> = steps(action)?
        .iter()
        .filter(|step| field(step, key) == Some(value))
        .collect();
    match found.as_slice() {
        [step] => Ok(step),
        _ => Err(format!("unique automation step required: {value}")),
    }
}

fn prepare(action: &Node) -> Result<(), String> {
    let input = action
        .get("inputs")
        .and_then(|inputs| inputs.get("tool"))
        .ok_or("prepare-automation tool input absent")?;
    if field(input, "default") != Some("xtask") {
        return Err("prepare-automation tool must default to full xtask".into());
    }
    let validate = step(action, "name", "Validate preparation profile")?;
    if !field(validate, "run").is_some_and(|run| run.contains(CLOSED_TOOLS)) {
        return Err(
            "prepare-automation tool must admit only xtask and authority-automation".into(),
        );
    }
    for id in ["bootstrap_unix", "bootstrap_windows"] {
        let build = step(action, "id", id)?;
        if build
            .get("env")
            .and_then(|env| field(env, "AUTOMATION_TOOL"))
            != Some("${{ inputs.tool }}")
        {
            return Err(format!("{id} must build the closed tool input"));
        }
    }
    let cache = field(step(action, "id", "hosted_binary")?, "if").unwrap_or_default();
    if !cache.contains("inputs.tool == 'xtask'") {
        return Err("only full xtask may reuse the hosted executable cache".into());
    }
    Ok(())
}

fn upload(action: &Node) -> Result<(), String> {
    let prepare = step(action, "id", "prepare")?;
    if prepare.get("with").and_then(|with| field(with, "tool")) != Some(TOOL_BINDING) {
        return Err("protected-clean producers must build authority-automation".into());
    }
    let stage = step(action, "id", "stage")?;
    if stage
        .get("env")
        .and_then(|env| field(env, "AUTOMATION_TOOL"))
        != Some(TOOL_BINDING)
    {
        return Err("image fallback must build the producer's mapped tool".into());
    }
    if !field(stage, "run").is_some_and(|run| run.contains(IMAGE_FALLBACK)) {
        return Err("image fallback must select the mapped bootstrap recipe".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "automation_tool_tests.rs"]
mod tests;
