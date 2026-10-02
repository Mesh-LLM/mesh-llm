//! Protected controller preparation before manual selected-source admission.
use super::{Node, field, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let workflow = workflows
        .get("llama-upstream-canary.yml")
        .ok_or("upstream canary missing")?;
    let job = h::job(workflow, "resolve")?;
    h::condition(
        job,
        "github.repository == 'Mesh-LLM/mesh-llm' && github.ref == 'refs/heads/main'",
    )?;
    let steps = h::steps(job)?;
    let checkout = h::checkout(steps, "${{ github.sha }}", None)?;
    let (bootstrap, prepare) = h::step(steps, "uses", "./.github/actions/prepare-automation")?;
    h::binding(h::member(prepare, "with")?, "runner-profile", "hosted-bare")?;
    let (resolve, step) = h::step(steps, "id", "resolve")?;
    if !(checkout < bootstrap && bootstrap < resolve) {
        return Err(
            "protected controller checkout/bootstrap must precede selected resolution".into(),
        );
    }
    let env = h::member(step, "env")?;
    h::binding(env, "MESH_REF", "${{ inputs.mesh_ref }}")?;
    h::binding(env, "UPSTREAM", "${{ inputs.upstream_sha }}")?;
    h::command(
        step,
        &["\"$MESH_LLM_AUTOMATION_BIN\"", "repository", "selected-ref"],
        &[
            ("--ref", "\"$MESH_REF\""),
            ("--event", "\"$GITHUB_EVENT_NAME\""),
            ("--upstream", "\"$UPSTREAM\""),
            (
                "--expected-origin",
                "\"$GITHUB_SERVER_URL/$GITHUB_REPOSITORY\"",
            ),
            ("--github-output", "\"$GITHUB_OUTPUT\""),
            ("--summary", "\"$GITHUB_STEP_SUMMARY\""),
        ],
    )?;
    selected_output_order(field(step, "run").ok_or("resolver run missing")?)
}
fn selected_output_order(run: &str) -> DynResult<()> {
    let run = run.replace("\\\n", " ");
    let (prefix, rest) = run
        .split_once("if [[ -n \"$MESH_REF\" ]]; then")
        .ok_or("explicit selected-ref branch missing")?;
    for key in ["mesh_source=", "certify=", "mode=", "upstream=", "changed="] {
        if prefix.contains(key) {
            return Err("selected output precedes source admission".into());
        }
    }
    let mut lines = Vec::new();
    let mut terminated = false;
    for line in rest.lines().map(str::trim) {
        if line == "fi" {
            terminated = true;
            break;
        }
        if !line.is_empty() && !line.starts_with('#') {
            lines.push(line);
        }
    }
    if !terminated {
        return Err("selected-ref branch terminator missing".into());
    }
    if lines.len() != 2
        || !lines[0].starts_with("\"$MESH_LLM_AUTOMATION_BIN\" repository selected-ref ")
        || lines[0].contains("||")
        || lines[0].contains(';')
        || lines[1] != "exit 0"
    {
        return Err(
            "selected branch must admit once then exit without pre-publishing or masking failure"
                .into(),
        );
    }
    Ok(())
}
#[cfg(test)]
#[path = "selected_ref_tests.rs"]
mod tests;
