//! Finalization of persistent-runner jobs after uploads and explicit cache saves.
use super::{Node, field};
use crate::command::DynResult;
use std::collections::BTreeMap;

#[path = "prepared_cleanup.rs"]
mod prepared_cleanup;

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let mut found = 0;
    for (workflow, document) in workflows {
        for (job_name, job) in document.get("jobs").into_iter().flat_map(Node::entries) {
            if !persistent_runner(job) {
                continue;
            }
            found += 1;
            final_step(job).map_err(|error| format!("{workflow}/{job_name}: {error}"))?;
            if (workflow == "ci-runner-contract-slice.yml" && job_name == "trusted_runner_image")
                || (workflow == "ci-linux-product-smoke-slice.yml"
                    && matches!(job_name.as_str(), "laya_cuda" | "laya_vulkan" | "laya_rocm"))
            {
                prepared_cleanup::check(job)
                    .map_err(|error| format!("{workflow}/{job_name}: {error}"))?;
            }
        }
    }
    if found == 0 {
        return Err("persistent-runner finalization census is empty".into());
    }
    let release = workflows
        .get("release.yml")
        .ok_or("release workflow missing")?;
    let cuda = release
        .get("jobs")
        .and_then(|jobs| jobs.get("build_native_runtime_linux_x86_64_cuda"))
        .ok_or("CUDA release producer missing")?;
    release_cache_order(cuda)
}

fn persistent_runner(job: &Node) -> bool {
    let literal = job.get("runs-on").is_some_and(|runner| {
        runner
            .list()
            .iter()
            .any(|value| value.contains("self-hosted"))
    });
    let matrix = job
        .get("strategy")
        .and_then(|strategy| strategy.get("matrix"))
        .and_then(|matrix| matrix.get("include"));
    literal
        || matches!(matrix, Some(Node::Seq(rows)) if rows.iter().any(|row| {
            field(row, "runner").is_some_and(|runner| runner.starts_with("mesh-llm-"))
        }))
}

fn steps(job: &Node) -> DynResult<&[Node]> {
    match job.get("steps") {
        Some(Node::Seq(steps)) if !steps.is_empty() => Ok(steps),
        _ => Err("persistent job requires nonempty steps".into()),
    }
}

fn final_step(job: &Node) -> DynResult<()> {
    let cleanup = steps(job)?.last().ok_or("cleanup missing")?;
    if field(cleanup, "timeout-minutes") != Some("5") {
        return Err("final cleanup requires a five-minute budget".into());
    }
    if !all_outcomes(field(cleanup, "if").unwrap_or("")) {
        return Err("final cleanup must run after success, failure and cancellation".into());
    }
    if !cleanup_invocation(field(cleanup, "run").unwrap_or("")) {
        return Err("final step must execute the typed runner-cleanup owner".into());
    }
    if prepared_cleanup::invocation(field(cleanup, "run").unwrap_or("")) {
        prepared_cleanup::check(job)?;
    }
    Ok(())
}

fn all_outcomes(condition: &str) -> bool {
    let normalized = condition
        .trim()
        .trim_start_matches("${{")
        .trim_end_matches("}}")
        .replace("success()", "S")
        .replace("failure()", "F")
        .replace("cancelled()", "C");
    let compact: String = normalized.chars().filter(|c| !c.is_whitespace()).collect();
    let (outcomes, selector) = compact.split_once("&&").unwrap_or((&compact, ""));
    let outcomes = outcomes.trim_matches(['(', ')']);
    let mut terms: Vec<_> = outcomes.split("||").collect();
    terms.sort_unstable();
    terms == ["C", "F", "S"]
        && matches!(
            selector,
            "" | "inputs.runner=='gpu-nvidia'"
                | "vars.USE_SELF_HOSTED=='true'&&needs.metadata.outputs.force_hosted_runners!='true'"
        )
}

fn cleanup_invocation(run: &str) -> bool {
    if prepared_cleanup::invocation(run) {
        return true;
    }
    // Only the existing direct command and frozen-controller/facade choice are admitted.
    // This closed stanza contract does not evaluate arbitrary shell control flow.
    let joined = run.replace("\\\n", " ");
    let mut lines: Vec<_> = joined
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    if lines.first() == Some(&"set -euo pipefail") {
        lines.remove(0);
    }
    match lines.as_slice() {
        [command] => direct_cleanup(command),
        [selector, frozen, otherwise, facade, end]
            if *selector == "if [[ -n \"${MESH_LLM_AUTOMATION_BIN:-}\" ]]; then"
                && *otherwise == "else"
                && *end == "fi" =>
        {
            frozen.starts_with("\"$MESH_LLM_AUTOMATION_BIN\" ")
                && facade.starts_with("cargo xtool ")
                && direct_cleanup(frozen)
                && direct_cleanup(facade)
        }
        _ => false,
    }
}

fn direct_cleanup(line: &str) -> bool {
    let command = line
        .strip_prefix("cargo xtool ")
        .or_else(|| line.strip_prefix("\"$MESH_LLM_AUTOMATION_BIN\" "));
    command.is_some_and(|command| {
        command.starts_with("ci-ops runner-cleanup --job ")
            && !command.contains(['|', ';', '&', '\x60', '<', '>', '(', ')'])
    })
}

fn release_cache_order(job: &Node) -> DynResult<()> {
    final_step(job)?;
    let steps = steps(job)?;
    let save = steps
        .get(steps.len().saturating_sub(2))
        .and_then(|step| field(step, "uses"));
    if !save.is_some_and(|action| action.starts_with("actions/cache/save@")) {
        return Err("CUDA release cache save must immediately precede final cleanup".into());
    }
    if !steps.iter().any(|step| {
        field(step, "uses").is_some_and(|action| action.starts_with("actions/cache/restore@"))
    }) {
        return Err("CUDA release cache restore missing".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "runner_finalization_tests.rs"]
mod tests;
