//! Main runtime producers save verified ABI trees; manual maintenance never builds them.
use super::{Node, cache_predicate, field, workflow_yaml};
use crate::command::DynResult;
use std::{fs, path::Path};

const WARMER: &str = ".github/workflows/windows-warm-caches.yml";
const RUNTIME: &str = ".github/workflows/ci-windows-runtime-slice.yml";
const SAVE: &str = "./.github/actions/save-and-verify-actions-cache";
const PREPARE: &str = "./.github/actions/prepare-native-runtime-input";
const CACHE_SAVE: &str = "actions/cache/save@caa296126883cff596d87d8935842f9db880ef25";

fn is_runtime_save(step: &Node) -> bool {
    field(step, "uses").is_some_and(|uses| uses.starts_with(CACHE_SAVE))
        && !field(step, "if").is_some_and(|condition| {
            cache_predicate::requires(condition, "inputs.original_event_name == 'pull_request'")
        })
}

pub(super) fn check(root: &Path) -> DynResult<()> {
    let warmer = workflow_yaml::parse(&fs::read_to_string(root.join(WARMER))?)?;
    maintenance(&warmer)?;
    let runtime = workflow_yaml::parse(&fs::read_to_string(root.join(RUNTIME))?)?;
    publication(&runtime)?;
    let release = workflow_yaml::parse(&fs::read_to_string(
        root.join(".github/workflows/release.yml"),
    )?)?;
    release_ownership(&release)
}

fn release_ownership(document: &Node) -> DynResult<()> {
    for (name, scope) in [
        (
            "build_native_runtime_windows_cpu",
            "github.ref != 'refs/heads/main'",
        ),
        (
            "build_native_runtime_windows_gpu",
            "github.ref != 'refs/heads/main' || matrix.backend != 'vulkan'",
        ),
    ] {
        let job = document
            .get("jobs")
            .and_then(|jobs| jobs.get(name))
            .ok_or("release ABI job missing")?;
        let save = steps(job)?
            .iter()
            .find(|step| field(step, "uses") == Some(SAVE))
            .ok_or("release ABI save missing")?;
        if !cache_predicate::requires(field(save, "if").unwrap_or(""), scope) {
            return Err(
                format!("release.yml/{name}: shared main keys belong to Main Windows").into(),
            );
        }
    }
    Ok(())
}

fn maintenance(document: &Node) -> DynResult<()> {
    let events = document
        .get("on")
        .ok_or("Windows maintenance events missing")?;
    if events.entries().len() != 1 || events.get("workflow_dispatch").is_none() {
        return Err(format!("{WARMER}: maintenance must remain manual-only").into());
    }
    for (_, job) in document
        .get("jobs")
        .ok_or("maintenance jobs missing")?
        .entries()
    {
        for step in steps(job)? {
            if field(step, "uses").is_some_and(|uses| {
                uses == SAVE
                    || uses == PREPARE
                    || uses.starts_with("actions/cache/save@")
                    || uses.starts_with("actions/cache@")
            }) {
                return Err(
                    format!("{WARMER}: maintenance must not build or publish ABI trees").into(),
                );
            }
        }
    }
    Ok(())
}

fn steps(job: &Node) -> DynResult<&[Node]> {
    match job.get("steps") {
        Some(Node::Seq(steps)) => Ok(steps),
        _ => Err("Windows ABI job steps missing".into()),
    }
}

fn publication(document: &Node) -> DynResult<()> {
    let job = document
        .get("jobs")
        .and_then(|jobs| jobs.get("windows_runtime"))
        .ok_or("Windows runtime producer missing")?;
    let steps = steps(job)?;
    let saves: Vec<_> = steps
        .iter()
        .enumerate()
        .filter(|(_, step)| field(step, "uses") == Some(SAVE) || is_runtime_save(step))
        .collect();
    let [(save_index, save)] = saves.as_slice() else {
        return Err(format!("{RUNTIME}: require one exact shared ABI publisher").into());
    };
    let prepare_index = steps
        .iter()
        .position(|step| field(step, "uses") == Some(PREPARE))
        .ok_or("Windows runtime preparation missing")?;
    if prepare_index >= *save_index
        || field(&steps[prepare_index], "if").is_some()
        || field(save, "continue-on-error").is_some()
        || field(&steps[prepare_index], "continue-on-error").is_some()
    {
        return Err(
            format!("{RUNTIME}: publication must follow successful package verification").into(),
        );
    }
    let condition = field(save, "if").unwrap_or("");
    if condition.contains("always(")
        || condition.contains("failure(")
        || condition.contains("cancelled(")
        || condition.contains("false")
    {
        return Err(format!(
            "{RUNTIME}: publication must retain implicit success and remain enabled"
        )
        .into());
    }
    for clause in [
        "github.repository == 'Mesh-LLM/mesh-llm'",
        "github.ref == 'refs/heads/main'",
        "github.event_name == 'push'",
        "inputs.original_event_name == 'push'",
        "needs.runner_policy.outputs.allow_native_github_cache == 'true'",
        "steps.llama_cache.outputs.cache-hit != 'true'",
    ] {
        if !cache_predicate::requires(condition, clause) {
            return Err(format!("{RUNTIME}: ABI publication requires {clause}").into());
        }
    }
    let inputs = save.get("with").ok_or("ABI publication inputs missing")?;
    if !is_runtime_save(save) {
        return Err(
            format!("{RUNTIME}: ABI publication must use the pinned exact cache save").into(),
        );
    }
    for (name, value) in [
        ("path", "${{ steps.llama_cache.outputs.cache-path }}"),
        ("key", "${{ steps.llama_cache.outputs.cache-primary-key }}"),
    ] {
        if field(inputs, name) != Some(value) {
            return Err(format!("{RUNTIME}: ABI publication must preserve exact {name}").into());
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "windows_abi_publication_tests.rs"]
mod tests;
