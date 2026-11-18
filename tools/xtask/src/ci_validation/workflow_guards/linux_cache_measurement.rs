//! The manual Linux cache measurement is nonpublishing: it may only restore the
//! exact production keys it measures, on a manual dispatch, with a read-only
//! token. Every step it reaches, including through local actions, is checked.
use super::{Node, cache_predicate::requires, field, workflow_yaml};
use crate::ci_operations::cache_inputs::Inventory;
use crate::command::DynResult;
use std::path::Path;

pub(super) const WORKFLOW: &str = "linux-cache-measurement.yml";
const RESTORE: &str = "actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25";
const UPLOAD: &str = "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f";
const EVIDENCE: &str = "linux-cache-measurement-evidence";
const NOT_LINUX: &str = "runner.os != 'Linux'";
const ADMITTED: &str = "${{ !cancelled() && steps.admit.outcome == 'success' }}";
const COHORTS: [(&str, Inventory); 3] = [
    ("measure_cpu_runtime", Inventory::LinuxRuntime),
    ("measure_hosted_xtask", Inventory::HostedXtask),
    ("measure_hosted_controller", Inventory::HostedController),
];
const DENIED_ACTIONS: [&str; 6] = [
    "actions/cache@",
    "actions/cache/save@",
    "./.github/actions/save-and-verify-actions-cache",
    "./.github/actions/prepare-automation",
    "./.github/actions/prepare-ci-controller",
    "./.github/actions/upload-automation",
];
const DENIED_COMMANDS: [&str; 7] = [
    "gh cache",
    "gh api",
    "gh release",
    "windows-cache-maintenance",
    "docker push",
    "cargo publish",
    "ACTIONS_RUNTIME_TOKEN",
];

pub(super) fn check(root: &Path) -> DynResult<()> {
    let workflows = root.join(".github/workflows");
    let document = workflow_yaml::parse(&std::fs::read_to_string(workflows.join(WORKFLOW))?)?;
    check_document(&document, &|name| {
        let path = root.join(format!(".github/actions/{name}/action.yml"));
        Ok(workflow_yaml::parse(&std::fs::read_to_string(path)?)?)
    })?;
    for entry in std::fs::read_dir(&workflows)? {
        let path = entry?.path();
        if path.file_name().and_then(|name| name.to_str()) != Some(WORKFLOW)
            && std::fs::read_to_string(&path)?.contains(WORKFLOW)
        {
            return Err(format!("{} reaches {WORKFLOW}", path.display()).into());
        }
    }
    Ok(())
}

type Actions<'a> = dyn Fn(&str) -> DynResult<Node> + 'a;

pub(super) fn check_document(document: &Node, actions: &Actions) -> DynResult<()> {
    triggers(document)?;
    per_sample_concurrency(document)?;
    permissions(document.get("permissions"), WORKFLOW)?;
    let jobs = document.get("jobs").ok_or("measurement jobs missing")?;
    for (name, job) in jobs.entries() {
        let context = format!("{WORKFLOW}/{name}");
        if field(job, "uses").is_some() {
            return Err(format!("{context}: measurement must not call a reusable workflow").into());
        }
        if field(job, "runs-on") != Some("ubuntu-24.04") {
            return Err(format!("{context}: measurement requires hosted ubuntu-24.04").into());
        }
        permissions(job.get("permissions"), &context)?;
        let mut restores = Vec::new();
        reached(steps(job), &context, actions, &mut restores)?;
        exact_restores(name, &restores).map_err(|error| format!("{context}: {error}"))?;
        if COHORTS.iter().any(|(cohort, _)| cohort == name) {
            evidence_upload(steps(job)).map_err(|error| format!("{context}: {error}"))?;
            admitted_finalization(steps(job)).map_err(|error| format!("{context}: {error}"))?;
        }
    }
    for (name, _) in COHORTS {
        jobs.get(name)
            .ok_or_else(|| format!("{WORKFLOW}: cohort job {name} missing"))?;
    }
    Ok(())
}

fn triggers(document: &Node) -> Result<(), String> {
    let events: Vec<_> = document
        .get("on")
        .ok_or("measurement trigger missing")?
        .entries()
        .iter()
        .map(|(event, _)| event.as_str())
        .collect();
    if events != ["workflow_dispatch"] {
        return Err(format!(
            "{WORKFLOW}: only workflow_dispatch may run the measurement, found {events:?}"
        ));
    }
    Ok(())
}

/// GitHub keeps one pending run per group, so samples dispatched together need distinct groups.
fn per_sample_concurrency(document: &Node) -> Result<(), String> {
    let concurrency = document
        .get("concurrency")
        .ok_or("measurement concurrency missing")?;
    let group = field(concurrency, "group").unwrap_or("");
    for input in ["cohort", "condition", "sample"] {
        if !group.contains(&format!("${{{{ inputs.{input} }}}}")) {
            return Err(format!(
                "{WORKFLOW}: concurrency group must include inputs.{input}"
            ));
        }
    }
    if field(concurrency, "cancel-in-progress") != Some("false") {
        return Err(format!(
            "{WORKFLOW}: measurements must not cancel in progress"
        ));
    }
    Ok(())
}

fn permissions(grant: Option<&Node>, context: &str) -> Result<(), String> {
    let grant = grant.ok_or_else(|| format!("{context}: explicit permissions required"))?;
    let entries = grant.entries();
    let read_only = !entries.is_empty()
        && entries.iter().all(|(name, level)| {
            matches!(name.as_str(), "contents" | "packages") && level.text() == Some("read")
        });
    if !read_only || grant.get("contents").is_none() {
        return Err(format!(
            "{context}: permissions must be contents: read (and packages: read for the runner image)"
        ));
    }
    Ok(())
}

fn steps(node: &Node) -> &[Node] {
    match node.get("steps") {
        Some(Node::Seq(steps)) => steps,
        _ => &[],
    }
}

/// Walks every step, descending into local actions unless the edge can run
/// only off Linux; every measurement job runs on hosted Linux.
fn reached<'a>(
    list: &'a [Node],
    context: &str,
    actions: &Actions,
    restores: &mut Vec<&'a Node>,
) -> DynResult<()> {
    let nested = context.contains(" -> ");
    for step in list {
        let uses = field(step, "uses").unwrap_or("");
        let run = field(step, "run").unwrap_or("");
        let condition = field(step, "if").unwrap_or("");
        if let Some(denied) = DENIED_ACTIONS
            .iter()
            .find(|denied| uses.starts_with(**denied))
            && !requires(condition, NOT_LINUX)
        {
            return Err(
                format!("{context}: reaches denied cache or publication step {denied}").into(),
            );
        }
        if let Some(denied) = DENIED_COMMANDS.iter().find(|denied| run.contains(**denied)) {
            return Err(format!("{context}: runs denied command {denied}").into());
        }
        if uses.starts_with("actions/upload-artifact@") {
            let path = step.get("with").and_then(|with| field(with, "path"));
            if uses != UPLOAD || path != Some(EVIDENCE) {
                return Err(
                    format!("{context}: only the evidence artifact may be uploaded").into(),
                );
            }
        }
        if uses.starts_with("actions/cache/restore@") && !requires(condition, NOT_LINUX) {
            if nested {
                return Err(format!(
                    "{context}: nested cache restore is outside the measured cohort"
                )
                .into());
            }
            restores.push(step);
        }
        if uses == "./.github/actions/configure-sccache-gha" {
            for flag in ["allow_depot_remote_cache", "allow_native_github_cache"] {
                if step.get("with").and_then(|with| field(with, flag)) != Some("false") {
                    return Err(format!("{context}: sccache {flag} must be literal false").into());
                }
            }
        }
        if let Some(name) = uses.strip_prefix("./.github/actions/")
            && !requires(condition, NOT_LINUX)
        {
            let action = actions(name)?;
            let mut ignored = Vec::new();
            let inner = action.get("runs").map_or(&[][..], steps);
            reached(
                inner,
                &format!("{context} -> {name}"),
                actions,
                &mut ignored,
            )?;
        }
    }
    Ok(())
}

fn evidence_upload(list: &[Node]) -> Result<(), String> {
    let last = list.last().ok_or("measurement steps missing")?;
    if field(last, "uses") != Some(UPLOAD) || field(last, "if") != Some("${{ !cancelled() }}") {
        return Err("the job must end by uploading its evidence even after a refusal".into());
    }
    let retention = last
        .get("with")
        .and_then(|with| field(with, "retention-days"));
    if retention != Some("14") {
        return Err("measurement evidence retains for 14 days".into());
    }
    Ok(())
}

/// An admitted sample must reach its receipt even when preparation fails.
fn admitted_finalization(list: &[Node]) -> Result<(), String> {
    let named = |name: &str| list.iter().find(|step| field(step, "name") == Some(name));
    if named("Admit measurement sample").and_then(|step| field(step, "id")) != Some("admit") {
        return Err("the admission step must have id admit".into());
    }
    for name in ["Mark prepare end", "Write measurement receipt"] {
        let step = named(name).ok_or_else(|| format!("{name} step missing"))?;
        if field(step, "if") != Some(ADMITTED) {
            return Err(format!("{name} must run for every admitted sample"));
        }
    }
    let receipt = named("Write measurement receipt").ok_or("receipt step missing")?;
    let outcome = receipt
        .get("env")
        .and_then(|env| field(env, "PREPARE_OUTCOME"));
    if !outcome.is_some_and(|value| value.contains("steps.prepare.outcome")) {
        return Err("the receipt must receive the preparation outcome".into());
    }
    Ok(())
}

fn exact_restores(job: &str, restores: &[&Node]) -> Result<(), String> {
    let expected = COHORTS
        .iter()
        .find(|(name, _)| *name == job)
        .map(|(_, inventory)| inventory.expression());
    match (expected, restores) {
        (None, []) => Ok(()),
        (Some(key), [step]) => {
            let with = step.get("with").ok_or("restore inputs missing")?;
            if field(step, "uses") != Some(RESTORE) {
                return Err("restore must use the pinned actions/cache/restore".into());
            }
            if with.get("restore-keys").is_some() || with.get("lookup-only").is_some() {
                return Err("restore must require the exact identity".into());
            }
            if field(with, "key") != Some(key.as_str()) {
                return Err("restore key must equal the production inventory expression".into());
            }
            if field(step, "if") != Some("${{ inputs.condition != 'cold' }}") {
                return Err("only warm arms may restore".into());
            }
            Ok(())
        }
        _ => Err("each cohort job restores its production key exactly once".into()),
    }
}

#[cfg(test)]
#[path = "linux_cache_measurement_tests.rs"]
mod tests;
