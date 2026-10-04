//! Label-admitted, credential-free PR diagnostics with protected runner policy.
use super::{Node, field, handoffs as h};
use crate::command::DynResult;
use std::collections::{BTreeMap, BTreeSet};

const LANE: &str = "Mesh-LLM/mesh-llm/.github/workflows/ci-pr-canary-lane.yml@main";
const ADMISSION: &str = "contains(github.event.pull_request.labels.*.name, 'ci:canary') && (github.event.action == 'opened' || github.event.action == 'synchronize' || github.event.action == 'reopened' || github.event.action == 'ready_for_review' || (github.event.action == 'labeled' && github.event.label.name == 'ci:canary'))";
const GROUP: &str = "pr-ci-canary-${{ github.event.pull_request.number }}-${{ ((github.event.action == 'labeled' || github.event.action == 'unlabeled') && github.event.label.name != 'ci:canary') && format('unrelated-label-{0}', github.run_id) || 'active' }}";
const SLICES: &[(&str, &str)] = &[
    ("ui_artifact", "ci-ui-artifact-slice.yml"),
    ("hosts", "ci-linux-host-slice.yml"),
    ("native_runtime", "ci-linux-runtime-slice.yml"),
    ("product", "ci-linux-product-slice.yml"),
];
fn document<'a>(workflows: &'a BTreeMap<String, Node>, name: &str) -> DynResult<&'a Node> {
    workflows
        .get(name)
        .ok_or_else(|| format!("missing PR canary workflow {name}").into())
}
fn compact(value: &str) -> String {
    value.split_whitespace().collect()
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    entry(document(workflows, "pr_ci_canary.yml")?)?;
    let lane = document(workflows, "ci-pr-canary-lane.yml")?;
    lane_handoffs(lane)?;
    source_identity(lane)?;
    diagnostic_summary(lane)?;
    for (_, name) in SLICES {
        policy(document(workflows, name)?)?;
    }
    let mut visited = BTreeSet::new();
    closure(workflows, "pr_ci_canary.yml", &mut visited)?;
    let expected = ["pr_ci_canary.yml", "ci-pr-canary-lane.yml"]
        .into_iter()
        .chain(SLICES.iter().map(|(_, name)| *name))
        .collect::<BTreeSet<_>>();
    if visited != expected {
        return Err("PR canary must retain its six-workflow diagnostic closure".into());
    }
    Ok(())
}
fn entry(workflow: &Node) -> DynResult<()> {
    let events = h::member(workflow, "on")?;
    if events.entries().len() != 1 {
        return Err("PR canary requires pull_request-only admission".into());
    }
    let pr = h::member(events, "pull_request")?;
    let expected = [
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "labeled",
        "unlabeled",
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    if h::member(pr, "types")?
        .list()
        .into_iter()
        .collect::<BTreeSet<_>>()
        != expected
        || pr.get("paths").is_some()
        || pr.get("paths-ignore").is_some()
    {
        return Err("PR canary must observe label and source events without path filters".into());
    }
    let concurrency = h::member(workflow, "concurrency")?;
    h::binding(concurrency, "cancel-in-progress", "true")?;
    if compact(field(concurrency, "group").ok_or("missing PR canary concurrency identity")?)
        != compact(GROUP)
    {
        return Err("unrelated label events cannot cancel an active PR canary".into());
    }
    let canary = h::job(workflow, "canary")?;
    h::condition(canary, ADMISSION)?;
    h::condition(h::job(workflow, "inactive")?, &format!("!({ADMISSION})"))?;
    h::binding(canary, "uses", LANE)?;
    h::binding(h::member(canary, "with")?, "merge_sha", "${{ github.sha }}")
}
fn lane_handoffs(lane: &Node) -> DynResult<()> {
    h::member(h::member(lane, "on")?, "workflow_call")?;
    for (job_name, callee) in SLICES {
        let job = h::job(lane, job_name)?;
        h::binding(job, "uses", &format!("./.github/workflows/{callee}"))?;
        let inputs = h::member(job, "with")?;
        h::binding(inputs, "source_sha", "${{ inputs.merge_sha }}")?;
        h::binding(inputs, "force_hosted", "true")?;
        if inputs.get("policy_source_sha").is_some() {
            return Err("PR source cannot override protected runner-policy lineage".into());
        }
    }
    let jobs = h::member(lane, "jobs")?;
    for (_, callee) in SLICES {
        let expected = format!("./.github/workflows/{callee}");
        if jobs
            .entries()
            .iter()
            .filter(|(_, job)| field(job, "uses") == Some(&expected))
            .count()
            != 1
        {
            return Err("PR canary must call each existing slice exactly once".into());
        }
    }
    for (job, key, value) in [
        (
            "hosts",
            "hosts_matrix",
            "${{ needs.plan.outputs.host_matrix }}",
        ),
        (
            "native_runtime",
            "runtime_matrix",
            "${{ needs.plan.outputs.runtime_matrix }}",
        ),
        (
            "product",
            "runtime_matrix",
            "${{ needs.plan.outputs.product_matrix }}",
        ),
    ] {
        h::binding(h::member(h::job(lane, job)?, "with")?, key, value)?;
    }
    let plan = h::job(lane, "plan")?;
    let steps = h::steps(plan)?;
    h::step(steps, "id", "source_matrix")?;
    let outputs = h::member(plan, "outputs")?;
    for key in ["host_matrix", "runtime_matrix", "product_matrix"] {
        h::binding(
            outputs,
            key,
            &format!("${{{{ steps.source_matrix.outputs.{key} }}}}"),
        )?;
    }
    Ok(())
}
fn source_identity(lane: &Node) -> DynResult<()> {
    let steps = h::steps(h::job(lane, "plan")?)?;
    let checkout = h::checkout(steps, "${{ inputs.merge_sha }}", None)?;
    let (validate, step) = h::step(steps, "name", "Validate merge and head identities")?;
    let (changes, _) = h::step(steps, "id", "changes")?;
    h::before(checkout, validate)?;
    h::before(validate, changes)?;
    if step.get("if").is_some() || step.get("continue-on-error").is_some() {
        return Err("PR canary source identity must be checked unconditionally".into());
    }
    let env = h::member(step, "env")?;
    for (key, expected) in [
        ("BASE_SHA", "${{ inputs.base_sha }}"),
        ("HEAD_SHA", "${{ inputs.head_sha }}"),
        ("MERGE_SHA", "${{ inputs.merge_sha }}"),
        ("PR_NUMBER", "${{ inputs.pr_number }}"),
    ] {
        h::binding(env, key, expected)?;
    }
    h::command(
        step,
        &[
            "git",
            "fetch",
            "--no-tags",
            "origin",
            "\"$BASE_SHA\"",
            "\"$MERGE_SHA\"",
            "\"refs/pull/${PR_NUMBER}/head\"",
        ],
        &[],
    )?;
    h::command(
        step,
        &[
            "git",
            "merge-base",
            "--is-ancestor",
            "\"$HEAD_SHA\"",
            "\"$MERGE_SHA\"",
        ],
        &[],
    )?;
    let body = field(step, "run").ok_or("missing source identity command")?;
    for required in [
        "set -euo pipefail",
        "^[0-9a-f]{40}$",
        "^[1-9][0-9]*$",
        "git cat-file -e \"${!value}^{commit}\"",
        "parent_count < 3",
        "git diff --quiet \"$BASE_SHA\" \"$MERGE_SHA\" -- ci/ownership.yml ci/slices.yml",
    ] {
        if !body
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .any(|line| line.contains(required))
        {
            return Err(format!("PR canary source admission lost {required}").into());
        }
    }
    Ok(())
}
fn diagnostic_summary(lane: &Node) -> DynResult<()> {
    let summary = h::job(lane, "summary")?;
    h::binding(summary, "name", "Canary / CI")?;
    h::condition(summary, "${{ !cancelled() }}")?;
    let actual = h::member(summary, "needs")?
        .list()
        .into_iter()
        .collect::<BTreeSet<_>>();
    let expected = ["plan", "ui_artifact", "hosts", "native_runtime", "product"]
        .into_iter()
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err("PR canary summary must wait for its complete bounded graph".into());
    }
    let (_, step) = h::step(h::steps(summary)?, "name", "Validate canary graph")?;
    let env = h::member(step, "env")?;
    let body = field(step, "run").ok_or("missing canary summary")?;
    for (key, job) in [
        ("PLAN_RESULT", "plan"),
        ("UI_RESULT", "ui_artifact"),
        ("HOST_RESULT", "hosts"),
        ("RUNTIME_RESULT", "native_runtime"),
        ("PRODUCT_RESULT", "product"),
    ] {
        h::binding(env, key, &format!("${{{{ needs.{job}.result }}}}"))?;
        let assertion = format!("[[ \"${key}\" == success ]]");
        if !body.lines().any(|line| line.trim() == assertion) {
            return Err(format!("PR canary summary cannot skip {job} result").into());
        }
    }
    for required in [
        "set -euo pipefail",
        "native runtime-event gate",
        "Linux lane orchestration",
        "GITHUB_STEP_SUMMARY",
    ] {
        if !body.contains(required) {
            return Err(format!("PR canary summary lost {required}").into());
        }
    }
    Ok(())
}

fn policy(slice: &Node) -> DynResult<()> {
    h::member(
        h::member(h::member(slice, "on")?, "workflow_call")?,
        "inputs",
    )?
    .get("policy_source_sha")
    .ok_or("slice must declare protected policy source input")?;
    let mut found = false;
    for (_, job) in h::member(slice, "jobs")?.entries() {
        for step in job.get("steps").into_iter().flat_map(|steps| match steps {
            Node::Seq(items) => items.as_slice(),
            _ => &[],
        }) {
            if field(step, "uses").is_some_and(|value| value.starts_with("actions/checkout@"))
                && field(h::member(step, "with")?, "ref")
                    == Some(
                        "${{ inputs.policy_source_sha || github.event.repository.default_branch }}",
                    )
            {
                h::binding(h::member(step, "with")?, "persist-credentials", "false")?;
                found = true;
            }
        }
    }
    if !found {
        return Err("slice must resolve runner policy from protected default branch".into());
    }
    Ok(())
}
fn closure<'a>(
    workflows: &'a BTreeMap<String, Node>,
    name: &'a str,
    visited: &mut BTreeSet<&'a str>,
) -> DynResult<()> {
    if !visited.insert(name) {
        return Ok(());
    }
    let workflow = document(workflows, name)?;
    credential_free(workflow)?;
    for (_, job) in h::member(workflow, "jobs")?.entries() {
        if let Some(uses) = field(job, "uses") {
            let callee = if uses == LANE {
                "ci-pr-canary-lane.yml"
            } else {
                uses.strip_prefix("./.github/workflows/")
                    .ok_or("PR canary cannot add an external reusable workflow")?
            };
            closure(workflows, callee, visited)?;
        }
    }
    Ok(())
}
fn credential_free(node: &Node) -> DynResult<()> {
    match node {
        Node::Map(entries) => {
            for (key, value) in entries {
                if matches!(key.as_str(), "secrets" | "environment" | "id-token")
                    || (key == "use_depot" && value.text() == Some("true"))
                {
                    return Err("PR canary cannot acquire credentials or privileged runners".into());
                }
                if key == "permissions" {
                    read_permissions(value)?;
                }
                if key == "runs-on"
                    && value
                        .list()
                        .iter()
                        .any(|value| value.contains("self-hosted"))
                {
                    return Err("PR canary cannot run on self-hosted hardware".into());
                }
                credential_free(value)?;
            }
        }
        Node::Seq(items) => {
            for item in items {
                credential_free(item)?;
            }
        }
        Node::Scalar(_) => (),
    }
    Ok(())
}
fn read_permissions(node: &Node) -> DynResult<()> {
    match node {
        Node::Map(entries)
            if entries
                .iter()
                .all(|(_, value)| matches!(value.text(), Some("read" | "none"))) =>
        {
            Ok(())
        }
        Node::Scalar(value) if matches!(value.as_str(), "{}" | "read-all") => Ok(()),
        _ => Err("PR canary permissions must remain read-only".into()),
    }
}
#[cfg(test)]
#[path = "pr_canary_tests.rs"]
mod tests;
