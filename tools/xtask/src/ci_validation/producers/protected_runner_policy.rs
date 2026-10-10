//! Protected SDK producer authority is bound to active workflow nodes.
use super::super::lane_results::workflow_yaml::{self, Node};
use crate::command::DynResult;

#[derive(Clone, Copy)]
pub(super) enum Producer {
    NativeSdk,
    StaticAbi,
}

fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn require(node: &Node, key: &str, expected: &str) -> Result<(), String> {
    if text(node, key) != Some(expected) {
        return Err(format!("SDK runner policy {key} must bind {expected}"));
    }
    Ok(())
}
fn step<'a>(steps: &'a [Node], id: &str) -> Result<(usize, &'a Node), String> {
    let found = steps
        .iter()
        .enumerate()
        .filter(|(_, node)| text(node, "id") == Some(id))
        .collect::<Vec<_>>();
    let [(index, node)] = found.as_slice() else {
        return Err(format!("SDK runner policy requires exactly one {id} step"));
    };
    Ok((*index, *node))
}
fn selector(node: &Node) -> Result<(), String> {
    require(node, "uses", "./.github/actions/select-ci-runners")?;
    let inputs = node.get("with").ok_or("SDK selector inputs missing")?;
    for (key, value) in [
        ("event_name", "${{ github.event_name }}"),
        ("original_event_name", "${{ inputs.original_event_name }}"),
        ("repository", "${{ github.repository }}"),
        (
            "head_repository",
            "${{ github.event.pull_request.head.repo.full_name }}",
        ),
        (
            "head_sha",
            "${{ github.event.pull_request.head.sha || github.sha }}",
        ),
        ("ref", "${{ github.ref }}"),
        (
            "depot_main_enabled",
            "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
        ),
        (
            "depot_pr_enabled",
            "${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
        ),
        ("pr_canary_ref", "${{ vars.DEPOT_PR_CANARY_REF }}"),
        ("pr_approved_ref", "${{ vars.DEPOT_PR_APPROVED_REF }}"),
        ("pr_approved_sha", "${{ vars.DEPOT_PR_APPROVED_SHA }}"),
        ("force_hosted", "${{ inputs.force_hosted }}"),
        ("manual_use_depot", "${{ inputs.use_depot }}"),
    ] {
        require(inputs, key, value)?;
    }
    Ok(())
}
fn resolver(node: &Node, macos: bool) -> Result<(), String> {
    require(node, "shell", "bash")?;
    let env = node.get("env").ok_or("SDK resolver environment missing")?;
    require(env, "TARGET", "${{ inputs.target }}")?;
    require(env, "RUNNER_SIZE", "${{ inputs.runner_size }}")?;
    for (key, output) in [
        ("RUNNER_DEFAULT", "runner"),
        ("RUNNER_4", "runner_4"),
        ("RUNNER_8", "runner_8"),
        ("RUNNER_16", "runner_16"),
        ("RUNNER_ARM", "runner_arm"),
        ("RUNNER_ARM_4", "runner_arm_4"),
        ("RUNNER_ARM_8", "runner_arm_8"),
        ("RUNNER_ARM_16", "runner_arm_16"),
    ] {
        require(
            env,
            key,
            &format!("${{{{ steps.policy.outputs.{output} }}}}"),
        )?;
    }
    if macos {
        require(env, "POLICY_EVENT_NAME", "${{ github.event_name }}")?;
        require(
            env,
            "RUNNER_MACOS",
            "${{ steps.policy.outputs.runner_macos }}",
        )?;
    }
    if text(node, "run").is_none_or(str::is_empty) {
        return Err("SDK runner resolver must execute its bounded target/size adapter".into());
    }
    Ok(())
}
pub(super) fn check(source: &str, context: &str, producer: Producer) -> DynResult<()> {
    let document = workflow_yaml::parse(source).map_err(|error| format!("{context}: {error}"))?;
    verify(&document, producer).map_err(|error| format!("{context}: {error}").into())
}
fn verify(document: &Node, producer: Producer) -> Result<(), String> {
    let inputs = document
        .get("on")
        .and_then(|n| n.get("workflow_call"))
        .and_then(|n| n.get("inputs"))
        .ok_or("SDK workflow-call inputs missing")?;
    for forbidden in [
        "runs_on",
        "allow_depot_remote_cache",
        "allow_native_github_cache",
    ] {
        if inputs.get(forbidden).is_some() {
            return Err(format!("SDK caller cannot grant {forbidden}"));
        }
    }
    let size = inputs
        .get("runner_size")
        .ok_or("SDK runner_size input missing")?;
    require(size, "type", "string")?;
    require(size, "default", "8")?;
    let jobs = document.get("jobs").ok_or("SDK jobs missing")?;
    let macos = matches!(producer, Producer::NativeSdk);
    let policy = jobs
        .get("runner_policy")
        .ok_or("SDK protected runner policy missing")?;
    require(policy, "runs-on", "ubuntu-24.04")?;
    let Node::Seq(steps) = policy.get("steps").ok_or("SDK policy steps missing")? else {
        return Err("SDK policy steps must be a sequence".into());
    };
    let (select_index, select) = step(steps, "policy")?;
    let (resolve_index, resolve) = step(steps, "resolve")?;
    let checkouts = steps
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            text(node, "uses").is_some_and(|uses| uses.starts_with("actions/checkout@"))
        })
        .collect::<Vec<_>>();
    let [(checkout_index, checkout)] = checkouts.as_slice() else {
        return Err("SDK policy must have exactly one protected checkout".into());
    };
    if !(*checkout_index < select_index && select_index < resolve_index) {
        return Err("SDK protected checkout, selector and resolver order is invalid".into());
    }
    let checkout_inputs = checkout
        .get("with")
        .ok_or("SDK protected checkout inputs missing")?;
    require(
        checkout_inputs,
        "ref",
        "${{ github.event.repository.default_branch }}",
    )?;
    require(checkout_inputs, "persist-credentials", "false")?;
    selector(select)?;
    resolver(resolve, macos)?;
    require(
        policy.get("outputs").ok_or("SDK policy outputs missing")?,
        "runner",
        "${{ steps.resolve.outputs.runner }}",
    )?;
    let names: &[&str] = if macos {
        &["linux_native_sdk_artifact", "macos_native_sdk_artifact"]
    } else {
        &["static_abi_artifact"]
    };
    for name in names {
        let producer = jobs
            .get(name)
            .ok_or_else(|| format!("SDK producer {name} missing"))?;
        if !producer
            .get("needs")
            .is_some_and(|needs| needs.list().contains(&"runner_policy"))
        {
            return Err(format!("SDK producer {name} must need runner policy"));
        }
        require(
            producer,
            "runs-on",
            "${{ needs.runner_policy.outputs.runner }}",
        )?;
    }
    Ok(())
}
#[cfg(test)]
#[path = "protected_runner_policy_tests.rs"]
mod tests;
