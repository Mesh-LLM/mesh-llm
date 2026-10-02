//! Controller versus selected-source authority and immutable worker handoffs.
use super::{Node, field, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let document = workflows
        .get("llama-canary-family-pass.yml")
        .ok_or("missing reusable canary workflow")?;
    h::binding(h::member(document, "permissions")?, "contents", "read")?;
    let env = h::member(document, "env")?;
    h::binding(env, "CANARY_CONTROLLER_SHA", "${{ inputs.source }}")?;
    h::binding(env, "CANARY_MESH_SOURCE", "${{ inputs.mesh_source }}")?;
    h::binding(
        env,
        "CANARY_SOURCE_ROOT",
        "${{ github.workspace }}${{ inputs.mesh_source != '' && '/canary-source' || '' }}",
    )?;
    let build = h::job(document, "build")?;
    h::condition(
        build,
        "github.repository == 'Mesh-LLM/mesh-llm' && github.ref == 'refs/heads/main'",
    )?;
    build_handoffs(build)?;
    worker(h::job(document, "family")?)?;
    let aggregate = h::job(document, "aggregate")?;
    super::canary_execution::aggregate(aggregate)?;
    h::needs(aggregate, &["build", "family"])?;
    h::condition(
        aggregate,
        "${{ !cancelled() && needs.build.result == 'success' }}",
    )?;
    let (_, admission) = h::step(h::steps(aggregate)?, "id", "aggregate")?;
    let env = h::member(admission, "env")?;
    h::binding(env, "IDENTITY", "${{ needs.build.outputs.identity }}")?;
    h::binding(env, "FAMILY_RESULT", "${{ needs.family.result }}")?;
    h::command(
        admission,
        &["test", "\"$FAMILY_RESULT\""],
        &[("=", "success")],
    )
}

fn build_handoffs(build: &Node) -> DynResult<()> {
    let steps = h::steps(build)?;
    let controller = h::checkout(steps, "${{ inputs.source }}", None)?;
    let selected = h::checkout(steps, "${{ inputs.mesh_source }}", Some("canary-source"))?;
    h::condition(&steps[selected], "inputs.mesh_source != ''")?;
    let (prepare, _) = h::step(steps, "uses", "./.github/actions/prepare-automation")?;
    let (execute, command) = h::step(steps, "id", "build")?;
    h::before(controller, prepare)?;
    h::before(selected, execute)?;
    h::before(prepare, execute)?;
    let env = h::member(command, "env")?;
    for (key, value) in [
        ("CANARY_HARNESS_MODE", "${{ inputs.mode }}"),
        ("CANARY_PASS_ID", "${{ inputs.pass_id }}"),
        ("UPSTREAM_SHA_INPUT", "${{ inputs.upstream }}"),
        (
            "CANARY_PREVIOUS_IDENTITY",
            "${{ inputs.previous_identity }}",
        ),
        ("CANARY_CANDIDATE_SHA", "${{ inputs.previous_head }}"),
        (
            "CANARY_BUILD_SOURCE_REVISION",
            "${{ inputs.mesh_source || inputs.source }}",
        ),
        (
            "CANARY_AGENT_PROVIDER",
            "${{ vars.LLAMA_CANARY_GOOSE_PROVIDER || 'zai_coding_plan' }}",
        ),
        (
            "CANARY_AGENT_MODEL",
            "${{ vars.LLAMA_CANARY_GOOSE_MODEL || 'glm-5.3-flash' }}",
        ),
        (
            "CANARY_PREVIOUS_PACKAGE",
            "${{ inputs.previous_package != '' && format('{0}/canary-previous-{1}', runner.temp, inputs.pass_id) || '' }}",
        ),
    ] {
        h::binding(env, key, value)?;
    }
    h::command(
        command,
        &[
            "\"$MESH_LLM_AUTOMATION_BIN\"",
            "automation",
            "canary-receipts",
            "build",
        ],
        &[("--input", "\"$input\"")],
    )?;
    h::command(
        command,
        &["jq", "-n"],
        &[
            ("controller_root", "\"$GITHUB_WORKSPACE\""),
            ("source_root", "\"$CANARY_SOURCE_ROOT\""),
            ("controller_revision", "\"$CANARY_CONTROLLER_SHA\""),
            ("selected_revision", "\"$CANARY_BUILD_SOURCE_REVISION\""),
            ("previous_identity", "\"$CANARY_PREVIOUS_IDENTITY\""),
            ("previous_candidate", "\"$CANARY_CANDIDATE_SHA\""),
        ],
    )?;
    let outputs = h::member(build, "outputs")?;
    for (key, value) in [
        ("matrix", "${{ steps.build.outputs.matrix }}"),
        ("identity", "${{ steps.build.outputs.identity_sha256 }}"),
        ("head", "${{ steps.build.outputs.candidate }}"),
    ] {
        h::binding(outputs, key, value)?;
    }
    let package = "llama-family-input-${{ github.run_id }}-${{ github.run_attempt }}-${{ inputs.pass_id }}-${{ steps.build.outputs.identity_sha256 }}";
    h::binding(outputs, "package", package)?;
    let (upload, step) = h::step(steps, "id", "upload_package")?;
    h::before(execute, upload)?;
    h::binding(h::member(step, "with")?, "name", package)
}

fn worker(job: &Node) -> DynResult<()> {
    h::needs(job, &["build"])?;
    let strategy = h::member(job, "strategy")?;
    h::binding(
        strategy,
        "matrix",
        "${{ fromJSON(needs.build.outputs.matrix) }}",
    )?;
    h::binding(strategy, "fail-fast", "false")?;
    let env = h::member(job, "env")?;
    h::binding(env, "IDENTITY", "${{ needs.build.outputs.identity }}")?;
    h::binding(
        env,
        "CANARY_SOURCE_ROOT",
        "${{ github.workspace }}/canary-source",
    )?;
    let steps = h::steps(job)?;
    super::canary_execution::worker(steps)?;
    let controller = h::checkout(steps, "${{ inputs.source }}", None)?;
    let selected = h::checkout(
        steps,
        "${{ inputs.mesh_source || inputs.source }}",
        Some("canary-source"),
    )?;
    let (prepare, _) = h::step(steps, "uses", "./.github/actions/prepare-automation")?;
    let (restore, command) = h::step(
        steps,
        "name",
        "Verify immutable handoff and restore producer executables",
    )?;
    h::before(controller, prepare)?;
    h::before(selected, restore)?;
    h::before(prepare, restore)?;
    let download = steps
        .iter()
        .enumerate()
        .find(|(_, step)| {
            field(step, "uses").is_some_and(|value| value.starts_with("actions/download-artifact@"))
        })
        .ok_or("missing immutable canary download")?;
    h::binding(
        h::member(download.1, "with")?,
        "name",
        "${{ needs.build.outputs.package }}",
    )?;
    h::binding(h::member(download.1, "with")?, "path", "${{ env.PACKAGE }}")?;
    h::before(download.0, restore)?;
    h::command(
        command,
        &[
            "\"$MESH_LLM_AUTOMATION_BIN\"",
            "automation",
            "canary-receipts",
            "restore",
        ],
        &[("--input", "\"$input\"")],
    )?;
    h::command(
        command,
        &["jq", "-n"],
        &[
            ("root", "\"$CANARY_SOURCE_ROOT\""),
            ("package", "\"$PACKAGE\""),
            ("identity_sha256", "\"$IDENTITY\""),
            ("controller_revision", "\"$CANARY_CONTROLLER_SHA\""),
        ],
    )?;
    let (certify, battery) = h::step(steps, "id", "certify")?;
    h::before(restore, certify)?;
    h::binding(battery, "continue-on-error", "true")?;
    h::binding(battery, "timeout-minutes", "720")?;
    h::binding(
        h::member(battery, "env")?,
        "SHARD_INDEX",
        "${{ matrix.shard_index }}",
    )?;
    h::binding(
        h::member(battery, "env")?,
        "MEMORY_TIER",
        "${{ matrix.memory_tier }}",
    )?;
    let (_, admission) = h::step(steps, "name", "Require successful family certification")?;
    h::binding(
        h::member(admission, "env")?,
        "OUTCOME",
        "${{ steps.certify.outcome }}",
    )?;
    h::command(admission, &["test", "\"$OUTCOME\""], &[("=", "success")])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn document() -> Node {
        super::super::workflow_yaml::parse(include_str!(
            "../../../../../.github/workflows/llama-canary-family-pass.yml"
        ))
        .unwrap()
    }
    fn validate(node: Node) -> DynResult<()> {
        check(&BTreeMap::from([(
            "llama-canary-family-pass.yml".into(),
            node,
        )]))
    }
    #[test]
    fn current_controller_worker_handoffs_are_bound() {
        validate(document()).unwrap();
    }
    #[test]
    fn selected_source_cannot_supply_controller_or_replace_artifact_identity() {
        for (path, value) in [
            (
                vec!["env", "CANARY_CONTROLLER_SHA"],
                "${{ inputs.mesh_source }}",
            ),
            (
                vec!["jobs", "build", "outputs", "identity"],
                "${{ steps.build.outputs.candidate }}",
            ),
            (
                vec!["jobs", "family", "env", "IDENTITY"],
                "${{ needs.build.outputs.head }}",
            ),
            (
                vec!["jobs", "family", "strategy", "matrix"],
                "${{ inputs.matrix }}",
            ),
            (vec!["jobs", "build", "if"], "true"),
        ] {
            let mut node = document();
            h::replace(&mut node, &path, Node::Scalar(value.into()));
            assert!(validate(node).is_err(), "{path:?}");
        }
    }
    #[test]
    fn restore_requires_bootstrap_and_actual_command_not_comment() {
        for change in [0, 1] {
            let mut node = document();
            let family = h::mutable(h::mutable(&mut node, "jobs"), "family");
            let Node::Seq(steps) = h::mutable(family, "steps") else {
                unreachable!()
            };
            let prepare = steps
                .iter()
                .position(|step| {
                    field(step, "uses") == Some("./.github/actions/prepare-automation")
                })
                .unwrap();
            let restore = steps
                .iter()
                .position(|step| {
                    field(step, "name")
                        == Some("Verify immutable handoff and restore producer executables")
                })
                .unwrap();
            if change == 0 {
                steps.swap(prepare, restore);
            } else {
                h::replace(&mut steps[restore],&["run"],Node::Scalar("# \"$MESH_LLM_AUTOMATION_BIN\" automation canary-receipts restore --input \"$input\"\necho skipped".into()));
            }
            assert!(validate(node).is_err());
        }
    }

    #[test]
    fn provider_and_previous_identity_cannot_be_rebound() {
        for (key, value) in [
            ("CANARY_AGENT_PROVIDER", "unapproved"),
            ("CANARY_PREVIOUS_IDENTITY", "${{ inputs.previous_head }}"),
        ] {
            let mut node = document();
            let build = h::mutable(h::mutable(&mut node, "jobs"), "build");
            let Node::Seq(steps) = h::mutable(build, "steps") else {
                unreachable!()
            };
            let command = steps
                .iter_mut()
                .find(|step| field(step, "id") == Some("build"))
                .unwrap();
            h::replace(command, &["env", key], Node::Scalar(value.into()));
            assert!(validate(node).is_err());
        }
    }
}
