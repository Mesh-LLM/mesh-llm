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
    super::canary_sdk::check(document)?;
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
    classified_outputs(aggregate, "aggregate")?;
    classified_outputs(h::job(document, "reconcile")?, "reconcile")?;
    classified_recheck(document)
}

fn classified_outputs(job: &Node, id: &str) -> DynResult<()> {
    let outputs = h::member(job, "outputs")?;
    for key in [
        "green",
        "state",
        "repairable",
        "failure_class",
        "failure_stage",
    ] {
        h::binding(
            outputs,
            key,
            &format!("${{{{ steps.{id}.outputs.{key} }}}}"),
        )?;
    }
    if id == "aggregate" {
        for key in ["retry_matrix", "feedback_ready"] {
            h::binding(
                outputs,
                key,
                &format!("${{{{ steps.{id}.outputs.{key} }}}}"),
            )?;
        }
    }
    let (kind, name, path) = if id == "aggregate" {
        (
            "feedback",
            "Upload classified family failure evidence",
            "canary-family-feedback",
        )
    } else {
        (
            "reconciled-feedback",
            "Upload reconciled candidate repair evidence",
            "reconciled-family-feedback",
        )
    };
    let identity = format!(
        "llama-family-{kind}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}-${{{{ inputs.pass_id }}}}-${{{{ needs.build.outputs.identity }}}}"
    );
    h::binding(outputs, "feedback", &identity)?;
    let steps = h::steps(job)?;
    let (execute, _) = h::step(steps, "id", id)?;
    let (upload, step) = h::step(steps, "name", name)?;
    h::before(execute, upload)?;
    h::condition(
        step,
        &format!("${{{{ !cancelled() && steps.{id}.outputs.feedback_ready == 'true' }}}}"),
    )?;
    if !field(step, "uses").is_some_and(|value| value.starts_with("actions/upload-artifact@")) {
        return Err("classified feedback requires an artifact upload".into());
    }
    let inputs = h::member(step, "with")?;
    h::binding(inputs, "name", &identity)?;
    h::binding(inputs, "path", &format!("${{{{ runner.temp }}}}/{path}/"))?;
    h::binding(inputs, "if-no-files-found", "error")
}

fn classified_recheck(document: &Node) -> DynResult<()> {
    let retry = h::job(document, "retry_family")?;
    h::needs(retry, &["build", "aggregate"])?;
    h::condition(
        retry,
        "${{ !cancelled() && needs.aggregate.outputs.state == 'infrastructure_retryable' }}",
    )?;
    h::binding(
        h::member(retry, "strategy")?,
        "matrix",
        "${{ fromJSON(needs.aggregate.outputs.retry_matrix) }}",
    )?;
    h::binding(
        h::member(retry, "env")?,
        "IDENTITY",
        "${{ needs.build.outputs.identity }}",
    )?;
    super::canary_execution::retry_worker(retry)?;
    let reconcile = h::job(document, "reconcile")?;
    h::needs(reconcile, &["build", "aggregate", "retry_family"])?;
    h::condition(
        reconcile,
        "${{ !cancelled() && needs.aggregate.outputs.state == 'infrastructure_retryable' }}",
    )?;
    let steps = h::steps(reconcile)?;
    let (execute, command) = h::step(steps, "id", "reconcile")?;
    let checkout = h::checkout(steps, "${{ inputs.source }}", None)?;
    h::before(checkout, execute)?;
    super::canary_execution::preparation(steps, "${{ inputs.source }}", execute)?;
    super::canary_execution::classified_context(command)?;
    reconciliation_downloads(steps, execute)?;
    h::binding(
        h::member(command, "env")?,
        "IDENTITY",
        "${{ needs.build.outputs.identity }}",
    )?;
    h::binding(
        h::member(command, "env")?,
        "FAMILY_RESULT",
        "${{ needs.retry_family.result }}",
    )?;
    h::command(
        command,
        &[
            "\"$MESH_LLM_AUTOMATION_BIN\"",
            "automation",
            "canary-receipts",
            "reconcile",
        ],
        &[
            ("--package", "\"$RUNNER_TEMP/reconcile-input\""),
            ("--identity", "\"$IDENTITY\""),
            ("--previous-feedback", "\"$RUNNER_TEMP/previous-feedback\""),
            ("--evidence", "\"$RUNNER_TEMP/retry-evidence\""),
            ("--family-result", "\"$FAMILY_RESULT\""),
            (
                "--feedback-output",
                "\"$RUNNER_TEMP/reconciled-family-feedback\"",
            ),
        ],
    )
}

fn reconciliation_downloads(steps: &[Node], execute: usize) -> DynResult<()> {
    for (key, identity, path) in [
        (
            "name",
            "${{ needs.build.outputs.package }}",
            "${{ runner.temp }}/reconcile-input",
        ),
        (
            "name",
            "${{ needs.aggregate.outputs.feedback }}",
            "${{ runner.temp }}/previous-feedback",
        ),
        (
            "pattern",
            "llama-family-retry-${{ github.run_id }}-${{ needs.build.outputs.identity }}-*-${{ inputs.pass_id }}-*",
            "${{ runner.temp }}/retry-evidence",
        ),
    ] {
        super::canary_execution::download(steps, key, identity, path, execute)?;
    }
    Ok(())
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
        (
            "CANARY_PREVIOUS_FEEDBACK",
            "${{ inputs.previous_feedback != '' && format('{0}/canary-feedback-{1}', runner.temp, inputs.pass_id) || '' }}",
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
            ("previous_feedback", "\"$CANARY_PREVIOUS_FEEDBACK\""),
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
    super::canary_execution::final_certification_gate(
        steps,
        "Require successful family certification",
    )
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
    #[test]
    fn retry_cannot_replace_feedback_custody() {
        for (path, value) in [
            (vec!["jobs", "retry_family", "if"], "true"),
            (
                vec!["jobs", "retry_family", "strategy", "matrix"],
                "${{ needs.build.outputs.matrix }}",
            ),
            (
                vec!["jobs", "retry_family", "env", "IDENTITY"],
                "${{ needs.build.outputs.head }}",
            ),
            (vec!["jobs", "reconcile", "if"], "true"),
        ] {
            let mut node = document();
            h::replace(&mut node, &path, Node::Scalar(value.into()));
            assert!(validate(node).is_err(), "{path:?}");
        }
        let mut node = document();
        let Node::Seq(steps) = h::mutable(
            h::mutable(h::mutable(&mut node, "jobs"), "reconcile"),
            "steps",
        ) else {
            unreachable!()
        };
        let execute = steps
            .iter_mut()
            .find(|step| field(step, "id") == Some("reconcile"))
            .unwrap();
        let source = field(execute, "run").unwrap().replace(
            "--previous-feedback \"$RUNNER_TEMP/previous-feedback\"",
            "--previous-feedback unbound",
        );
        h::replace(execute, &["run"], Node::Scalar(source));
        assert!(validate(node).is_err());
    }

    #[test]
    fn classified_transactions_require_native_context_and_feedback_destination() {
        for id in ["aggregate", "reconcile"] {
            for (before, after) in [
                ("--run-id \"$GITHUB_RUN_ID\"", "--run-id foreign"),
                ("--run-attempt \"$GITHUB_RUN_ATTEMPT\"", "--run-attempt 1"),
                (
                    "--controller-revision \"$CANARY_CONTROLLER_SHA\"",
                    "--controller-revision foreign",
                ),
                (
                    "--selected-source \"$CANARY_MESH_SOURCE\"",
                    "--selected-source foreign",
                ),
                (
                    "--family-result \"$FAMILY_RESULT\"",
                    "--family-result success",
                ),
                ("--feedback-output", "--feedback"),
                ("\"$MESH_LLM_AUTOMATION_BIN\" automation", "echo automation"),
            ] {
                let mut node = document();
                let job = h::mutable(h::mutable(&mut node, "jobs"), id);
                let Node::Seq(steps) = h::mutable(job, "steps") else {
                    unreachable!()
                };
                let command = steps
                    .iter_mut()
                    .find(|step| field(step, "id") == Some(id))
                    .unwrap();
                let run = field(command, "run").unwrap();
                assert!(run.contains(before), "missing mutation {id}: {before}");
                let run = run.replace(before, after);
                h::replace(command, &["run"], Node::Scalar(run));
                assert!(validate(node).is_err(), "{id}: {before}");
            }
        }
    }

    #[test]
    fn classified_transactions_cannot_skip_or_reorder_controller_preparation() {
        for id in ["aggregate", "reconcile"] {
            for mutation in 0..4 {
                let mut node = document();
                let job = h::mutable(h::mutable(&mut node, "jobs"), id);
                let Node::Seq(steps) = h::mutable(job, "steps") else {
                    unreachable!()
                };
                let prepare = steps
                    .iter()
                    .position(|step| {
                        field(step, "uses") == Some("./.github/actions/prepare-automation")
                    })
                    .unwrap();
                let execute = steps
                    .iter()
                    .position(|step| field(step, "id") == Some(id))
                    .unwrap();
                match mutation {
                    0 => {
                        steps.remove(prepare);
                    }
                    1 => steps.swap(prepare, execute),
                    2 => {
                        let Node::Map(entries) = &mut steps[prepare] else {
                            unreachable!()
                        };
                        entries.push(("if".into(), Node::Scalar("false".into())));
                    }
                    _ => h::replace(
                        &mut steps[prepare],
                        &["with", "runner-profile"],
                        Node::Scalar("untrusted".into()),
                    ),
                }
                assert!(validate(node).is_err(), "{id}: preparation {mutation}");
            }
        }
    }

    #[test]
    fn classified_outputs_and_feedback_uploads_cannot_forge_or_lose_readiness() {
        for id in ["aggregate", "reconcile"] {
            for key in [
                "green",
                "state",
                "repairable",
                "failure_class",
                "failure_stage",
                "feedback",
            ] {
                let mut node = document();
                h::replace(
                    &mut node,
                    &["jobs", id, "outputs", key],
                    Node::Scalar("true".into()),
                );
                assert!(validate(node).is_err(), "{id} output {key}");
            }
            for (path, value) in [
                (vec!["if"], "${{ success() }}"),
                (vec!["with", "name"], "foreign-feedback"),
                (vec!["with", "path"], "foreign-evidence/"),
                (vec!["with", "if-no-files-found"], "ignore"),
            ] {
                let mut node = document();
                let job = h::mutable(h::mutable(&mut node, "jobs"), id);
                let Node::Seq(steps) = h::mutable(job, "steps") else {
                    unreachable!()
                };
                let upload = steps
                    .iter_mut()
                    .find(|step| {
                        field(step, "name").is_some_and(|name| {
                            name.starts_with("Upload classified")
                                || name.starts_with("Upload reconciled")
                        })
                    })
                    .unwrap();
                h::replace(upload, &path, Node::Scalar(value.into()));
                assert!(validate(node).is_err(), "{id} upload {path:?}");
            }
        }
    }
    #[test]
    fn family_final_gate_preserves_uploaded_evidence_and_actual_failure() {
        for mutation in 0..6 {
            let mut node = document();
            let Node::Seq(steps) =
                h::mutable(h::mutable(h::mutable(&mut node, "jobs"), "family"), "steps")
            else {
                unreachable!()
            };
            let gate = steps
                .iter()
                .position(|step| {
                    field(step, "name") == Some("Require successful family certification")
                })
                .unwrap();
            let upload = steps
                .iter()
                .position(|step| field(step, "id") == Some("upload_evidence"))
                .unwrap();
            match mutation {
                0 => steps.swap(gate, upload),
                1 => h::replace(&mut steps[gate], &["if"], Node::Scalar("false".into())),
                2 => {
                    let Node::Map(fields) = &mut steps[gate] else {
                        unreachable!()
                    };
                    fields.push(("continue-on-error".into(), Node::Scalar("true".into())));
                }
                3 => h::replace(
                    &mut steps[gate],
                    &["env", "OUTCOME"],
                    Node::Scalar("success".into()),
                ),
                4 => h::replace(
                    &mut steps[gate],
                    &["run"],
                    Node::Scalar("echo skipped".into()),
                ),
                _ => h::replace(
                    &mut steps[gate],
                    &["run"],
                    Node::Scalar("test \"$OUTCOME\" = success || true".into()),
                ),
            }
            assert!(validate(node).is_err(), "final gate mutation {mutation}");
        }
    }
}
