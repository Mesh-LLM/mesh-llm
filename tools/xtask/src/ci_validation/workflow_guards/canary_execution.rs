//! Current Rust certification, receipt, aggregation and publication owners.
use super::{Node, field, handoffs as h};
use crate::command::DynResult;

fn input_owner(step: &Node, name: &str) -> DynResult<()> {
    h::command(
        step,
        &[
            "\"$MESH_LLM_AUTOMATION_BIN\"",
            "automation",
            "canary-receipts",
            name,
        ],
        &[("--input", "\"$input\"")],
    )
}
fn context(step: &Node) -> DynResult<()> {
    h::command(
        step,
        &["jq", "-n"],
        &[
            ("run_id", "\"$GITHUB_RUN_ID\""),
            ("run_attempt", "\"$GITHUB_RUN_ATTEMPT\""),
            ("controller_revision", "\"$CANARY_CONTROLLER_SHA\""),
            ("selected_source", "\"$CANARY_MESH_SOURCE\""),
        ],
    )
}

pub(super) fn worker(steps: &[Node]) -> DynResult<()> {
    let (certify, battery) = h::step(steps, "id", "certify")?;
    input_owner(battery, "certify")?;
    context(battery)?;
    h::command(
        battery,
        &["jq", "-n"],
        &[
            ("controller_root", "\"$GITHUB_WORKSPACE\""),
            ("root", "\"$CANARY_SOURCE_ROOT\""),
            ("package", "\"$PACKAGE\""),
            ("identity_sha256", "\"$IDENTITY\""),
            ("shard_index", "\"$SHARD_INDEX\""),
            ("memory_tier", "\"$MEMORY_TIER\""),
            (
                "evidence",
                "\"$FAMILY_BATTERY_ARTIFACT_ROOT/$FAMILY_BATTERY_RUN_ID\"",
            ),
        ],
    )?;
    let (receipt, command) = h::step(steps, "name", "Bind worker result to candidate and plan")?;
    h::before(certify, receipt)?;
    h::condition(command, "${{ !cancelled() }}")?;
    let env = h::member(command, "env")?;
    h::binding(env, "FAMILY", "${{ matrix.families }}")?;
    h::binding(env, "OUTCOME", "${{ steps.certify.outcome }}")?;
    input_owner(command, "receipt")?;
    context(command)?;
    h::command(
        command,
        &["jq", "-n"],
        &[
            ("package", "\"$PACKAGE\""),
            ("identity_sha256", "\"$IDENTITY\""),
            ("family", "\"$FAMILY\""),
            ("outcome", "\"$OUTCOME\""),
            ("runner", "\"$RUNNER_NAME\""),
            (
                "evidence",
                "\"$FAMILY_BATTERY_ARTIFACT_ROOT/$FAMILY_BATTERY_RUN_ID\"",
            ),
        ],
    )?;
    let (upload, _) = h::step(steps, "id", "upload_evidence")?;
    h::before(receipt, upload)
}

pub(super) fn aggregate(job: &Node) -> DynResult<()> {
    let steps = h::steps(job)?;
    let (execute, command) = h::step(steps, "id", "aggregate")?;
    preparation(steps, "${{ inputs.source }}", execute)?;
    h::command(
        command,
        &[
            "\"$MESH_LLM_AUTOMATION_BIN\"",
            "automation",
            "canary-receipts",
            "aggregate",
        ],
        &[
            ("--package", "\"$RUNNER_TEMP/aggregate-input\""),
            ("--identity", "\"$IDENTITY\""),
            ("--evidence", "\"$RUNNER_TEMP/aggregate-evidence\""),
            ("--run-id", "\"$GITHUB_RUN_ID\""),
            ("--run-attempt", "\"$GITHUB_RUN_ATTEMPT\""),
            ("--controller-revision", "\"$CANARY_CONTROLLER_SHA\""),
            ("--selected-source", "\"$CANARY_MESH_SOURCE\""),
        ],
    )?;
    download(
        steps,
        "name",
        "${{ needs.build.outputs.package }}",
        "${{ runner.temp }}/aggregate-input",
        execute,
    )?;
    download(
        steps,
        "pattern",
        "llama-family-${{ github.run_id }}-${{ needs.build.outputs.identity }}-*-${{ inputs.pass_id }}-*",
        "${{ runner.temp }}/aggregate-evidence",
        execute,
    )
}

pub(super) fn publication(job: &Node) -> DynResult<()> {
    let steps = h::steps(job)?;
    let (verify, command) = h::step(
        steps,
        "name",
        "Verify final handoff and write evidence summary",
    )?;
    preparation(steps, "${{ needs.resolve.outputs.source }}", verify)?;
    download(
        steps,
        "name",
        "${{ needs.result.outputs.package }}",
        "${{ runner.temp }}/certified-canary",
        verify,
    )?;
    let env = h::member(command, "env")?;
    for (key, value) in [
        ("IDENTITY", "${{ needs.result.outputs.identity }}"),
        (
            "CANARY_CONTROLLER_SHA",
            "${{ needs.resolve.outputs.source }}",
        ),
        (
            "CANARY_MESH_SOURCE",
            "${{ needs.resolve.outputs.mesh_source }}",
        ),
    ] {
        h::binding(env, key, value)?;
    }
    input_owner(command, "publication")?;
    context(command)?;
    h::command(
        command,
        &["jq", "-n"],
        &[
            ("package", "\"$RUNNER_TEMP/certified-canary\""),
            ("identity_sha256", "\"$IDENTITY\""),
            ("repository", "\"$GITHUB_REPOSITORY\""),
        ],
    )?;
    let (publish, _) = h::step(steps, "id", "publish")?;
    h::before(verify, publish)
}

fn preparation(steps: &[Node], revision: &str, execute: usize) -> DynResult<()> {
    let checkout = h::checkout(steps, revision, None)?;
    let (prepare, step) = h::step(steps, "uses", "./.github/actions/prepare-automation")?;
    h::binding(h::member(step, "with")?, "runner-profile", "hosted-bare")?;
    h::before(checkout, prepare)?;
    h::before(prepare, execute)?;
    for (index, _) in steps.iter().enumerate().filter(|(_, step)| {
        field(step, "uses").is_some_and(|value| value.starts_with("actions/download-artifact@"))
    }) {
        h::before(prepare, index)?;
    }
    Ok(())
}
fn download(steps: &[Node], key: &str, value: &str, path: &str, execute: usize) -> DynResult<()> {
    for (index, step) in steps.iter().enumerate() {
        if !field(step, "uses").is_some_and(|value| value.starts_with("actions/download-artifact@"))
        {
            continue;
        }
        let inputs = h::member(step, "with")?;
        if field(inputs, key) == Some(value) {
            h::binding(inputs, "path", path)?;
            return h::before(index, execute);
        }
    }
    Err(format!("missing immutable canary download {key}={value}").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn family() -> Node {
        super::super::workflow_yaml::parse(include_str!(
            "../../../../../.github/workflows/llama-canary-family-pass.yml"
        ))
        .unwrap()
    }
    fn upstream() -> Node {
        super::super::workflow_yaml::parse(include_str!(
            "../../../../../.github/workflows/llama-upstream-canary.yml"
        ))
        .unwrap()
    }
    #[test]
    fn current_rust_owners_bind_their_context_and_evidence() {
        worker(h::steps(h::job(&family(), "family").unwrap()).unwrap()).unwrap();
        aggregate(h::job(&family(), "aggregate").unwrap()).unwrap();
        publication(h::job(&upstream(), "publish-certified-canary").unwrap()).unwrap();
    }
    #[test]
    fn worker_outcome_swap_or_comment_only_certification_is_rejected() {
        for change in [0, 1] {
            let mut node = family();
            let job = h::mutable(h::mutable(&mut node, "jobs"), "family");
            let Node::Seq(steps) = h::mutable(job, "steps") else {
                unreachable!()
            };
            let step = if change == 0 {
                steps
                    .iter_mut()
                    .find(|step| {
                        field(step, "name") == Some("Bind worker result to candidate and plan")
                    })
                    .unwrap()
            } else {
                steps
                    .iter_mut()
                    .find(|step| field(step, "id") == Some("certify"))
                    .unwrap()
            };
            if change == 0 {
                h::replace(step, &["env", "OUTCOME"], Node::Scalar("success".into()));
            } else {
                h::replace(step,&["run"],Node::Scalar("# \"$MESH_LLM_AUTOMATION_BIN\" automation canary-receipts certify --input \"$input\"\necho skipped".into()));
            }
            assert!(worker(h::steps(h::job(&node, "family").unwrap()).unwrap()).is_err());
        }
    }
    #[test]
    fn publication_requires_prepared_controller_and_resolved_identity() {
        let mut node = upstream();
        let job = h::mutable(h::mutable(&mut node, "jobs"), "publish-certified-canary");
        let Node::Seq(steps) = h::mutable(job, "steps") else {
            unreachable!()
        };
        let prepare = steps
            .iter()
            .position(|step| field(step, "uses") == Some("./.github/actions/prepare-automation"))
            .unwrap();
        let verify = steps
            .iter()
            .position(|step| {
                field(step, "name") == Some("Verify final handoff and write evidence summary")
            })
            .unwrap();
        steps.swap(prepare, verify);
        assert!(publication(h::job(&node, "publish-certified-canary").unwrap()).is_err());
    }
    #[test]
    fn aggregate_cannot_substitute_source_for_package_identity() {
        let mut node = family();
        let job = h::mutable(h::mutable(&mut node, "jobs"), "aggregate");
        let Node::Seq(steps) = h::mutable(job, "steps") else {
            unreachable!()
        };
        let execute = steps
            .iter_mut()
            .find(|step| field(step, "id") == Some("aggregate"))
            .unwrap();
        let source = field(execute, "run").unwrap().replace(
            "--identity \"$IDENTITY\"",
            "--identity \"$CANARY_MESH_SOURCE\"",
        );
        h::replace(execute, &["run"], Node::Scalar(source));
        assert!(aggregate(h::job(&node, "aggregate").unwrap()).is_err());
    }
    #[test]
    fn family_memory_dispatch_preserves_selected_source_sdk_before_certification() {
        let document = family();
        let job = h::job(&document, "family").unwrap();
        let Node::Seq(runners) = h::member(job, "runs-on").unwrap() else {
            panic!("family runner list required")
        };
        assert!(runners.iter().any(
            |runner| matches!(runner, Node::Scalar(value) if value == "${{ matrix.memory_tier }}")
        ));
        let steps = h::steps(job).unwrap();
        let (sdk, _) = h::step(steps, "uses", "./.github/actions/setup-canary-python").unwrap();
        let (certify, command) = h::step(steps, "id", "certify").unwrap();
        h::before(sdk, certify).unwrap();
        h::binding(
            h::member(command, "env").unwrap(),
            "MEMORY_TIER",
            "${{ matrix.memory_tier }}",
        )
        .unwrap();
        worker(steps).unwrap();
    }
}
