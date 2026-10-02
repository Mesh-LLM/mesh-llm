//! Preflight admission and independent verification before canary publication.
use super::{Node, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let document = workflows
        .get("llama-upstream-canary.yml")
        .ok_or("missing upstream canary workflow")?;
    h::binding(h::member(document, "permissions")?, "contents", "read")?;
    h::binding(
        h::member(document, "concurrency")?,
        "cancel-in-progress",
        "false",
    )?;
    let schedule = h::member(h::member(document, "on")?, "schedule")?;
    if !matches!(schedule, Node::Seq(items) if items.iter().any(|item| super::field(item,"cron") == Some("47 3 * * *")))
    {
        return Err("upstream canary schedule must be retained".into());
    }
    let resolve = h::job(document, "resolve")?;
    h::condition(
        resolve,
        "github.repository == 'Mesh-LLM/mesh-llm' && github.ref == 'refs/heads/main'",
    )?;
    preflight(h::job(document, "preflight")?)?;
    candidate(h::job(document, "candidate")?)?;
    verification(h::job(document, "verification")?)?;
    result(h::job(document, "result")?)?;
    let publish = h::job(document, "publish-certified-canary")?;
    h::needs(publish, &["resolve", "result"])?;
    h::condition(publish, "${{ needs.result.outputs.publish == 'true' }}")?;
    super::canary_execution::publication(publish)?;
    let steps = h::steps(publish)?;
    h::checkout(steps, "${{ needs.resolve.outputs.source }}", None)?;
    let (_, writer) = h::step(steps, "id", "publish")?;
    let env = h::member(writer, "env")?;
    h::binding(
        env,
        "CANARY_CERTIFIED_SHA",
        "${{ needs.result.outputs.head }}",
    )?;
    h::binding(env, "CANARY_BRANCH", "${{ needs.result.outputs.branch }}")
}

fn result(job: &Node) -> DynResult<()> {
    h::needs(job, &["resolve", "preflight", "candidate", "verification"])?;
    h::condition(job, "${{ !cancelled() }}")?;
    let steps = h::steps(job)?;
    let (resolution, guard) = h::step(
        steps,
        "name",
        "Require successful protected source resolution before checkout",
    )?;
    h::binding(
        h::member(guard, "env")?,
        "RESOLVE_RESULT",
        "${{ needs.resolve.result }}",
    )?;
    h::command(
        guard,
        &[
            "if",
            "[",
            "\"$RESOLVE_RESULT\"",
            "!=",
            "success",
            "];",
            "then",
        ],
        &[],
    )?;
    h::command(guard, &["exit", "1"], &[])?;
    let controller = h::checkout(steps, "${{ needs.resolve.outputs.source }}", None)?;
    let (prepare, prepared) = h::step(steps, "uses", "./.github/actions/prepare-automation")?;
    h::binding(
        h::member(prepared, "with")?,
        "runner-profile",
        "hosted-bare",
    )?;
    let (decision, command) = h::step(steps, "id", "result")?;
    h::binding(
        h::member(command, "env")?,
        "NEEDS_JSON",
        "${{ toJSON(needs) }}",
    )?;
    h::command(
        command,
        &[
            "\"$MESH_LLM_AUTOMATION_BIN\"",
            "automation",
            "canary-receipts",
            "result",
        ],
        &[],
    )?;
    h::before(resolution, controller)?;
    h::before(controller, prepare)?;
    h::before(prepare, decision)
}

fn preflight(job: &Node) -> DynResult<()> {
    h::needs(job, &["resolve"])?;
    h::condition(job, "${{ needs.resolve.outputs.certify == 'true' }}")?;
    h::binding(
        h::member(job, "env")?,
        "CANARY_SOURCE_ROOT",
        "${{ github.workspace }}${{ needs.resolve.outputs.mesh_source != '' && '/canary-source' || '' }}",
    )?;
    let steps = h::steps(job)?;
    let controller = h::checkout(steps, "${{ needs.resolve.outputs.source }}", None)?;
    let selected = h::checkout(
        steps,
        "${{ needs.resolve.outputs.mesh_source }}",
        Some("canary-source"),
    )?;
    h::condition(&steps[selected], "needs.resolve.outputs.mesh_source != ''")?;
    let (prepare, _) = h::step(steps, "uses", "./.github/actions/prepare-automation")?;
    let (index, command) = h::step(
        steps,
        "name",
        "Verify immutable family plan and pinned cache",
    )?;
    h::before(controller, prepare)?;
    h::before(selected, index)?;
    h::before(prepare, index)?;
    let env = h::member(command, "env")?;
    h::binding(
        env,
        "CANARY_CONTROLLER_REVISION",
        "${{ needs.resolve.outputs.source }}",
    )?;
    h::binding(
        env,
        "CANARY_SELECTED_REVISION",
        "${{ needs.resolve.outputs.mesh_source || needs.resolve.outputs.source }}",
    )?;
    h::command(
        command,
        &["jq", "-n"],
        &[
            ("--arg", "controller_root"),
            ("controller_root", "\"$GITHUB_WORKSPACE\""),
            ("source_root", "\"$CANARY_SOURCE_ROOT\""),
            ("controller_revision", "\"$CANARY_CONTROLLER_REVISION\""),
            ("selected_revision", "\"$CANARY_SELECTED_REVISION\""),
            ("cache_root", "\"$HF_CACHE\""),
        ],
    )?;
    h::command(
        command,
        &[
            "\"$MESH_LLM_AUTOMATION_BIN\"",
            "automation",
            "canary-receipts",
            "preflight",
        ],
        &[("--input", "\"$input\"")],
    )
}
fn candidate(job: &Node) -> DynResult<()> {
    h::needs(job, &["resolve", "preflight"])?;
    h::condition(
        job,
        "${{ !cancelled() && needs.preflight.result == 'success' }}",
    )?;
    h::binding(
        job,
        "uses",
        "./.github/workflows/llama-canary-family-pass.yml",
    )?;
    let inputs = h::member(job, "with")?;
    for (key, value) in [
        ("source", "${{ needs.resolve.outputs.source }}"),
        ("mesh_source", "${{ needs.resolve.outputs.mesh_source }}"),
        ("upstream", "${{ needs.resolve.outputs.upstream }}"),
        ("pass_id", "repair-1"),
        ("mode", "${{ needs.resolve.outputs.mode }}"),
    ] {
        h::binding(inputs, key, value)?;
    }
    Ok(())
}
fn verification(job: &Node) -> DynResult<()> {
    h::needs(job, &["resolve", "candidate"])?;
    h::condition(
        job,
        "${{ !cancelled() && needs.resolve.outputs.changed == 'true' && needs.candidate.outputs.green == 'true' }}",
    )?;
    h::binding(
        job,
        "uses",
        "./.github/workflows/llama-canary-family-pass.yml",
    )?;
    let inputs = h::member(job, "with")?;
    for (key, value) in [
        ("source", "${{ needs.resolve.outputs.source }}"),
        ("upstream", "${{ needs.resolve.outputs.upstream }}"),
        ("pass_id", "verify-1"),
        ("mode", "verify-build"),
        ("previous_package", "${{ needs.candidate.outputs.package }}"),
        (
            "previous_identity",
            "${{ needs.candidate.outputs.identity }}",
        ),
        ("previous_head", "${{ needs.candidate.outputs.head }}"),
    ] {
        h::binding(inputs, key, value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn document() -> Node {
        super::super::workflow_yaml::parse(include_str!(
            "../../../../../.github/workflows/llama-upstream-canary.yml"
        ))
        .unwrap()
    }
    fn validate(node: Node) -> DynResult<()> {
        check(&BTreeMap::from([(
            "llama-upstream-canary.yml".into(),
            node,
        )]))
    }
    #[test]
    fn current_admission_graph_is_bound() {
        validate(document()).unwrap();
    }

    #[test]
    fn schedule_and_credential_free_controller_are_required() {
        let mut node = document();
        h::replace(
            &mut node,
            &["on", "schedule"],
            Node::Seq(vec![Node::Map(vec![(
                "cron".into(),
                Node::Scalar("0 0 * * *".into()),
            )])]),
        );
        assert!(validate(node).is_err());
        let mut node = document();
        let preflight = h::mutable(h::mutable(&mut node, "jobs"), "preflight");
        let Node::Seq(steps) = h::mutable(preflight, "steps") else {
            unreachable!()
        };
        h::replace(
            &mut steps[0],
            &["with", "persist-credentials"],
            Node::Scalar("true".into()),
        );
        assert!(validate(node).is_err());
    }
    #[test]
    fn omitted_preflight_gate_and_unverified_identity_are_rejected() {
        for (path, value) in [
            (
                vec!["jobs", "candidate", "needs"],
                Node::Scalar("resolve".into()),
            ),
            (vec!["jobs", "candidate", "if"], Node::Scalar("true".into())),
            (
                vec!["jobs", "verification", "with", "previous_identity"],
                Node::Scalar("${{ needs.candidate.outputs.head }}".into()),
            ),
            (
                vec!["jobs", "verification", "with", "mode"],
                Node::Scalar("repair-build".into()),
            ),
            (
                vec!["jobs", "publish-certified-canary", "if"],
                Node::Scalar("true".into()),
            ),
        ] {
            let mut node = document();
            h::replace(&mut node, &path, value);
            assert!(validate(node).is_err(), "{path:?}");
        }
    }
    #[test]
    fn result_cannot_substitute_input_or_use_only_a_comment() {
        for comment in [false, true] {
            let mut node = document();
            let Node::Seq(steps) =
                h::mutable(h::mutable(h::mutable(&mut node, "jobs"), "result"), "steps")
            else {
                unreachable!()
            };
            let result = steps
                .iter_mut()
                .find(|step| super::super::field(step, "id") == Some("result"))
                .unwrap();
            if comment {
                *h::mutable(result, "run") = Node::Scalar("# \"$MESH_LLM_AUTOMATION_BIN\" automation canary-receipts result\necho skipped".into());
            } else {
                *h::mutable(h::mutable(result, "env"), "NEEDS_JSON") = Node::Scalar("{}".into());
            }
            assert!(validate(node).is_err());
        }
    }
    #[test]
    fn preflight_cannot_use_selected_source_as_controller_or_precede_bootstrap() {
        for change in [0, 1] {
            let mut node = document();
            let Node::Map(jobs) = h::mutable(&mut node, "jobs") else {
                unreachable!()
            };
            let preflight = &mut jobs
                .iter_mut()
                .find(|(key, _)| key == "preflight")
                .unwrap()
                .1;
            let Node::Seq(steps) = h::mutable(preflight, "steps") else {
                unreachable!()
            };
            let prepare = steps
                .iter()
                .position(|step| {
                    super::super::field(step, "uses")
                        == Some("./.github/actions/prepare-automation")
                })
                .unwrap();
            let execute = steps
                .iter()
                .position(|step| {
                    super::super::field(step, "name")
                        == Some("Verify immutable family plan and pinned cache")
                })
                .unwrap();
            if change == 0 {
                h::replace(
                    &mut steps[execute],
                    &["env", "CANARY_CONTROLLER_REVISION"],
                    Node::Scalar("${{ needs.resolve.outputs.mesh_source }}".into()),
                );
            } else {
                steps.swap(prepare, execute);
            }
            assert!(validate(node).is_err());
        }
    }
}
