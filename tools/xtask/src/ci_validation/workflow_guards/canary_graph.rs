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
    distributed_attempts(document)?;
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
    h::needs(
        job,
        &[
            "resolve",
            "preflight",
            "attempt_1",
            "attempt_2",
            "attempt_3",
        ],
    )?;
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
    let (decision, command) = h::step(steps, "id", "result")?;
    let env = h::member(command, "env")?;
    for (key, value) in [
        ("CERTIFY", "${{ needs.resolve.outputs.certify }}"),
        ("CHANGED", "${{ needs.resolve.outputs.changed }}"),
        ("MESH_SOURCE", "${{ needs.resolve.outputs.mesh_source }}"),
        ("PREFLIGHT", "${{ needs.preflight.result }}"),
        ("ATTEMPTS_JSON", "${{ toJSON(needs) }}"),
    ] {
        h::binding(env, key, value)?;
    }
    h::command(
        command,
        &[
            "\"$MESH_LLM_AUTOMATION_BIN\"",
            "automation",
            "canary-receipts",
            "select-final",
        ],
        &[
            ("--certify", "\"$CERTIFY\""),
            ("--changed", "\"$CHANGED\""),
            ("--mesh-source", "\"$MESH_SOURCE\""),
            ("--preflight", "\"$PREFLIGHT\""),
            ("--attempts-json", "\"$ATTEMPTS_JSON\""),
        ],
    )?;
    h::before(resolution, controller)?;
    selector_preparation(job, steps, controller, decision)
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
// Native selectors bind the finite distributed attempts to protected job outputs.
// The guard binds all three bounded attempts to producer identities and a fresh
// independent verification instead of treating a local repair result as green.
fn distributed_attempts(document: &Node) -> DynResult<()> {
    attempt_outputs(document)?;
    for number in 1..=3 {
        let repair_name = format!("repair_{number}");
        let verify_name = format!("verify_{number}");
        let attempt_name = format!("attempt_{number}");
        let repair = h::job(document, &repair_name)?;
        let previous = format!("attempt_{}", number - 1);
        let dependency = if number == 1 { "preflight" } else { &previous };
        h::needs(repair, &["resolve", dependency])?;
        let gate = if number == 1 {
            "${{ !cancelled() && needs.preflight.result == 'success' }}".to_owned()
        } else {
            format!(
                "${{{{ !cancelled() && needs.resolve.outputs.changed == 'true' && needs.{previous}.outputs.state == 'repairable' }}}}"
            )
        };
        h::condition(repair, &gate)?;
        pass(repair, &format!("repair-{number}"))?;
        let inputs = h::member(repair, "with")?;
        h::binding(
            inputs,
            "mode",
            if number == 1 {
                "${{ needs.resolve.outputs.mode }}"
            } else {
                "repair-build"
            },
        )?;
        if number == 1 {
            h::binding(
                inputs,
                "mesh_source",
                "${{ needs.resolve.outputs.mesh_source }}",
            )?;
        } else {
            for field in ["package", "identity", "head", "feedback"] {
                h::binding(
                    inputs,
                    &format!("previous_{field}"),
                    &format!("${{{{ needs.{previous}.outputs.resume_{field} }}}}"),
                )?;
            }
        }
        independent_pass(document, number, &repair_name, &verify_name)?;
        let attempt = h::job(document, &attempt_name)?;
        h::needs(attempt, &["resolve", &repair_name, &verify_name])?;
        if number > 1 {
            h::needs(attempt, &[&previous])?;
        }
        h::condition(
            attempt,
            if number == 1 {
                "${{ !cancelled() && needs.resolve.outputs.certify == 'true' }}"
            } else {
                &gate
            },
        )?;
        select_attempt(attempt, &repair_name, &verify_name)?;
    }
    Ok(())
}
fn attempt_outputs(document: &Node) -> DynResult<()> {
    for number in 1..=3 {
        let outputs = h::member(h::job(document, &format!("attempt_{number}"))?, "outputs")?;
        for field in [
            "state",
            "green",
            "repairable",
            "resume_package",
            "resume_identity",
            "resume_head",
            "resume_feedback",
            "failure_class",
            "failure_stage",
            "package",
            "identity",
            "head",
            "branch",
        ] {
            h::binding(
                outputs,
                field,
                &format!("${{{{ steps.select.outputs.{field} }}}}"),
            )?;
        }
    }
    Ok(())
}
fn independent_pass(
    document: &Node,
    number: usize,
    repair_name: &str,
    verify_name: &str,
) -> DynResult<()> {
    let verify = h::job(document, verify_name)?;
    h::needs(verify, &["resolve", repair_name])?;
    let changed = if number == 1 {
        "needs.resolve.outputs.changed == 'true' && "
    } else {
        ""
    };
    h::condition(
        verify,
        &format!("${{{{ !cancelled() && {changed}needs.{repair_name}.outputs.green == 'true' }}}}"),
    )?;
    pass(verify, &format!("verify-{number}"))?;
    let inputs = h::member(verify, "with")?;
    h::binding(inputs, "mode", "verify-build")?;
    for field in ["package", "identity", "head"] {
        h::binding(
            inputs,
            &format!("previous_{field}"),
            &format!("${{{{ needs.{repair_name}.outputs.{field} }}}}"),
        )?;
    }
    Ok(())
}
fn select_attempt(attempt: &Node, repair_name: &str, verify_name: &str) -> DynResult<()> {
    let steps = h::steps(attempt)?;
    let checkout = h::checkout(steps, "${{ needs.resolve.outputs.source }}", None)?;
    let (select, command) = h::step(steps, "id", "select")?;
    selector_preparation(attempt, steps, checkout, select)?;
    let env = h::member(command, "env")?;
    h::binding(env, "CHANGED", "${{ needs.resolve.outputs.changed }}")?;
    h::binding(
        env,
        "REPAIR_JSON",
        &format!("${{{{ toJSON(needs.{repair_name}) }}}}"),
    )?;
    h::binding(
        env,
        "VERIFY_JSON",
        &format!("${{{{ toJSON(needs.{verify_name}) }}}}"),
    )?;
    h::command(
        command,
        &[
            "\"$MESH_LLM_AUTOMATION_BIN\"",
            "automation",
            "canary-receipts",
            "select-attempt",
        ],
        &[
            ("--changed", "\"$CHANGED\""),
            ("--repair-json", "\"$REPAIR_JSON\""),
            ("--verification-json", "\"$VERIFY_JSON\""),
        ],
    )?;
    Ok(())
}
fn selector_preparation(
    job: &Node,
    steps: &[Node],
    checkout: usize,
    decision: usize,
) -> DynResult<()> {
    h::binding(job, "runs-on", "ubuntu-24.04")?;
    let preparations = steps
        .iter()
        .enumerate()
        .filter(|(_, step)| {
            super::field(step, "uses") == Some("./.github/actions/prepare-automation")
        })
        .collect::<Vec<_>>();
    let [(prepare, step)] = preparations.as_slice() else {
        return Err("selector requires exactly one protected automation preparation".into());
    };
    if step.get("if").is_some()
        || step.get("continue-on-error").is_some()
        || step.get("run").is_some()
    {
        return Err("selector preparation must run unconditionally and fail closed".into());
    }
    let inputs = h::member(step, "with")?;
    h::binding(inputs, "runner-profile", "hosted-bare")?;
    h::binding(inputs, "allow_depot_remote_cache", "false")?;
    h::binding(inputs, "allow_native_github_cache", "false")?;
    h::before(checkout, *prepare)?;
    h::before(*prepare, decision)
}

fn pass(job: &Node, pass_id: &str) -> DynResult<()> {
    h::binding(
        job,
        "uses",
        "./.github/workflows/llama-canary-family-pass.yml",
    )?;
    let inputs = h::member(job, "with")?;
    h::binding(inputs, "source", "${{ needs.resolve.outputs.source }}")?;
    h::binding(inputs, "upstream", "${{ needs.resolve.outputs.upstream }}")?;
    h::binding(inputs, "pass_id", pass_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn document() -> Node {
        super::super::workflow_yaml::parse_resolved_aliases(include_str!(
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
                vec!["jobs", "repair_1", "needs"],
                Node::Scalar("resolve".into()),
            ),
            (vec!["jobs", "repair_1", "if"], Node::Scalar("true".into())),
            (
                vec!["jobs", "verify_1", "with", "previous_identity"],
                Node::Scalar("${{ needs.repair_1.outputs.head }}".into()),
            ),
            (
                vec!["jobs", "verify_1", "with", "mode"],
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
                *h::mutable(result, "run") = Node::Scalar(
                    "# \"$MESH_LLM_AUTOMATION_BIN\" automation canary-receipts select-final\necho skipped".into(),
                );
            } else {
                *h::mutable(h::mutable(result, "env"), "ATTEMPTS_JSON") = Node::Scalar("{}".into());
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
    #[test]
    fn resumed_attempts_require_previous_identity_feedback_and_fresh_verification() {
        for number in 2..=3 {
            for field in ["package", "identity", "head", "feedback"] {
                let mut node = document();
                h::replace(
                    &mut node,
                    &[
                        "jobs",
                        &format!("repair_{number}"),
                        "with",
                        &format!("previous_{field}"),
                    ],
                    Node::Scalar("unbound".into()),
                );
                assert!(validate(node).is_err());
            }
            let mut node = document();
            h::replace(
                &mut node,
                &[
                    "jobs",
                    &format!("verify_{number}"),
                    "with",
                    "previous_identity",
                ],
                Node::Scalar("unbound".into()),
            );
            assert!(validate(node).is_err());
        }
        let mut node = document();
        let Node::Seq(steps) = h::mutable(
            h::mutable(h::mutable(&mut node, "jobs"), "attempt_2"),
            "steps",
        ) else {
            unreachable!()
        };
        let select = steps
            .iter_mut()
            .find(|step| super::super::field(step, "id") == Some("select"))
            .unwrap();
        h::replace(
            select,
            &["env", "VERIFY_JSON"],
            Node::Scalar("${{ toJSON(needs.repair_2) }}".into()),
        );
        assert!(validate(node).is_err());
    }
    #[test]
    fn selector_outputs_cannot_claim_green_or_forward_unverified_repair_bytes() {
        for field in ["state", "green", "package", "identity", "head", "branch"] {
            let mut node = document();
            h::replace(
                &mut node,
                &["jobs", "attempt_1", "outputs", field],
                Node::Scalar(if field == "state" {
                    "green".into()
                } else {
                    format!("${{{{ needs.repair_1.outputs.{field} }}}}")
                }),
            );
            assert!(validate(node).is_err(), "{field}");
        }
        let mut node = document();
        h::replace(
            &mut node,
            &["jobs", "attempt_2", "outputs"],
            Node::Scalar("*unchecked_outputs".into()),
        );
        assert!(validate(node).is_err());
    }
    #[test]
    fn yaml_anchor_substitution_cannot_redirect_validated_attempt_outputs() {
        let source = include_str!("../../../../../.github/workflows/llama-upstream-canary.yml");
        let mut replaced = source.replace("outputs: &attempt_outputs", "outputs: &safe_outputs");
        // Earlier unrelated mapping now owns the alias consumed by attempts 2/3.
        replaced = replaced.replacen("jobs:\n", "env: &attempt_outputs\n  state: green\n  green: 'true'\n  package: forged\n  identity: forged\n  head: forged\n  branch: forged\njobs:\n", 1);
        let node = super::super::workflow_yaml::parse_resolved_aliases(&replaced).unwrap();
        assert!(validate(node).is_err());
        assert!(
            super::super::workflow_yaml::parse_resolved_aliases(
                &source.replace("*attempt_outputs", "*unknown_outputs")
            )
            .is_err()
        );
        assert!(
            super::super::workflow_yaml::parse_resolved_aliases(&source.replace(
                "  attempt_1:\n",
                "  duplicate: &attempt_outputs {}\n  attempt_1:\n"
            ))
            .is_err()
        );
    }
    fn selector_steps<'a>(node: &'a mut Node, job: &str) -> &'a mut Vec<Node> {
        let Node::Seq(steps) = h::mutable(h::mutable(h::mutable(node, "jobs"), job), "steps")
        else {
            unreachable!()
        };
        steps
    }

    #[test]
    fn native_selector_preparation_is_exact_ordered_unconditional_and_cache_denied() {
        for job in ["attempt_1", "attempt_2", "attempt_3", "result"] {
            for mutation in [
                "missing",
                "duplicate",
                "before-checkout",
                "after-decision",
                "profile",
                "depot",
                "github",
                "conditional",
                "optional",
            ] {
                let mut node = document();
                let steps = selector_steps(&mut node, job);
                let prepare = steps
                    .iter()
                    .position(|step| {
                        super::super::field(step, "uses")
                            == Some("./.github/actions/prepare-automation")
                    })
                    .unwrap();
                let checkout = steps
                    .iter()
                    .position(|step| {
                        super::super::field(step, "uses")
                            .is_some_and(|value| value.starts_with("actions/checkout@"))
                    })
                    .unwrap();
                let decision = steps
                    .iter()
                    .position(|step| {
                        super::super::field(step, "id")
                            == Some(if job == "result" { "result" } else { "select" })
                    })
                    .unwrap();
                match mutation {
                    "missing" => {
                        steps.remove(prepare);
                    }
                    "duplicate" => {
                        steps.push(steps[prepare].clone());
                    }
                    "before-checkout" => steps.swap(prepare, checkout),
                    "after-decision" => steps.swap(prepare, decision),
                    "profile" => h::replace(
                        &mut steps[prepare],
                        &["with", "runner-profile"],
                        Node::Scalar("image".into()),
                    ),
                    "depot" => h::replace(
                        &mut steps[prepare],
                        &["with", "allow_depot_remote_cache"],
                        Node::Scalar("true".into()),
                    ),
                    "github" => h::replace(
                        &mut steps[prepare],
                        &["with", "allow_native_github_cache"],
                        Node::Scalar("true".into()),
                    ),
                    "conditional" | "optional" => {
                        let Node::Map(fields) = &mut steps[prepare] else {
                            unreachable!()
                        };
                        fields.push((
                            if mutation == "conditional" {
                                "if"
                            } else {
                                "continue-on-error"
                            }
                            .into(),
                            Node::Scalar("true".into()),
                        ));
                    }
                    _ => unreachable!(),
                }
                assert!(validate(node).is_err(), "{job}: {mutation}");
            }
        }
    }

    #[test]
    fn native_selector_commands_require_executable_arguments_and_bound_environment() {
        for job in ["attempt_1", "attempt_2", "attempt_3", "result"] {
            for mutation in ["comment", "legacy", "executable", "argument", "environment"] {
                let mut node = document();
                let steps = selector_steps(&mut node, job);
                let command = steps
                    .iter_mut()
                    .find(|step| {
                        super::super::field(step, "id")
                            == Some(if job == "result" { "result" } else { "select" })
                    })
                    .unwrap();
                let original = super::super::field(command, "run").unwrap().to_owned();
                let changed = match mutation {
                    "comment" => format!("# {}\necho skipped", original.replace('\n', " ")),
                    "legacy" => original.replace(
                        "\"$MESH_LLM_AUTOMATION_BIN\" automation canary-receipts select-",
                        "python3 scripts/llama-canary-select-attempt.py ",
                    ),
                    "executable" => original.replace("$MESH_LLM_AUTOMATION_BIN", "$UNTRUSTED_BIN"),
                    "argument" => original.replace(
                        if job == "result" {
                            "--attempts-json"
                        } else {
                            "--verification-json"
                        },
                        "--unknown-input",
                    ),
                    "environment" => {
                        h::replace(
                            command,
                            &[
                                "env",
                                if job == "result" {
                                    "ATTEMPTS_JSON"
                                } else {
                                    "VERIFY_JSON"
                                },
                            ],
                            Node::Scalar(if job == "result" {
                                "{}".into()
                            } else {
                                format!(
                                    "${{{{ toJSON(needs.repair_{}) }}}}",
                                    job.trim_start_matches("attempt_")
                                )
                            }),
                        );
                        original
                    }
                    _ => unreachable!(),
                };
                *h::mutable(command, "run") = Node::Scalar(changed);
                assert!(validate(node).is_err(), "{job}: {mutation}");
            }
        }
    }
}
