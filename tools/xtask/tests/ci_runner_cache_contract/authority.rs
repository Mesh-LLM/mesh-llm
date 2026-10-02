use super::{
    cache_authority,
    support::root,
    workflow_yaml::{self, Node},
};
use std::fs;
fn current() -> Node {
    let document = workflow_yaml::parse(
        &fs::read_to_string(root().join(".github/workflows/ci-quality-slice.yml")).unwrap(),
    )
    .unwrap();
    document
        .get("jobs")
        .unwrap()
        .get("authority_sentinel")
        .unwrap()
        .clone()
}
fn replace(node: &mut Node, key: &str, value: Node) {
    let Node::Map(entries) = node else {
        panic!("mapping")
    };
    if let Some((_, v)) = entries.iter_mut().find(|(k, _)| k == key) {
        *v = value;
    } else {
        entries.push((key.into(), value));
    }
}

#[test]
fn sentinel_exemption_has_no_checkout_no_permissions_and_attests_before_pinned_cache() {
    assert!(cache_authority::sentinel(&current()).is_ok());
}
#[test]
fn sentinel_mutations_cannot_expand_credentials_ref_scope_or_ordinary_cache_gates() {
    for (key, value) in [
        ("if", Node::Scalar("true".into())),
        ("needs", Node::Scalar("untrusted_job".into())),
        (
            "permissions",
            Node::Map(vec![("contents".into(), Node::Scalar("read".into()))]),
        ),
        ("runs-on", Node::Scalar("ubuntu-24.04".into())),
    ] {
        let mut job = current();
        replace(&mut job, key, value);
        assert!(cache_authority::sentinel(&job).is_err());
    }
    for change in [
        "checkout",
        "local_audit",
        "unpinned",
        "before_attestation",
        "cache_gate",
        "endpoint_comment",
        "verify_after",
        "verify_source_pr",
        "verify_artifact_literal",
        "verify_digest_literal",
        "verify_empty",
        "endpoint_owner_override",
    ] {
        let mut job = current();
        mutate_steps(&mut job, change);
        assert!(cache_authority::sentinel(&job).is_err(), "{change}");
    }
}

fn mutate_steps(job: &mut Node, change: &str) {
    let Node::Map(entries) = job else { panic!() };
    let Node::Seq(steps) = &mut entries.iter_mut().find(|(k, _)| k == "steps").unwrap().1 else {
        panic!()
    };
    match change {
        "checkout" => steps.push(Node::Map(vec![(
            "uses".into(),
            Node::Scalar("actions/checkout@fixture".into()),
        )])),
        "local_audit" => steps.push(Node::Map(vec![(
            "uses".into(),
            Node::Scalar("./.github/actions/audit-depot-pr-isolation".into()),
        )])),
        "unpinned" => {
            let step = steps
                .iter_mut()
                .find(|s| {
                    s.get("uses")
                        .and_then(Node::text)
                        .is_some_and(|s| s.starts_with("actions/cache/"))
                })
                .unwrap();
            replace(step, "uses", Node::Scalar("actions/cache/save@main".into()));
        }
        "before_attestation" => {
            let at = steps
                .iter()
                .position(|s| {
                    s.get("uses")
                        .and_then(Node::text)
                        .is_some_and(|s| s.starts_with("actions/cache/"))
                })
                .unwrap();
            let cache = steps.remove(at);
            steps.insert(0, cache);
        }
        "endpoint_comment" => {
            let step = steps
                .iter_mut()
                .find(|s| {
                    s.get("name").and_then(Node::text)
                        == Some("Attest provider-injected cache backend")
                })
                .unwrap();
            replace(step, "run", Node::Scalar("set -euo pipefail\n# \"$MESH_LLM_AUTOMATION_BIN\" ci-ops authority-audit endpoint ACTIONS_CACHE_URL\n# \"$MESH_LLM_AUTOMATION_BIN\" ci-ops authority-audit endpoint ACTIONS_RESULTS_URL".into()));
        }
        "endpoint_owner_override" => {
            let step = steps
                .iter_mut()
                .find(|s| {
                    s.get("name").and_then(Node::text)
                        == Some("Attest provider-injected cache backend")
                })
                .unwrap();
            replace(
                step,
                "env",
                Node::Map(vec![(
                    "MESH_LLM_AUTOMATION_BIN".into(),
                    Node::Scalar("${{ inputs.source_sha }}".into()),
                )]),
            );
        }
        "verify_after" => {
            let at = steps
                .iter()
                .position(|s| {
                    s.get("name").and_then(Node::text)
                        == Some("Verify protected authority automation without executing it")
                })
                .unwrap();
            let verify = steps.remove(at);
            steps.push(verify);
        }
        "verify_source_pr"
        | "verify_artifact_literal"
        | "verify_digest_literal"
        | "verify_empty" => {
            let step = steps
                .iter_mut()
                .find(|s| {
                    s.get("name").and_then(Node::text)
                        == Some("Verify protected authority automation without executing it")
                })
                .unwrap();
            if change == "verify_empty" {
                replace(step, "run", Node::Scalar(String::new()));
            } else {
                let Node::Map(fields) = step else { panic!() };
                let environment = &mut fields.iter_mut().find(|(key, _)| key == "env").unwrap().1;
                let (key, value) = match change {
                    "verify_source_pr" => (
                        "AUTOMATION_SOURCE_SHA",
                        "${{ github.event.pull_request.head.sha }}",
                    ),
                    "verify_artifact_literal" => ("AUTOMATION_ARTIFACT_ID", "1234"),
                    _ => (
                        "AUTOMATION_BINARY_SHA256",
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    ),
                };
                replace(environment, key, Node::Scalar(value.into()));
            }
        }
        "cache_gate" => replace(
            &mut steps[0],
            "if",
            Node::Scalar(
                "${{ needs.runner_policy.outputs.allow_native_github_cache == 'true' }}".into(),
            ),
        ),
        _ => unreachable!(),
    }
}

#[test]
fn authority_producer_requires_protected_checkout_output_projection_and_clean_hosted_profile() {
    let document = workflow_yaml::parse(
        &fs::read_to_string(root().join(".github/workflows/ci-quality-slice.yml")).unwrap(),
    )
    .unwrap();
    let producer = document.get("jobs").unwrap().get("runner_policy").unwrap();
    cache_authority::producer(producer).unwrap();
    for change in ["projection", "checkout", "profile", "source"] {
        let mut job = producer.clone();
        let Node::Map(fields) = &mut job else {
            panic!()
        };
        if change == "projection" {
            let outputs = &mut fields
                .iter_mut()
                .find(|(key, _)| key == "outputs")
                .unwrap()
                .1;
            replace(
                outputs,
                "authority_automation_source_sha",
                Node::Scalar("${{ inputs.source_sha }}".into()),
            );
        } else {
            let Node::Seq(steps) =
                &mut fields.iter_mut().find(|(key, _)| key == "steps").unwrap().1
            else {
                panic!()
            };
            let step = if change == "checkout" {
                steps
                    .iter_mut()
                    .find(|s| {
                        s.get("uses")
                            .and_then(Node::text)
                            .is_some_and(|v| v.starts_with("actions/checkout@"))
                    })
                    .unwrap()
            } else {
                steps
                    .iter_mut()
                    .find(|s| s.get("id").and_then(Node::text) == Some("authority_upload"))
                    .unwrap()
            };
            let Node::Map(fields) = step else { panic!() };
            let inputs = &mut fields.iter_mut().find(|(key, _)| key == "with").unwrap().1;
            let (key, value) = match change {
                "checkout" => ("ref", "${{ github.event.pull_request.head.sha }}"),
                "profile" => ("runner-profile", "native"),
                _ => ("source-profile", "selected"),
            };
            replace(inputs, key, Node::Scalar(value.into()));
        }
        assert!(cache_authority::producer(&job).is_err(), "{change}");
    }
}
