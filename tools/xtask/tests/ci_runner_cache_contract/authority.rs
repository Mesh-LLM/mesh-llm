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

// Append to existing ci_runner_cache_contract/authority.rs (already registered integration target).
fn identity_script(workflow: &str, job: &str) -> String {
    let node = workflow_yaml::parse(
        &fs::read_to_string(root().join(".github/workflows").join(workflow)).unwrap(),
    )
    .unwrap();
    let Node::Seq(steps) = node
        .get("jobs")
        .unwrap()
        .get(job)
        .unwrap()
        .get("steps")
        .unwrap()
    else {
        panic!("steps")
    };
    let step = steps
        .iter()
        .find(|step| step.get("id").and_then(Node::text) == Some("validate"))
        .unwrap();
    assert_eq!(step.get("shell").and_then(Node::text), Some("bash"));
    step.get("run").and_then(Node::text).unwrap().to_owned()
}

fn execute_identity(script: &str, fields: &[(&str, &str)]) -> (bool, String) {
    use crate::process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    };
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("outputs");
    let mut environment = std::collections::BTreeMap::from([
        ("PATH".into(), Value::Public("/usr/bin:/bin".into())),
        ("GITHUB_OUTPUT".into(), Value::Public(output.clone().into())),
    ]);
    for (key, value) in fields {
        environment.insert((*key).into(), Value::Public((*value).into()));
    }
    let captured = process::supervise_raw(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: directory.path().to_owned(),
            arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
            environment,
        },
        &Limits {
            execution: std::time::Duration::from_secs(3),
            graceful_shutdown: std::time::Duration::from_millis(100),
            forced_shutdown: std::time::Duration::from_millis(100),
            retained_bytes_per_stream: 16384,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(16384),
            stderr: std::num::NonZeroUsize::new(16384),
        },
    )
    .unwrap();
    let report = captured.process;
    let raw_stdout = captured.stdout.unwrap();
    let raw_stderr = captured.stderr.unwrap();
    assert_eq!(raw_stdout.as_bytes().len() as u64, report.stdout.bytes_seen);
    assert_eq!(raw_stderr.as_bytes().len() as u64, report.stderr.bytes_seen);
    assert!(report.cleanup.failure.is_none());
    assert!(!report.cleanup.graceful_signal_failed);
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert!(report.cleanup.complete && !report.cleanup.forced && report.failure.is_none());
    assert!(!report.stdout.truncated && !report.stderr.truncated);
    assert_eq!(report.stdout.suppressed_lines, 0);
    assert_eq!(report.stderr.suppressed_lines, 0);
    assert!(report.stdout.bytes_retained.is_empty());
    let values = fs::read_to_string(&output).unwrap_or_default();
    let accepted = report.status.unwrap().success();
    directory.close().unwrap();
    (accepted, values)
}

#[test]
fn actual_three_sentinel_identity_bodies_enforce_canonical_inputs_and_derive_fixed_keys() {
    let id = "0123456789abcdef0123456789abcdef";
    for (workflow, job, expected) in [
        (
            "ci-quality-slice.yml",
            "authority_sentinel",
            format!(
                "sentinel_id={id}\npr_number=42\nseed_key=mesh-llm-depot-authority-seed-v1-{id}\npoison_key=mesh-llm-depot-authority-pr-v1-{id}-pr-42\n"
            ),
        ),
        (
            "depot-canary.yml",
            "seed_authority_marker",
            format!(
                "sentinel_id={id}\nseed_key=mesh-llm-depot-authority-seed-v1-{id}\npoison_key=mesh-llm-depot-authority-pr-v1-{id}-pr-42\n"
            ),
        ),
        (
            "depot-canary.yml",
            "verify_pr_write",
            format!(
                "sentinel_id={id}\npr_number=42\npoison_key=mesh-llm-depot-authority-pr-v1-{id}-pr-42\n"
            ),
        ),
    ] {
        let script = identity_script(workflow, job);
        let fields = [
            ("SENTINEL_ID", id),
            ("PR_NUMBER", "42"),
            ("CONFIGURED_SENTINEL_ID", id),
            ("CONFIGURED_SENTINEL_REF", "refs/pull/42/merge"),
        ];
        assert_eq!(execute_identity(&script, &fields), (true, expected));
        for (key, value) in [
            ("SENTINEL_ID", ""),
            ("SENTINEL_ID", "ABCDEF0123456789abcdef0123456789"),
            ("SENTINEL_ID", "0000000000000000000000000000000"),
            ("SENTINEL_ID", "000000000000000000000000000000000"),
            ("PR_NUMBER", ""),
            ("PR_NUMBER", "0"),
            ("PR_NUMBER", "01"),
            ("PR_NUMBER", "+1"),
            ("PR_NUMBER", " 1"),
            ("PR_NUMBER", "1 "),
            ("PR_NUMBER", "1111111111"),
            ("PR_NUMBER", "43"),
            ("CONFIGURED_SENTINEL_REF", "refs/pull/43/merge"),
        ] {
            let mut changed = fields;
            changed.iter_mut().find(|(name, _)| *name == key).unwrap().1 = value;
            assert_eq!(
                execute_identity(&script, &changed),
                (false, String::new()),
                "{job}:{key}"
            );
        }
        if job != "authority_sentinel" {
            let mut changed = fields;
            changed[2].1 = "fedcba9876543210fedcba9876543210";
            assert_eq!(execute_identity(&script, &changed), (false, String::new()));
        }
    }
}
