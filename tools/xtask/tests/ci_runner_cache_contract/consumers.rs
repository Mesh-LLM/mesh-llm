use super::{
    cache_boundaries, cache_callers, cache_consumers, cache_predicate,
    support::root,
    workflow_yaml::{self, Node},
};
use std::{collections::BTreeMap, fs};
fn current() -> BTreeMap<String, Node> {
    fs::read_dir(root().join(".github/workflows"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|v| v.to_str()),
                Some("yml" | "yaml")
            )
        })
        .map(|path| {
            (
                path.file_name().unwrap().to_str().unwrap().to_owned(),
                workflow_yaml::parse(&fs::read_to_string(path).unwrap()).unwrap(),
            )
        })
        .collect()
}
fn replace(node: &mut Node, key: &str, value: &str) {
    let Node::Map(entries) = node else {
        panic!("map")
    };
    if let Some((_, field)) = entries.iter_mut().find(|(name, _)| name == key) {
        *field = Node::Scalar(value.into());
    } else {
        entries.push((key.into(), Node::Scalar(value.into())));
    }
}
fn each_step(document: &mut Node, visit: &mut impl FnMut(&mut Node)) {
    let Node::Map(jobs) = document.get_mut_jobs() else {
        panic!("jobs")
    };
    for (_, job) in jobs {
        let Node::Map(fields) = job else {
            continue;
        };
        let Some((_, Node::Seq(steps))) = fields.iter_mut().find(|(k, _)| k == "steps") else {
            continue;
        };
        for step in steps {
            visit(step);
        }
    }
}
trait JobMap {
    fn get_mut_jobs(&mut self) -> &mut Node;
}
impl JobMap for Node {
    fn get_mut_jobs(&mut self) -> &mut Node {
        let Node::Map(fields) = self else {
            panic!("workflow")
        };
        &mut fields.iter_mut().find(|(k, _)| k == "jobs").unwrap().1
    }
}
fn inputs(step: &mut Node) -> &mut Node {
    let Node::Map(fields) = step else {
        panic!("step")
    };
    &mut fields.iter_mut().find(|(k, _)| k == "with").unwrap().1
}
fn action(step: &Node) -> &str {
    step.get("uses").and_then(Node::text).unwrap_or("")
}
#[test]
fn current_consumers_require_central_authority_without_suppressing_workload_jobs() {
    cache_consumers::check(&current()).unwrap();
    cache_callers::check(&current()).unwrap();
    cache_boundaries::check(&current()).unwrap();
    for name in ["restore-windows-abi-cache", "setup-windows-rocm-sdk"] {
        let path = root().join(format!(".github/actions/{name}/action.yml"));
        let document = workflow_yaml::parse(&fs::read_to_string(path).unwrap()).unwrap();
        cache_callers::nested_windows(&document).unwrap();
    }
}
#[test]
fn cache_authority_proof_rejects_disjunction_truthy_strings_negation_and_comment_decoys() {
    let clause = "needs.runner_policy.outputs.allow_native_github_cache == 'true'";
    for valid in [
        clause.to_owned(),
        format!("({clause}) && success()"),
        format!("(x && {clause}) || (y && {clause})"),
        format!("{clause} && 'pnpm' || ''"),
    ] {
        assert!(cache_predicate::requires(&valid, clause), "{valid}");
    }
    for invalid in [
        format!("{clause} || true"),
        format!("!({clause})"),
        "needs.runner_policy.outputs.allow_native_github_cache".into(),
        format!("contains('comment {clause}', 'true')"),
        format!("(x && {clause}) || y"),
    ] {
        assert!(!cache_predicate::requires(&invalid, clause), "{invalid}");
    }
}
#[test]
fn changed_cache_consumer_gate_or_either_policy_output_fails_closed() {
    for mutation in ["gate", "native", "remote", "save", "head"] {
        let mut workflows = current();
        let document = workflows.get_mut("ci-macos-host-slice.yml").unwrap();
        let mut changed = false;
        each_step(document, &mut |step| {
            if mutation == "head" && action(step) == "./.github/actions/select-ci-runners" {
                replace(inputs(step), "head_sha", "${{ github.sha }}");
                changed = true;
            }
            if action(step).starts_with("Swatinem/rust-cache@") {
                if mutation == "gate" {
                    replace(step, "if", "${{ true }}");
                    changed = true;
                }
                if mutation == "save" {
                    replace(
                        inputs(step),
                        "save-if",
                        "${{ github.ref == 'refs/heads/main' }}",
                    );
                    changed = true;
                }
            }
        });
        if matches!(mutation, "native" | "remote") {
            each_step(
                workflows.get_mut("ci-quality-slice.yml").unwrap(),
                &mut |step| {
                    if action(step) == "./.github/actions/configure-sccache-gha" {
                        let flag = if mutation == "native" {
                            "allow_native_github_cache"
                        } else {
                            "allow_depot_remote_cache"
                        };
                        replace(inputs(step), flag, "true");
                        changed = true;
                    }
                },
            );
        }
        assert!(changed, "mutation must reach actual consumer {mutation}");
        assert!(
            cache_consumers::check(&workflows).is_err()
                || cache_callers::check(&workflows).is_err(),
            "{mutation}"
        );
    }
}
#[test]
fn swift_package_cache_cannot_enable_on_a_policy_denied_branch() {
    let mut workflows = current();
    let mut changed = false;
    each_step(
        workflows.get_mut("swift-sdk-artifact.yml").unwrap(),
        &mut |step| {
            if action(step).starts_with("actions/setup-node@") {
                replace(
                    inputs(step),
                    "cache",
                    "${{ needs.runner_policy.outputs.allow_native_github_cache == 'true' && 'pnpm' || 'pnpm' }}",
                );
                changed = true;
            }
        },
    );
    assert!(changed);
    assert!(cache_consumers::check(&workflows).is_err());
}

#[test]
fn release_depot_exclusion_and_native_namespace_cannot_be_widened() {
    let mut workflows = current();
    each_step(workflows.get_mut("release.yml").unwrap(), &mut |step| {
        if step.get("name").and_then(Node::text) == Some("Cache native runtime ROCm backend build")
        {
            replace(step, "if", "${{ true }}");
        }
    });
    assert!(cache_boundaries::check(&workflows).is_err());
    let mut workflows = current();
    let Node::Map(fields) = workflows.get_mut("ci-linux-host-slice.yml").unwrap() else {
        panic!()
    };
    let env = &mut fields.iter_mut().find(|(key, _)| key == "env").unwrap().1;
    replace(env, "CACHE_NAMESPACE", "mesh-llm-pr");
    assert!(cache_boundaries::check(&workflows).is_err());
}

#[test]
fn native_sdk_resolver_preserves_provider_cache_outputs_and_explicit_hosted_macos_exception() {
    use super::support::Fixture;
    use std::process::Command;
    let workflows = current();
    let resolver = workflows["native-sdk-artifact.yml"]
        .get("jobs")
        .unwrap()
        .get("runner_policy")
        .unwrap();
    let Node::Seq(steps) = resolver.get("steps").unwrap() else {
        panic!()
    };
    let script = steps
        .iter()
        .find(|s| s.get("id").and_then(Node::text) == Some("resolve"))
        .unwrap()
        .get("run")
        .and_then(Node::text)
        .unwrap();
    for (target, event, native, expected_runner, expected_native) in [
        (
            "x86_64-unknown-linux-gnu",
            "pull_request",
            "false",
            "linux4",
            "false",
        ),
        (
            "aarch64-unknown-linux-gnu",
            "pull_request",
            "false",
            "arm4",
            "false",
        ),
        (
            "aarch64-apple-darwin",
            "pull_request",
            "false",
            "depot-macos",
            "false",
        ),
        (
            "aarch64-apple-darwin",
            "workflow_dispatch",
            "false",
            "macos-15",
            "true",
        ),
    ] {
        let fixture = Fixture::new();
        let output_path = fixture.path().join("outputs");
        let mut command = Command::new("/bin/bash");
        command
            .args(["-c", script])
            .env("TARGET", target)
            .env("RUNNER_SIZE", "4")
            .env("POLICY_EVENT_NAME", event)
            .env("ALLOW_NATIVE_GITHUB_CACHE", native)
            .env("ALLOW_DEPOT_REMOTE_CACHE", "false")
            .env("GITHUB_OUTPUT", &output_path);
        for (key, value) in [
            ("RUNNER_DEFAULT", "linux"),
            ("RUNNER_4", "linux4"),
            ("RUNNER_8", "linux8"),
            ("RUNNER_16", "linux16"),
            ("RUNNER_ARM", "arm"),
            ("RUNNER_ARM_4", "arm4"),
            ("RUNNER_ARM_8", "arm8"),
            ("RUNNER_ARM_16", "arm16"),
            ("RUNNER_MACOS", "depot-macos"),
        ] {
            command.env(key, value);
        }
        let output = fixture.run(command);
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            fs::read_to_string(output_path).unwrap(),
            format!(
                "runner={expected_runner}\nallow_depot_remote_cache=false\nallow_native_github_cache={expected_native}\n"
            )
        );
    }
}

#[test]
fn replacing_the_central_decision_with_a_literal_cannot_authorize_consumers() {
    let mut workflows = current();
    let Node::Map(jobs) = workflows
        .get_mut("ci-quality-slice.yml")
        .unwrap()
        .get_mut_jobs()
    else {
        panic!()
    };
    let policy = &mut jobs
        .iter_mut()
        .find(|(name, _)| name == "runner_policy")
        .unwrap()
        .1;
    let Node::Map(fields) = policy else { panic!() };
    let outputs = &mut fields
        .iter_mut()
        .find(|(name, _)| name == "outputs")
        .unwrap()
        .1;
    replace(outputs, "allow_native_github_cache", "true");
    assert!(cache_consumers::check(&workflows).is_err());
}
