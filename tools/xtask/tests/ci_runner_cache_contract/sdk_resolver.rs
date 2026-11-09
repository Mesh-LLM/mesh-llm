//! Execute the real selector and SDK resolver, without compiling SDK products.
use super::{
    support::{Fixture, root},
    workflow_yaml::{self, Node},
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    process::{Command, Output},
};

#[derive(Clone, Copy)]
struct Case {
    event: &'static str,
    original_event: &'static str,
    reference: &'static str,
    repository: &'static str,
    head_repository: &'static str,
    main: &'static str,
    manual: &'static str,
    pr: &'static str,
    force_hosted: &'static str,
    date: &'static str,
    canary_reference: &'static str,
    target: &'static str,
    size: &'static str,
}
impl Default for Case {
    fn default() -> Self {
        Self {
            event: "push",
            original_event: "",
            reference: "refs/heads/main",
            repository: "Mesh-LLM/mesh-llm",
            head_repository: "Mesh-LLM/mesh-llm",
            main: "true",
            manual: "false",
            pr: "false",
            force_hosted: "false",
            date: "2026-09-03",
            canary_reference: "",
            target: "x86_64-unknown-linux-gnu",
            size: "8",
        }
    }
}
fn workflows() -> [&'static str; 2] {
    ["native-sdk-artifact.yml", "static-abi-artifact.yml"]
}
fn resolve_step(name: &str) -> Node {
    let source = fs::read_to_string(root().join(".github/workflows").join(name)).unwrap();
    let document = workflow_yaml::parse(&source).unwrap();
    let policy = document.get("jobs").unwrap().get("runner_policy").unwrap();
    let Node::Seq(steps) = policy.get("steps").unwrap() else {
        panic!("steps must be a sequence")
    };
    let found = steps
        .iter()
        .filter(|n| n.get("id").and_then(Node::text) == Some("resolve"))
        .collect::<Vec<_>>();
    assert_eq!(found.len(), 1);
    found[0].clone()
}
fn execute(name: &str, case: Case) -> (Output, BTreeMap<String, String>, Value) {
    let fixture = Fixture::new();
    let selected = fixture.selector(&[
        ("INPUT_EVENT_NAME", case.event),
        ("INPUT_ORIGINAL_EVENT_NAME", case.original_event),
        ("DISPATCH_ORIGINAL_EVENT_NAME", case.original_event),
        ("INPUT_REF", case.reference),
        ("INPUT_REPOSITORY", case.repository),
        ("INPUT_HEAD_REPOSITORY", case.head_repository),
        ("INPUT_DEPOT_MAIN_ENABLED", case.main),
        ("INPUT_DEPOT_PR_ENABLED", case.pr),
        ("INPUT_MANUAL_USE_DEPOT", case.manual),
        ("INPUT_FORCE_HOSTED", case.force_hosted),
        ("FIXTURE_DATE", case.date),
        ("INPUT_PR_CANARY_REF", case.canary_reference),
    ]);
    assert!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let selection = fixture.outputs();
    let step = resolve_step(name);
    let mut command = Command::new("/bin/bash");
    let output = fixture.path().join("resolved-output");
    command
        .env_clear()
        .current_dir(fixture.path())
        .env("PATH", "/usr/bin:/bin")
        .env("GITHUB_OUTPUT", &output)
        .env("TARGET", case.target)
        .env("RUNNER_SIZE", case.size)
        .env("POLICY_EVENT_NAME", case.event);
    for (variable, key) in [
        ("RUNNER_DEFAULT", "runner"),
        ("RUNNER_4", "runner_4"),
        ("RUNNER_8", "runner_8"),
        ("RUNNER_16", "runner_16"),
        ("RUNNER_ARM", "runner_arm"),
        ("RUNNER_ARM_4", "runner_arm_4"),
        ("RUNNER_ARM_8", "runner_arm_8"),
        ("RUNNER_ARM_16", "runner_arm_16"),
        ("RUNNER_MACOS", "runner_macos"),
        ("ALLOW_DEPOT_REMOTE_CACHE", "allow_depot_remote_cache"),
        ("ALLOW_NATIVE_GITHUB_CACHE", "allow_native_github_cache"),
    ] {
        command.env(variable, selection[key].as_str().unwrap());
    }
    command.args(["-c", step.get("run").and_then(Node::text).unwrap()]);
    let result = fixture.run(command);
    let mut values = BTreeMap::new();
    if output.exists() {
        for line in fs::read_to_string(output).unwrap().lines() {
            let (key, value) = line.split_once('=').unwrap();
            assert!(
                values.insert(key.into(), value.into()).is_none(),
                "duplicate resolver output"
            );
        }
    }
    (result, values, selection)
}
fn assert_selection(name: &str, case: Case, runner: &str, native: bool) {
    let (result, values, selection) = execute(name, case);
    assert!(
        result.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        values.get("runner").map(String::as_str),
        Some(runner),
        "{name}"
    );
    if name == "native-sdk-artifact.yml" {
        assert_eq!(values["allow_native_github_cache"], native.to_string());
        assert_eq!(values["allow_depot_remote_cache"], "false");
        assert_eq!(values.len(), 3);
    } else {
        assert_eq!(values.len(), 1);
        // This workflow directly projects cache authority from the selector.
        assert_eq!(selection["allow_native_github_cache"], native.to_string());
        assert_eq!(selection["allow_depot_remote_cache"], "false");
    }
}
#[test]
fn actual_sdk_resolvers_keep_untrusted_events_and_external_repositories_hosted() {
    for name in workflows() {
        for case in [
            Case {
                event: "pull_request",
                reference: "refs/pull/12/merge",
                ..Case::default()
            },
            Case {
                event: "pull_request_target",
                ..Case::default()
            },
            Case {
                reference: "refs/tags/v1.2.3",
                ..Case::default()
            },
            Case {
                event: "workflow_dispatch",
                reference: "refs/heads/feature",
                ..Case::default()
            },
            Case {
                repository: "attacker/mesh-llm",
                head_repository: "attacker/mesh-llm",
                ..Case::default()
            },
        ] {
            assert_selection(name, Case { size: "16", ..case }, "ubuntu-24.04", true);
        }
    }
}
#[test]
fn actual_sdk_resolvers_select_bounded_linux_sizes_and_architectures_from_trusted_policy() {
    for name in workflows() {
        for (target, base) in [
            ("x86_64-unknown-linux-gnu", "depot-ubuntu-24.04"),
            ("aarch64-unknown-linux-gnu", "depot-ubuntu-24.04-arm"),
        ] {
            for size in ["default", "4", "8", "16"] {
                let expected = if size == "default" {
                    base.to_owned()
                } else {
                    format!("{base}-{size}")
                };
                assert_selection(
                    name,
                    Case {
                        target,
                        size,
                        ..Case::default()
                    },
                    &expected,
                    false,
                );
            }
        }
        assert_selection(
            name,
            Case {
                main: "false",
                target: "aarch64-unknown-linux-gnu",
                ..Case::default()
            },
            "ubuntu-24.04-arm",
            true,
        );
    }
}
#[test]
fn actual_sdk_resolvers_admit_manual_depot_only_for_trusted_main_dispatch() {
    for name in workflows() {
        assert_selection(
            name,
            Case {
                event: "workflow_dispatch",
                main: "false",
                manual: "true",
                ..Case::default()
            },
            "depot-ubuntu-24.04-8",
            false,
        );
        for case in [
            Case {
                event: "workflow_dispatch",
                main: "false",
                ..Case::default()
            },
            Case {
                event: "workflow_dispatch",
                main: "false",
                manual: "true",
                reference: "refs/heads/feature",
                ..Case::default()
            },
            Case {
                event: "pull_request",
                reference: "refs/pull/12/merge",
                main: "false",
                manual: "true",
                ..Case::default()
            },
            Case {
                main: "false",
                manual: "true",
                ..Case::default()
            },
            Case {
                event: "workflow_dispatch",
                original_event: "pull_request",
                manual: "true",
                ..Case::default()
            },
        ] {
            assert_selection(name, case, "ubuntu-24.04", true);
        }
    }
}
#[test]
fn actual_sdk_resolvers_reject_invalid_size_and_target_without_emitting_runner() {
    for name in workflows() {
        let mut cases = vec![
            Case {
                size: "unbounded",
                ..Case::default()
            },
            Case {
                target: "unexpected-target",
                ..Case::default()
            },
        ];
        if name == "static-abi-artifact.yml" {
            cases.push(Case {
                target: "aarch64-apple-darwin",
                ..Case::default()
            });
        }
        for case in cases {
            let (result, values, _) = execute(name, case);
            assert!(!result.status.success());
            assert!(values.is_empty());
            let diagnostic = if case.size == "unbounded" {
                "runner_size must be one of"
            } else {
                "unsupported"
            };
            assert!(String::from_utf8_lossy(&result.stderr).contains(diagnostic));
        }
    }
}
#[test]
fn actual_sdk_resolvers_bind_direct_pr_exception_to_same_repository_and_preserve_cache_authority() {
    for name in workflows() {
        let admitted = Case {
            event: "pull_request",
            reference: "refs/pull/12/merge",
            main: "false",
            pr: "true",
            ..Case::default()
        };
        assert_selection(name, admitted, "depot-ubuntu-24.04-8", true);
        assert_selection(
            name,
            Case {
                head_repository: "attacker/mesh-llm",
                ..admitted
            },
            "ubuntu-24.04",
            true,
        );
        assert_selection(
            name,
            Case {
                force_hosted: "true",
                ..admitted
            },
            "ubuntu-24.04",
            true,
        );
    }
}
#[test]
fn actual_native_sdk_macos_resolver_preserves_main_hosted_and_direct_pr_provider_policy() {
    assert_selection(
        "native-sdk-artifact.yml",
        Case {
            target: "aarch64-apple-darwin",
            ..Case::default()
        },
        "macos-15",
        true,
    );
    assert_selection(
        "native-sdk-artifact.yml",
        Case {
            target: "aarch64-apple-darwin",
            event: "pull_request",
            reference: "refs/pull/12/merge",
            main: "false",
            pr: "true",
            ..Case::default()
        },
        "depot-macos-15",
        true,
    );
}

#[test]
fn actual_sdk_resolvers_preserve_expiry_and_distinguish_exact_canary_cache_authority() {
    for name in workflows() {
        for date in ["2026-09-14", "2026-10-04"] {
            assert_selection(
                name,
                Case {
                    event: "pull_request",
                    reference: "refs/pull/12/merge",
                    main: "false",
                    pr: "true",
                    date,
                    ..Case::default()
                },
                "ubuntu-24.04",
                true,
            );
        }
        assert_selection(
            name,
            Case {
                pr: "true",
                date: "2026-10-04",
                ..Case::default()
            },
            "depot-ubuntu-24.04-8",
            false,
        );
        assert_selection(
            name,
            Case {
                event: "pull_request",
                reference: "refs/pull/12/merge",
                canary_reference: "refs/pull/12/merge",
                main: "false",
                date: "2026-10-04",
                ..Case::default()
            },
            "depot-ubuntu-24.04-8",
            false,
        );
    }
}
