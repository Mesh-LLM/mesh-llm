//! Execute the checked-in runner mapping steps with projected selector outputs.
use super::{text, workflow};
use crate::support::Fixture;
use crate::workflow_yaml::Node;
use serde_json::Value;
use std::process::Command;

fn resolver_step<'a>(workflow: &'a Node, id: &str) -> &'a Node {
    let steps = workflow
        .get("jobs")
        .unwrap()
        .get("runner_policy")
        .unwrap()
        .get("steps")
        .unwrap();
    let Node::Seq(steps) = steps else {
        panic!("runner policy steps")
    };
    steps
        .iter()
        .find(|step| text(step, "id") == Some(id))
        .unwrap()
}

pub(super) fn check_native_resolver(file: &str, selected: &Value) {
    let called = workflow(file);
    let policy = called.get("jobs").unwrap().get("runner_policy").unwrap();
    let resolver = resolver_step(&called, "resolve");
    let declared = resolver.get("env").unwrap();
    for (key, expected) in [
        ("TARGET", "${{ inputs.target }}"),
        ("RUNNER_SIZE", "${{ inputs.runner_size }}"),
    ] {
        assert_eq!(text(declared, key), Some(expected), "{file}/{key}");
    }
    let fixture = Fixture::new();
    let mut command = Command::new("bash");
    command
        .env_clear()
        .current_dir(fixture.path())
        .env("PATH", std::env::var_os("PATH").unwrap())
        .env("HOME", fixture.path())
        .env("GITHUB_OUTPUT", fixture.path().join("outputs"))
        .env("TARGET", "x86_64-unknown-linux-gnu")
        .env("RUNNER_SIZE", "8");
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
        assert_eq!(
            text(declared, key),
            Some(format!("${{{{ steps.policy.outputs.{output} }}}}").as_str()),
            "{file}/{key}"
        );
        command.env(key, selected[output].as_str().unwrap());
    }
    if file == "native-sdk-artifact.yml" {
        for (key, output) in [
            ("RUNNER_MACOS", "runner_macos"),
            ("ALLOW_DEPOT_REMOTE_CACHE", "allow_depot_remote_cache"),
            ("ALLOW_NATIVE_GITHUB_CACHE", "allow_native_github_cache"),
        ] {
            assert_eq!(
                text(declared, key),
                Some(format!("${{{{ steps.policy.outputs.{output} }}}}").as_str()),
                "{file}/{key}"
            );
            command.env(key, selected[output].as_str().unwrap());
        }
        assert_eq!(
            text(declared, "POLICY_EVENT_NAME"),
            Some("${{ github.event_name }}")
        );
        command.env("POLICY_EVENT_NAME", "workflow_dispatch");
    }
    let outputs = policy.get("outputs").unwrap();
    assert_eq!(
        text(outputs, "runner"),
        Some("${{ steps.resolve.outputs.runner }}")
    );
    let job_name = if file == "native-sdk-artifact.yml" {
        "linux_native_sdk_artifact"
    } else {
        "static_abi_artifact"
    };
    let job = called.get("jobs").unwrap().get(job_name).unwrap();
    assert_eq!(
        text(job, "runs-on"),
        Some("${{ needs.runner_policy.outputs.runner }}")
    );
    command.args(["-c", text(resolver, "run").unwrap()]);
    let result = fixture.run(command);
    assert!(result.status.success(), "{file}: {result:?}");
    let resolved = fixture.outputs();
    assert_eq!(resolved["runner"], selected["runner_8"], "{file}");
    if file == "native-sdk-artifact.yml" {
        for key in ["allow_depot_remote_cache", "allow_native_github_cache"] {
            assert_eq!(resolved[key], selected[key], "{file}/{key}");
            assert_eq!(
                text(outputs, key),
                Some(format!("${{{{ steps.resolve.outputs.{key} }}}}").as_str())
            );
        }
    }
}

pub(super) fn check_platform_resolver(selected: &Value) {
    let called = workflow("ci-platform-checks-slice.yml");
    let policy = called.get("jobs").unwrap().get("runner_policy").unwrap();
    let resolver = resolver_step(&called, "platform_runners");
    let declared = resolver.get("env").unwrap();
    let fixture = Fixture::new();
    let mut command = Command::new("bash");
    command
        .env_clear()
        .current_dir(fixture.path())
        .env("PATH", std::env::var_os("PATH").unwrap())
        .env("HOME", fixture.path())
        .env("GITHUB_OUTPUT", fixture.path().join("outputs"));
    for (key, output) in [
        ("RUNNER_MACOS", "runner_macos"),
        ("RUNNER_WINDOWS", "runner_windows"),
    ] {
        assert_eq!(
            text(declared, key),
            Some(format!("${{{{ steps.policy.outputs.{output} }}}}").as_str())
        );
        command.env(key, selected[output].as_str().unwrap());
    }
    assert_eq!(
        text(policy.get("outputs").unwrap(), "runner_by_platform"),
        Some("${{ steps.platform_runners.outputs.runner_by_platform }}")
    );
    let job = called.get("jobs").unwrap().get("platform_checks").unwrap();
    assert_eq!(
        text(job, "runs-on"),
        Some(
            "${{ fromJSON(needs.runner_policy.outputs.runner_by_platform)[matrix.check.platform] }}"
        )
    );
    command.args(["-c", text(resolver, "run").unwrap()]);
    let result = fixture.run(command);
    assert!(result.status.success(), "platform mapping: {result:?}");
    let outputs = fixture.outputs();
    let mapping: Value =
        serde_json::from_str(outputs["runner_by_platform"].as_str().unwrap()).unwrap();
    assert_eq!(mapping["macos"], selected["runner_macos"]);
    assert_eq!(mapping["windows"], selected["runner_windows"]);
}
