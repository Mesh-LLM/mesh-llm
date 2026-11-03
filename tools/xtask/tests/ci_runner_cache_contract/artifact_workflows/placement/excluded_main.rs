//! Main-path fixtures for callers outside the trusted-main Depot allowlist.

use super::super as workflows;
use super::{plan, support};
use serde_json::Value;

fn main_plan() -> Value {
    let planned = plan(
        "main",
        "push",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "README.md",
    );
    assert_eq!(planned["signals"]["force_hosted"], false);
    planned
}

fn require_slice(planned: &Value, name: &str) {
    assert!(
        planned["required_slices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|slice| slice == name),
        "{name}"
    );
}

fn lane_caller(file: &str, job: &str, uses: &str) -> workflows::Node {
    let lane = workflows::document(file);
    let caller = workflows::job(&lane, job);
    assert_eq!(workflows::text(caller, "uses"), Some(uses));
    for (key, expected) in [
        (
            "force_hosted",
            "${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}",
        ),
        ("source_sha", "${{ inputs.source_sha }}"),
        ("original_event_name", "${{ inputs.original_event_name }}"),
    ] {
        assert_eq!(
            workflows::input(caller, key),
            Some(expected),
            "{file}/{job}/{key}"
        );
    }
    lane
}

fn hosted_selector(
    planned: &Value,
    file: &str,
    output: &str,
    jobs: &[&str],
    hosted: &str,
) -> workflows::Node {
    let slice = workflows::document(file);
    let policy = workflows::job(&slice, "runner_policy");
    let selector = workflows::steps(policy)
        .iter()
        .find(|step| workflows::text(step, "id") == Some("policy"))
        .unwrap();
    assert_eq!(
        workflows::text(selector, "uses"),
        Some("./.github/actions/select-ci-runners")
    );
    assert_eq!(
        workflows::input(selector, "depot_main_enabled"),
        Some("false")
    );
    assert_eq!(
        workflows::input(selector, "force_hosted"),
        Some("${{ inputs.force_hosted }}")
    );
    assert_eq!(
        workflows::text(policy.get("outputs").unwrap(), output),
        Some(format!("${{{{ steps.policy.outputs.{output} }}}}").as_str())
    );
    for job in jobs {
        assert_eq!(
            workflows::text(workflows::job(&slice, job), "runs-on"),
            Some(format!("${{{{ needs.runner_policy.outputs.{output} }}}}").as_str()),
            "{file}/{job}"
        );
    }
    let fixture = support::Fixture::new();
    let force = planned["signals"]["force_hosted"].as_bool().unwrap();
    let result = fixture.selector(&[
        ("INPUT_EVENT_NAME", "push"),
        ("INPUT_ORIGINAL_EVENT_NAME", "push"),
        ("INPUT_REF", "refs/heads/main"),
        ("INPUT_DEPOT_MAIN_ENABLED", "false"),
        ("INPUT_FORCE_HOSTED", if force { "true" } else { "false" }),
    ]);
    assert!(result.status.success(), "{file}: {result:?}");
    assert_eq!(fixture.outputs()[output], hosted, "{file}");
    fixture.0.close().unwrap();
    slice
}

#[test]
fn main_rust_tests_keep_batch_and_static_abi_wiring() {
    let planned = main_plan();
    require_slice(&planned, "rust-tests");
    assert!(
        !planned["matrices"]["rust_tests"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let lane = lane_caller(
        "ci-linux-lane.yml",
        "rust_tests",
        "./.github/workflows/ci-rust-tests-slice.yml",
    );
    let caller = workflows::job(&lane, "rust_tests");
    assert_eq!(
        workflows::input(caller, "rust_tests_matrix"),
        Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.rust_tests) }}")
    );
    assert_eq!(
        workflows::input(caller, "static_abi_artifact_name"),
        Some("ci-static-abi-${{ github.run_id }}")
    );
    hosted_selector(
        &planned,
        "ci-rust-tests-slice.yml",
        "runner",
        &["rust_tests"],
        "ubuntu-24.04",
    );
}

#[test]
fn main_ui_artifact_callers_keep_platform_names() {
    let planned = main_plan();
    require_slice(&planned, "ui-artifact");
    for platform in ["linux", "macos", "windows"] {
        let file = format!("ci-{platform}-lane.yml");
        let lane = lane_caller(
            &file,
            "ui_artifact",
            "./.github/workflows/ci-ui-artifact-slice.yml",
        );
        assert_eq!(
            workflows::input(workflows::job(&lane, "ui_artifact"), "artifact_name"),
            Some(format!("ci-ui-dist-{platform}-${{{{ github.run_id }}}}").as_str())
        );
    }
    let slice = hosted_selector(
        &planned,
        "ci-ui-artifact-slice.yml",
        "runner_4",
        &["ui_artifact", "ui_stamp"],
        "ubuntu-24.04",
    );
    let upload = workflows::named(
        workflows::job(&slice, "ui_artifact"),
        "Upload immutable console distribution",
    );
    assert_eq!(
        workflows::input(upload, "name"),
        Some(
            "${{ inputs.release_tag != '' && format('{0}-raw', inputs.artifact_name) || inputs.artifact_name }}"
        )
    );
}

#[test]
fn main_website_caller_keeps_selected_jobs() {
    let planned = main_plan();
    require_slice(&planned, "web");
    assert_eq!(planned["signals"]["ui_changed"], true);
    assert_eq!(planned["signals"]["website_changed"], true);
    lane_caller(
        "ci-website-lane.yml",
        "web",
        "./.github/workflows/ci-web-slice.yml",
    );
    hosted_selector(
        &planned,
        "ci-web-slice.yml",
        "runner_4",
        &["ui_quality", "ui_e2e", "website"],
        "ubuntu-24.04",
    );
}

#[test]
fn main_swift_caller_keeps_sdk_row_and_artifact_name() {
    let planned = main_plan();
    let swift = planned["matrices"]["sdk"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == "swift")
        .unwrap();
    assert_eq!(swift["runner_role"], "macos-build");
    let lane = lane_caller(
        "ci-macos-lane.yml",
        "swift_sdk_input",
        "./.github/workflows/swift-sdk-artifact.yml",
    );
    assert_eq!(
        workflows::input(workflows::job(&lane, "swift_sdk_input"), "mode"),
        Some(
            "${{ (fromJson(inputs.lane_plan_json).profile == 'main' || fromJson(inputs.lane_plan_json).profile == 'manual-full') && 'full' || 'host-only' }}"
        )
    );
    assert_eq!(
        workflows::input(workflows::job(&lane, "swift_sdk_input"), "artifact_name"),
        Some("ci-sdk-swift")
    );
    let slice = hosted_selector(
        &planned,
        "swift-sdk-artifact.yml",
        "runner_macos",
        &["swift_sdk_target", "swift_sdk_artifact"],
        "macos-15",
    );
    let upload = workflows::named(
        workflows::job(&slice, "swift_sdk_artifact"),
        "Upload immutable Swift SDK input",
    );
    assert_eq!(
        workflows::input(upload, "name"),
        Some("${{ inputs.artifact_name }}")
    );
}
