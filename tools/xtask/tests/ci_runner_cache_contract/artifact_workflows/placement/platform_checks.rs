//! Platform-check callers and the actual selector-to-platform map.

use super::super as workflows;
use super::{plan, support};
use std::fs;
use std::process::Command;

#[test]
fn main_platform_checks_keep_caller_gates_and_runner_mapping() {
    let planned = plan(
        "main",
        "push",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "README.md",
    );
    assert_eq!(planned["signals"]["force_hosted"], false);
    for (platform, row_id, role) in [
        ("macos", "macos-portable", "macos-build"),
        ("windows", "windows-portable", "windows-build"),
    ] {
        let row = planned["matrices"]["platform_checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == row_id)
            .unwrap_or_else(|| panic!("missing planned check {row_id}"));
        assert_eq!(row["runner_role"], role);
        let lane = workflows::document(&format!("ci-{platform}-lane.yml"));
        let caller = workflows::job(&lane, "platform_checks");
        assert_eq!(
            workflows::text(caller, "uses"),
            Some("./.github/workflows/ci-platform-checks-slice.yml")
        );
        for (key, expected) in [
            (
                "force_hosted",
                "${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}",
            ),
            (
                "platform_checks_matrix",
                "${{ toJson(fromJson(inputs.lane_plan_json).matrices.platform_checks) }}",
            ),
            ("source_sha", "${{ inputs.source_sha }}"),
            ("original_event_name", "${{ inputs.original_event_name }}"),
        ] {
            assert_eq!(
                workflows::input(caller, key),
                Some(expected),
                "{platform}/{key}"
            );
        }
    }

    let slice = workflows::document("ci-platform-checks-slice.yml");
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
    let mapper = workflows::steps(policy)
        .iter()
        .find(|step| workflows::text(step, "id") == Some("platform_runners"))
        .unwrap();
    let env = mapper.get("env").unwrap();
    assert_eq!(
        workflows::text(env, "RUNNER_MACOS"),
        Some("${{ steps.policy.outputs.runner_macos }}")
    );
    assert_eq!(
        workflows::text(env, "RUNNER_WINDOWS"),
        Some("${{ steps.policy.outputs.runner_windows }}")
    );
    assert_eq!(
        workflows::text(policy.get("outputs").unwrap(), "runner_by_platform"),
        Some("${{ steps.platform_runners.outputs.runner_by_platform }}")
    );
    assert_eq!(
        workflows::text(workflows::job(&slice, "platform_checks"), "runs-on"),
        Some(
            "${{ fromJSON(needs.runner_policy.outputs.runner_by_platform)[matrix.check.platform] }}"
        )
    );

    let fixture = support::Fixture::new();
    let selected = fixture.selector(&[
        ("INPUT_EVENT_NAME", "push"),
        ("INPUT_ORIGINAL_EVENT_NAME", "push"),
        ("INPUT_REF", "refs/heads/main"),
        ("INPUT_DEPOT_MAIN_ENABLED", "false"),
        ("INPUT_FORCE_HOSTED", "false"),
    ]);
    assert!(selected.status.success(), "{selected:?}");
    let outputs = fixture.outputs();
    let map_file = fixture.path().join("runner-map-output");
    let mut command = Command::new("bash");
    command
        .arg("-c")
        .arg(workflows::text(mapper, "run").unwrap())
        .env("GITHUB_OUTPUT", &map_file)
        .env("RUNNER_MACOS", outputs["runner_macos"].as_str().unwrap())
        .env(
            "RUNNER_WINDOWS",
            outputs["runner_windows"].as_str().unwrap(),
        );
    let mapped = fixture.run(command);
    assert!(mapped.status.success(), "{mapped:?}");
    let text = fs::read_to_string(map_file).unwrap();
    let json = text.strip_prefix("runner_by_platform=").unwrap().trim_end();
    let map: serde_json::Value = serde_json::from_str(json).unwrap();
    assert_eq!(map["macos"], "macos-15");
    assert_eq!(map["windows"], "windows-2022");
    fixture.0.close().unwrap();
}
