//! Release selector and nested SDK hosted-override projections.

use super::super as workflows;
use super::{assert_wiring_mutation_rejected, support};

fn release_selector(workflow: &workflows::Node) -> &workflows::Node {
    workflows::steps(workflows::job(workflow, "metadata"))
        .iter()
        .find(|step| workflows::text(step, "id") == Some("runners"))
        .unwrap()
}

fn release_selector_forces_hosted(workflow: &workflows::Node) -> bool {
    workflows::input(release_selector(workflow), "force_hosted") == Some("true")
}

#[test]
fn release_manual_request_and_nested_sdk_keep_hosted_override() {
    let release = workflows::document("release.yml");
    let metadata = workflows::job(&release, "metadata");
    let selector = release_selector(&release);
    assert_eq!(
        workflows::text(selector, "uses"),
        Some("./.github/actions/select-ci-runners")
    );
    for (key, expected) in [
        ("event_name", "${{ github.event_name }}"),
        ("original_event_name", ""),
        ("ref", "${{ github.ref }}"),
        (
            "depot_main_enabled",
            "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
        ),
        ("manual_use_depot", "${{ inputs.use_depot == true }}"),
        ("force_hosted", "true"),
    ] {
        assert_eq!(workflows::input(selector, key), Some(expected), "{key}");
    }
    assert!(release_selector_forces_hosted(&release));
    assert_wiring_mutation_rejected(
        "release.yml",
        "          manual_use_depot: ${{ inputs.use_depot == true }}\n          force_hosted: true",
        "          manual_use_depot: ${{ inputs.use_depot == true }}\n          force_hosted: false",
        release_selector_forces_hosted,
    );
    let outputs = metadata.get("outputs").unwrap();
    assert_eq!(
        workflows::text(outputs, "source_sha"),
        Some("${{ steps.source.outputs.sha }}")
    );
    for name in ["runner_4", "runner_8", "runner_16"] {
        assert_eq!(
            workflows::text(outputs, name),
            Some(format!("${{{{ steps.runners.outputs.{name} }}}}").as_str())
        );
    }

    let sdk = workflows::job(&release, "build_native_sdk_runtime");
    assert_eq!(
        workflows::text(sdk, "uses"),
        Some("./.github/workflows/native-sdk-artifact.yml")
    );
    for (key, expected) in [
        ("source_sha", "${{ needs.metadata.outputs.source_sha }}"),
        ("original_event_name", "${{ github.event_name }}"),
        ("force_hosted", "true"),
        ("use_depot", "${{ inputs.use_depot == true }}"),
        (
            "artifact_name",
            "release-native-sdk-${{ matrix.artifact_suffix }}",
        ),
        (
            "static_abi_artifact_name",
            "ci-release-native-sdk-static-abi-${{ matrix.artifact_suffix }}",
        ),
    ] {
        assert_eq!(workflows::input(sdk, key), Some(expected), "{key}");
    }
    let nested = workflows::document("native-sdk-artifact.yml");
    let abi = workflows::job(&nested, "produce_linux_static_abi");
    for key in [
        "source_sha",
        "original_event_name",
        "force_hosted",
        "use_depot",
    ] {
        assert_eq!(
            workflows::input(abi, key),
            Some(format!("${{{{ inputs.{key} }}}}").as_str()),
            "{key}"
        );
    }

    let fixture = support::Fixture::new();
    let result = fixture.selector(&[
        ("INPUT_EVENT_NAME", "workflow_dispatch"),
        ("INPUT_ORIGINAL_EVENT_NAME", ""),
        ("INPUT_REF", "refs/heads/main"),
        ("INPUT_DEPOT_MAIN_ENABLED", "true"),
        ("INPUT_DEPOT_PR_ENABLED", "false"),
        ("INPUT_MANUAL_USE_DEPOT", "true"),
        ("INPUT_FORCE_HOSTED", "true"),
    ]);
    assert!(result.status.success(), "{result:?}");
    let selected = fixture.outputs();
    assert_eq!(selected["depot_enabled"], "false");
    for name in ["runner_4", "runner_8", "runner_16"] {
        assert_eq!(selected[name], "ubuntu-24.04", "{name}");
    }
    assert_eq!(selected["allow_depot_remote_cache"], "false");
    fixture.0.close().unwrap();
}
