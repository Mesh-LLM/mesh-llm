//! Main macOS and Windows runtime selection and artifact identities.

use super::super as workflows;
use super::{plan, support};

#[test]
fn main_platform_runtimes_keep_hosted_gates_and_artifact_names() {
    let planned = plan(
        "main",
        "push",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "README.md",
    );
    assert_eq!(planned["signals"]["force_hosted"], false);
    for (platform, row_id, role, job_id, output, hosted) in [
        (
            "macos",
            "macos-metal",
            "macos-build",
            "macos_runtime",
            "runner_macos",
            "macos-15",
        ),
        (
            "windows",
            "windows-cpu",
            "windows-build",
            "windows_runtime",
            "runner_windows",
            "windows-2022",
        ),
    ] {
        let row = planned["matrices"]["runtime_products"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == row_id)
            .unwrap_or_else(|| panic!("missing planned runtime {row_id}"));
        assert_eq!(row["runner_role"], role);
        let lane = workflows::document(&format!("ci-{platform}-lane.yml"));
        let caller = workflows::job(&lane, "native_runtimes");
        assert_eq!(
            workflows::text(caller, "uses"),
            Some(format!("./.github/workflows/ci-{platform}-runtime-slice.yml").as_str())
        );
        for (key, expected) in [
            (
                "force_hosted",
                "${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}",
            ),
            (
                "runtime_matrix",
                "${{ toJson(fromJson(inputs.lane_plan_json).matrices.runtime_products) }}",
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
        let slice = workflows::document(&format!("ci-{platform}-runtime-slice.yml"));
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
        let runtime = workflows::job(&slice, job_id);
        assert_eq!(
            workflows::text(runtime, "runs-on"),
            Some(format!("${{{{ needs.runner_policy.outputs.{output} }}}}").as_str())
        );
        let artifact = format!(
            "ci-runtime-{platform}-${{{{ matrix.runtime.architecture }}}}-${{{{ matrix.runtime.backend }}}}"
        );
        let uploads = workflows::steps(runtime)
            .iter()
            .filter(|step| {
                workflows::text(step, "uses")
                    .is_some_and(|uses| uses.starts_with("actions/upload-artifact@"))
                    && workflows::input(step, "name") == Some(artifact.as_str())
            })
            .count();
        assert_eq!(uploads, 1, "{platform}");

        let fixture = support::Fixture::new();
        let result = fixture.selector(&[
            ("INPUT_EVENT_NAME", "push"),
            ("INPUT_ORIGINAL_EVENT_NAME", "push"),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_DEPOT_MAIN_ENABLED", "false"),
            ("INPUT_FORCE_HOSTED", "false"),
        ]);
        assert!(result.status.success(), "{platform}: {result:?}");
        assert_eq!(fixture.outputs()[output], hosted, "{platform}");
        fixture.0.close().unwrap();
    }
}
