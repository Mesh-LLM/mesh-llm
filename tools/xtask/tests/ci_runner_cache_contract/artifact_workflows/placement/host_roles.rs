//! Planner and selector projections for host and Linux runtime roles.

use super::super as workflows;

use super::{plan, support};

#[test]
fn planned_host_roles_follow_checked_in_callers_and_selector_labels() {
    let planned = plan(
        "main",
        "push",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "README.md",
    );
    assert_eq!(planned["signals"]["force_hosted"], false);
    let control = plan(
        "main",
        "push",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ".github/workflows/main_linux.yml",
    );
    assert_eq!(control["signals"]["force_hosted"], true);
    let cases = [
        (
            "linux",
            "linux-amd64-host",
            "linux-build-8",
            "linux_host",
            "runner_8",
            "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            "depot-ubuntu-24.04-8",
            "ubuntu-24.04",
        ),
        (
            "macos",
            "macos-arm64-host",
            "macos-build",
            "macos_host",
            "runner_macos",
            "false",
            "macos-15",
            "macos-15",
        ),
        (
            "windows",
            "windows-amd64-host",
            "windows-build",
            "windows_host",
            "runner_windows",
            "false",
            "windows-2022",
            "windows-2022",
        ),
    ];
    for (platform, row_id, role, job_id, output, main_gate, main_label, hosted_label) in cases {
        let row = planned["matrices"]["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == row_id)
            .unwrap_or_else(|| panic!("missing planned host {row_id}"));
        assert_eq!(row["runner_role"], role);

        let lane = workflows::document(&format!("ci-{platform}-lane.yml"));
        let caller = workflows::job(&lane, "hosts");
        assert_eq!(
            workflows::input(caller, "force_hosted"),
            Some("${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}")
        );
        assert_eq!(
            workflows::input(caller, "hosts_matrix"),
            Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.hosts) }}")
        );
        assert_eq!(
            workflows::input(caller, "source_sha"),
            Some("${{ inputs.source_sha }}")
        );
        assert_eq!(
            workflows::input(caller, "original_event_name"),
            Some("${{ inputs.original_event_name }}")
        );
        let slice = workflows::document(&format!("ci-{platform}-host-slice.yml"));
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
            workflows::input(selector, "force_hosted"),
            Some("${{ inputs.force_hosted }}")
        );
        assert_eq!(
            workflows::input(selector, "depot_main_enabled"),
            Some(main_gate)
        );
        let main_gate_value = match main_gate {
            "false" => "false",
            "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}" => "true",
            _ => panic!("unrecognized main gate for {platform}"),
        };
        assert_eq!(
            workflows::text(policy.get("outputs").unwrap(), output),
            Some(format!("${{{{ steps.policy.outputs.{output} }}}}").as_str())
        );
        assert_eq!(
            workflows::text(workflows::job(&slice, job_id), "runs-on"),
            Some(format!("${{{{ needs.runner_policy.outputs.{output} }}}}").as_str())
        );

        for (force, expected) in [
            (
                planned["signals"]["force_hosted"].as_bool().unwrap(),
                main_label,
            ),
            (
                control["signals"]["force_hosted"].as_bool().unwrap(),
                hosted_label,
            ),
        ] {
            let fixture = support::Fixture::new();
            let result = fixture.selector(&[
                ("INPUT_EVENT_NAME", "push"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_DEPOT_MAIN_ENABLED", main_gate_value),
                ("INPUT_FORCE_HOSTED", if force { "true" } else { "false" }),
            ]);
            assert!(result.status.success(), "{platform}: {result:?}");
            assert_eq!(fixture.outputs()[output], expected, "{platform}");
            fixture.0.close().unwrap();
        }
    }
}

#[test]
fn planned_linux_cpu_runtime_keeps_the_hosted_exception() {
    let planned = plan(
        "main",
        "push",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "README.md",
    );
    assert_eq!(planned["signals"]["force_hosted"], false);
    let rows = planned["matrices"]["runtime_products"].as_array().unwrap();
    assert!(
        rows.iter()
            .any(|row| row["id"] == "linux-cpu" && row["runner_role"] == "linux-build-8")
    );
    assert!(
        rows.iter()
            .any(|row| row["id"] == "linux-cuda" && row["runner_role"] == "linux-native-16")
    );

    let lane = workflows::document("ci-linux-lane.yml");
    let caller = workflows::job(&lane, "native_runtimes");
    assert_eq!(
        workflows::input(caller, "force_hosted"),
        Some("${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}")
    );
    assert_eq!(
        workflows::input(caller, "use_depot"),
        Some("${{ inputs.use_depot }}")
    );
    assert_eq!(
        workflows::input(caller, "runtime_matrix"),
        Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.runtime_products) }}")
    );
    let slice = workflows::document("ci-linux-runtime-slice.yml");
    let policy = workflows::job(&slice, "runner_policy");
    let normal = workflows::steps(policy)
        .iter()
        .find(|step| workflows::text(step, "id") == Some("policy"))
        .unwrap();
    let cpu = workflows::steps(policy)
        .iter()
        .find(|step| workflows::text(step, "id") == Some("cpu_policy"))
        .unwrap();
    for selector in [normal, cpu] {
        assert_eq!(
            workflows::text(selector, "uses"),
            Some("./.github/actions/select-ci-runners")
        );
        assert_eq!(
            workflows::input(selector, "depot_main_enabled"),
            Some("${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}")
        );
    }
    assert_eq!(
        workflows::input(normal, "force_hosted"),
        Some("${{ inputs.force_hosted }}")
    );
    assert_eq!(workflows::input(cpu, "force_hosted"), Some("true"));
    let outputs = policy.get("outputs").unwrap();
    assert_eq!(
        workflows::text(outputs, "runner_16"),
        Some("${{ steps.policy.outputs.runner_16 }}")
    );
    assert_eq!(
        workflows::text(outputs, "runner_cpu"),
        Some("${{ steps.cpu_policy.outputs.runner_16 }}")
    );
    assert_eq!(
        workflows::text(outputs, "allow_native_github_cache_cpu"),
        Some("${{ steps.cpu_policy.outputs.allow_native_github_cache }}")
    );
    assert_eq!(
        workflows::text(workflows::job(&slice, "linux_runtime"), "runs-on"),
        Some(
            "${{ matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.runner_cpu || needs.runner_policy.outputs.runner_16 }}"
        )
    );

    for (force, expected, native_cache) in [
        ("false", "depot-ubuntu-24.04-16", "false"),
        ("true", "ubuntu-24.04", "true"),
    ] {
        let fixture = support::Fixture::new();
        let result = fixture.selector(&[
            ("INPUT_EVENT_NAME", "push"),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ("INPUT_FORCE_HOSTED", force),
        ]);
        assert!(result.status.success(), "{result:?}");
        let selected = fixture.outputs();
        assert_eq!(selected["runner_16"], expected);
        assert_eq!(selected["allow_native_github_cache"], native_cache);
        fixture.0.close().unwrap();
    }
}
