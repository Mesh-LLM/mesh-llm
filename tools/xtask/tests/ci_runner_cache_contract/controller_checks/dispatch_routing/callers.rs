//! Checked-in lane-to-selector bindings for the manual dispatch contract.
use super::{accepts_manual_opt_in, text, workflow};
use crate::support;
use crate::workflow_yaml::Node;

pub(super) fn check_forwarding(lane: &str, job: &str, slice: &str) {
    let caller = workflow(&format!("ci-{lane}-lane.yml"));
    let caller_job = caller.get("jobs").unwrap().get(job).unwrap();
    assert_eq!(
        text(caller_job, "uses"),
        Some(format!("./.github/workflows/{slice}").as_str())
    );
    let inputs = caller_job.get("with").unwrap();
    for (key, expected) in [
        ("source_sha", "${{ inputs.source_sha }}"),
        ("original_event_name", "${{ inputs.original_event_name }}"),
        (
            "force_hosted",
            "${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}",
        ),
    ] {
        assert_eq!(text(inputs, key), Some(expected), "{lane}/{key}");
    }
    let accepts_manual_opt_in = accepts_manual_opt_in(lane, job);
    assert_eq!(
        text(inputs, "use_depot"),
        accepts_manual_opt_in.then_some("${{ inputs.use_depot }}"),
        "{lane}/use_depot"
    );
    match (lane, job) {
        ("quality", "quality") => assert_eq!(
            text(inputs, "clippy_matrix"),
            Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.clippy) }}")
        ),
        ("website", "web") => {
            for key in ["ui_changed", "website_changed"] {
                assert_eq!(
                    text(inputs, key),
                    Some(format!("${{{{ fromJson(inputs.lane_plan_json).signals.{key} || contains(fromJson(inputs.lane_plan_json).domains, 'ci-control') }}}}").as_str()),
                    "{lane}/{key}"
                );
            }
        }
        ("linux" | "macos" | "windows", "hosts") => {
            assert_eq!(
                text(inputs, "hosts_matrix"),
                Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.hosts) }}")
            );
            assert_eq!(
                text(inputs, "ui_artifact_name"),
                Some(format!("ci-ui-dist-{lane}-${{{{ github.run_id }}}}").as_str())
            );
        }
        ("linux" | "macos" | "windows", "ui_artifact") => assert_eq!(
            text(inputs, "artifact_name"),
            Some(format!("ci-ui-dist-{lane}-${{{{ github.run_id }}}}").as_str())
        ),
        ("linux", "static_abi" | "kotlin_sdk_input") => {
            for (key, expected) in [
                ("backend", "cpu"),
                ("target", "x86_64-unknown-linux-gnu"),
                ("runner_size", "8"),
            ] {
                assert_eq!(text(inputs, key), Some(expected), "{job}/{key}");
            }
            let artifact_key = if job == "static_abi" {
                "artifact_name"
            } else {
                "static_abi_artifact_name"
            };
            assert_eq!(
                text(inputs, artifact_key),
                Some("ci-static-abi-${{ github.run_id }}")
            );
        }
        ("linux", "rust_tests") => {
            assert_eq!(
                text(inputs, "rust_tests_matrix"),
                Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.rust_tests) }}")
            );
            assert_eq!(
                text(inputs, "static_abi_artifact_name"),
                Some("ci-static-abi-${{ github.run_id }}")
            );
            assert_eq!(
                text(inputs, "static_abi_toolchain_epoch"),
                Some("${{ needs.static_abi.outputs.toolchain_epoch }}")
            );
        }
        ("macos" | "windows", "platform_checks") => assert_eq!(
            text(inputs, "platform_checks_matrix"),
            Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.platform_checks) }}")
        ),
        ("macos", "swift_sdk_input") => {
            assert_eq!(text(inputs, "artifact_name"), Some("ci-sdk-swift"));
            assert_eq!(
                text(inputs, "mode"),
                Some(
                    "${{ (fromJson(inputs.lane_plan_json).profile == 'main' || fromJson(inputs.lane_plan_json).profile == 'manual-full') && 'full' || 'host-only' }}"
                )
            );
        }
        ("linux" | "macos" | "windows", "native_runtimes" | "runtime_product") => assert_eq!(
            text(inputs, "runtime_matrix"),
            Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.runtime_products) }}")
        ),
        _ => panic!("unexpected composed lane"),
    }
    let called = workflow(slice);
    let steps = called
        .get("jobs")
        .unwrap()
        .get("runner_policy")
        .unwrap()
        .get("steps")
        .unwrap();
    let Node::Seq(steps) = steps else {
        panic!("selector steps")
    };
    let selector = steps
        .iter()
        .find(|step| text(step, "id") == Some("policy"))
        .unwrap();
    assert_eq!(
        text(selector, "uses"),
        Some("./.github/actions/select-ci-runners")
    );
    let inputs = selector.get("with").unwrap();
    for (key, expected) in [
        ("event_name", "${{ github.event_name }}"),
        ("original_event_name", "${{ inputs.original_event_name }}"),
        ("force_hosted", "${{ inputs.force_hosted }}"),
    ] {
        assert_eq!(text(inputs, key), Some(expected), "{slice}/{key}");
    }
    assert_eq!(
        text(inputs, "manual_use_depot"),
        accepts_manual_opt_in.then_some("${{ inputs.use_depot }}"),
        "{slice}/manual_use_depot"
    );
    if !accepts_manual_opt_in {
        assert_eq!(text(inputs, "depot_main_enabled"), Some("false"));
        let action = support::action("select-ci-runners");
        let manual = action
            .get("inputs")
            .unwrap()
            .get("manual_use_depot")
            .unwrap();
        assert_eq!(text(manual, "default"), Some("false"));
    }
    if (lane, job) == ("linux", "native_runtimes") {
        let cpu = steps
            .iter()
            .find(|step| text(step, "id") == Some("cpu_policy"))
            .unwrap();
        assert_eq!(
            text(cpu, "uses"),
            Some("./.github/actions/select-ci-runners")
        );
        let cpu_inputs = cpu.get("with").unwrap();
        assert_eq!(text(cpu_inputs, "force_hosted"), Some("true"));
        assert_eq!(text(cpu_inputs, "manual_use_depot"), None);
        let policy = called.get("jobs").unwrap().get("runner_policy").unwrap();
        let outputs = policy.get("outputs").unwrap();
        assert_eq!(
            text(outputs, "runner_cpu"),
            Some("${{ steps.cpu_policy.outputs.runner_16 }}")
        );
        assert_eq!(
            text(outputs, "allow_native_github_cache_cpu"),
            Some("${{ steps.cpu_policy.outputs.allow_native_github_cache }}")
        );
        let runtime = called.get("jobs").unwrap().get("linux_runtime").unwrap();
        assert_eq!(
            text(runtime, "runs-on"),
            Some(
                "${{ matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.runner_cpu || needs.runner_policy.outputs.runner_16 }}"
            )
        );
    }
    if (lane, job) == ("linux", "kotlin_sdk_input") {
        // This lane consumes a separately produced ABI. The SDK's optional
        // nested producer remains off by default; validate its bindings for
        // callers that explicitly request it without claiming it ran here.
        assert_eq!(
            text(caller_job.get("with").unwrap(), "produce_static_abi"),
            None
        );
        let default = called
            .get("on")
            .unwrap()
            .get("workflow_call")
            .unwrap()
            .get("inputs")
            .unwrap()
            .get("produce_static_abi")
            .unwrap();
        assert_eq!(text(default, "default"), Some("false"));
        let nested = called
            .get("jobs")
            .unwrap()
            .get("produce_linux_static_abi")
            .unwrap();
        assert_eq!(
            text(nested, "uses"),
            Some("./.github/workflows/static-abi-artifact.yml")
        );
        let nested_inputs = nested.get("with").unwrap();
        for key in [
            "source_sha",
            "original_event_name",
            "force_hosted",
            "use_depot",
            "target",
            "runner_size",
        ] {
            assert_eq!(
                text(nested_inputs, key),
                Some(format!("${{{{ inputs.{key} }}}}").as_str()),
                "nested/{key}"
            );
        }
        assert_eq!(
            text(nested_inputs, "artifact_name"),
            Some("${{ inputs.static_abi_artifact_name }}")
        );
    }
    if lane == "quality" {
        let sentinel = steps
            .iter()
            .find(|step| text(step, "id") == Some("sentinel_policy"))
            .unwrap();
        assert_eq!(
            text(sentinel, "uses"),
            Some("./.github/actions/select-ci-runners")
        );
        let sentinel_inputs = sentinel.get("with").unwrap();
        for (key, value) in [
            ("event_name", "${{ github.event_name }}"),
            ("original_event_name", "${{ inputs.original_event_name }}"),
            ("force_hosted", "${{ inputs.force_hosted }}"),
        ] {
            assert_eq!(text(sentinel_inputs, key), Some(value), "{key}");
        }
        for key in ["depot_main_enabled", "depot_pr_enabled", "manual_use_depot"] {
            assert_eq!(text(sentinel_inputs, key), Some("false"), "{key}");
        }
        assert_eq!(
            text(sentinel_inputs, "pr_canary_ref"),
            Some("${{ vars.DEPOT_PR_SENTINEL_REF }}")
        );
    }
    if lane == "website" {
        for (name, input) in [
            ("ui_quality", "ui_changed"),
            ("ui_e2e", "ui_changed"),
            ("website", "website_changed"),
        ] {
            let web_job = called.get("jobs").unwrap().get(name).unwrap();
            assert_eq!(
                text(web_job, "if"),
                Some(format!("${{{{ inputs.{input} }}}}").as_str())
            );
            assert_eq!(
                text(web_job, "runs-on"),
                Some("${{ needs.runner_policy.outputs.runner_4 }}")
            );
        }
    }
}
