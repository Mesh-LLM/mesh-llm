//! Drive the real planner's placement signal into the real selector action.
use super::support;
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};

fn plan(profile: &str, event: &str, base: &str, path: &str) -> Value {
    let request = json!({
        "profile": profile,
        "event_name": event,
        "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "base_sha": base,
        "changed_files": [path],
        "affected_crates": ["mesh-llm"],
        "workspace_packages": [{"name": "mesh-llm", "path": "mesh/crates/mesh-llm"}]
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(support::root())
        .args(["ci", "plan"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn pr_and_dispatch_decisions_follow_planned_work_and_checked_in_linux_callers() {
    let lane = super::document("ci-linux-lane.yml");
    let static_abi = super::job(&lane, "static_abi");
    let nested_sdk = super::job(&lane, "kotlin_sdk_input");
    for caller in [static_abi, nested_sdk] {
        assert_eq!(
            super::input(caller, "force_hosted"),
            Some("${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}")
        );
        assert_eq!(
            super::input(caller, "use_depot"),
            Some("${{ inputs.use_depot }}")
        );
        assert_eq!(
            super::input(caller, "source_sha"),
            Some("${{ inputs.source_sha }}")
        );
    }
    assert_eq!(
        super::input(static_abi, "artifact_name"),
        Some("ci-static-abi-${{ github.run_id }}")
    );
    assert_eq!(
        super::input(nested_sdk, "static_abi_artifact_name"),
        Some("ci-static-abi-${{ github.run_id }}")
    );
    for file in ["static-abi-artifact.yml", "native-sdk-artifact.yml"] {
        let workflow = super::document(file);
        let selectors = super::steps(super::job(&workflow, "runner_policy"))
            .iter()
            .filter(|step| super::text(step, "uses") == Some("./.github/actions/select-ci-runners"))
            .collect::<Vec<_>>();
        assert_eq!(selectors.len(), 1, "{file}");
        for (key, expected) in [
            ("force_hosted", "${{ inputs.force_hosted }}"),
            ("manual_use_depot", "${{ inputs.use_depot }}"),
            ("original_event_name", "${{ inputs.original_event_name }}"),
            (
                "depot_main_enabled",
                "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            ),
            (
                "depot_pr_enabled",
                "${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
            ),
        ] {
            assert_eq!(
                super::input(selectors[0], key),
                Some(expected),
                "{file}/{key}"
            );
        }
    }
    let sdk = super::document("native-sdk-artifact.yml");
    let nested_abi = super::job(&sdk, "produce_linux_static_abi");
    for key in [
        "force_hosted",
        "use_depot",
        "original_event_name",
        "source_sha",
    ] {
        assert_eq!(
            super::input(nested_abi, key),
            Some(format!("${{{{ inputs.{key} }}}}").as_str()),
            "{key}"
        );
    }

    let base = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let product = plan(
        "pr-ready",
        "pull_request",
        base,
        "mesh/crates/mesh-llm/src/main.rs",
    );
    assert_eq!(product["signals"]["force_hosted"], false);
    for required in [
        "static-abi",
        "rust-tests",
        "runtime-product",
        "product-smoke",
    ] {
        assert!(
            product["required_slices"]
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value == required),
            "{required}"
        );
    }
    assert!(
        product["matrices"]["runtime_products"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["id"] == "linux-cpu")
    );
    assert!(
        product["matrices"]["rust_tests"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["crates"]
                .as_array()
                .unwrap()
                .iter()
                .any(|name| name == "mesh-llm"))
    );
    let control = plan(
        "pr-ready",
        "pull_request",
        base,
        ".github/workflows/main_linux.yml",
    );
    assert_eq!(control["signals"]["force_hosted"], true);

    // Fixed dates exercise the expired exception boundary without renewing it.
    let cases = [
        (
            "same-repo active",
            "pull_request",
            "",
            "Mesh-LLM/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-13",
            "true",
            "false",
            false,
            true,
            true,
            false,
        ),
        (
            "expired",
            "pull_request",
            "",
            "Mesh-LLM/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-14",
            "true",
            "false",
            false,
            false,
            true,
            true,
        ),
        (
            "fork",
            "pull_request",
            "",
            "other/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-13",
            "true",
            "false",
            false,
            false,
            true,
            true,
        ),
        (
            "target event",
            "pull_request_target",
            "",
            "Mesh-LLM/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-13",
            "true",
            "false",
            false,
            false,
            true,
            true,
        ),
        (
            "PR dispatch",
            "workflow_dispatch",
            "pull_request",
            "Mesh-LLM/mesh-llm",
            "refs/heads/main",
            "2026-09-13",
            "true",
            "true",
            false,
            false,
            true,
            true,
        ),
        (
            "policy override",
            "workflow_dispatch",
            "workflow_dispatch",
            "Mesh-LLM/mesh-llm",
            "refs/heads/main",
            "2026-09-13",
            "true",
            "true",
            true,
            false,
            true,
            true,
        ),
    ];
    for (
        name,
        event,
        original,
        head_repo,
        reference,
        date,
        pr_gate,
        manual,
        policy_change,
        depot,
        native_cache,
        seed,
    ) in cases
    {
        let plan = if policy_change { &control } else { &product };
        let force = plan["signals"]["force_hosted"].as_bool().unwrap();
        let fixture = support::Fixture::new();
        let output = fixture.selector(&[
            ("INPUT_EVENT_NAME", event),
            ("INPUT_ORIGINAL_EVENT_NAME", original),
            ("INPUT_HEAD_REPOSITORY", head_repo),
            ("INPUT_REF", reference),
            ("FIXTURE_DATE", date),
            ("INPUT_DEPOT_PR_ENABLED", pr_gate),
            ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ("INPUT_MANUAL_USE_DEPOT", manual),
            ("INPUT_FORCE_HOSTED", if force { "true" } else { "false" }),
        ]);
        assert!(output.status.success(), "{name}: {output:?}");
        let selected = fixture.outputs();
        assert_eq!(selected["depot_enabled"], depot.to_string(), "{name}");
        assert_eq!(selected["allow_depot_remote_cache"], "false", "{name}");
        assert_eq!(
            selected["allow_native_github_cache"],
            native_cache.to_string(),
            "{name}"
        );
        assert_eq!(
            selected["allow_trusted_sccache_seed"],
            seed.to_string(),
            "{name}"
        );
        let cache = fixture.cache(&[
            ("GITHUB_EVENT_NAME", event),
            ("DISPATCH_ORIGINAL_EVENT_NAME", original),
            (
                "INPUT_ALLOW_DEPOT_REMOTE_CACHE",
                selected["allow_depot_remote_cache"].as_str().unwrap(),
            ),
            (
                "INPUT_ALLOW_NATIVE_GITHUB_CACHE",
                selected["allow_native_github_cache"].as_str().unwrap(),
            ),
        ]);
        assert_eq!(cache["failures"], json!([]), "{name}");
        let expected_chain = if name == "policy override" {
            "disk,gha"
        } else {
            "disk"
        };
        assert_eq!(
            cache["exports"]["SCCACHE_MULTILEVEL_CHAIN"], expected_chain,
            "{name}"
        );
        fixture.0.close().unwrap();
    }

    let fixture = support::Fixture::new();
    let malformed = fixture.selector(&[("INPUT_DEPOT_PR_ENABLED", "not-a-boolean")]);
    assert!(!malformed.status.success());
    assert!(!fixture.path().join("outputs").exists());
}

#[test]
fn exhaustive_coverage_and_provider_selection_remain_separate() {
    let base = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let cases = [
        (
            "main",
            "push",
            base,
            "README.md",
            "true",
            "false",
            false,
            true,
        ),
        (
            "main",
            "push",
            base,
            "README.md",
            "false",
            "false",
            false,
            false,
        ),
        (
            "main",
            "push",
            base,
            "README.md",
            "true",
            "false",
            true,
            false,
        ),
        (
            "main",
            "push",
            base,
            ".github/workflows/main_linux.yml",
            "true",
            "false",
            false,
            false,
        ),
        (
            "main",
            "push",
            "",
            "__force_all__",
            "true",
            "false",
            false,
            false,
        ),
        (
            "manual-full",
            "workflow_dispatch",
            base,
            "README.md",
            "false",
            "true",
            false,
            true,
        ),
        (
            "manual-full",
            "workflow_dispatch",
            base,
            "README.md",
            "false",
            "false",
            false,
            false,
        ),
    ];
    for (profile, event, base, path, main_gate, manual_gate, override_hosted, depot) in cases {
        let plan = plan(profile, event, base, path);
        assert_eq!(plan["signals"]["runner_contract_required"], true);
        let force = override_hosted || plan["signals"]["force_hosted"].as_bool().unwrap();
        let fixture = support::Fixture::new();
        let output = fixture.selector(&[
            ("INPUT_EVENT_NAME", event),
            ("INPUT_ORIGINAL_EVENT_NAME", event),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_DEPOT_MAIN_ENABLED", main_gate),
            ("INPUT_MANUAL_USE_DEPOT", manual_gate),
            ("INPUT_FORCE_HOSTED", if force { "true" } else { "false" }),
            ("FIXTURE_DATE", "2026-11-03"),
        ]);
        assert!(output.status.success(), "{profile} {path}: {output:?}");
        let selected = fixture.outputs();
        assert_eq!(
            selected["depot_enabled"],
            depot.to_string(),
            "{profile} {path}"
        );
        if depot {
            assert_eq!(selected["allow_depot_remote_cache"], "false");
            assert_eq!(selected["allow_native_github_cache"], "false");
        }
        fixture.0.close().unwrap();
    }
}
