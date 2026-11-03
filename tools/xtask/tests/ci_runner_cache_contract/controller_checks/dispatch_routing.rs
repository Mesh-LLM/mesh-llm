//! Compose the actual manual planner projection, dispatcher and selector.
//!
//! GitHub calls remain finite local seams; this does not execute a workflow or
//! authenticate a protected revision, runner, cache, or artifact.

use super::{LANES, common, run_with_plan};
use crate::support::{self, Fixture};
use crate::workflow_yaml::{self, Node};
use serde_json::{Value, json};
use std::{fs, process::Command, time::Duration};

#[path = "dispatch_routing/callers.rs"]
mod callers;
#[path = "dispatch_routing/resolvers.rs"]
mod resolvers;

const SOURCE: &str = "0123456789abcdef0123456789abcdef01234567";
const BASE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn plan_action(changed_file: &str) -> Value {
    let fixture = Fixture::new();
    let inherited_path = std::env::var_os("PATH").expect("fixture tool PATH");
    if !std::env::split_paths(&inherited_path)
        .map(|directory| directory.join("sha256sum"))
        .any(|candidate| candidate.is_file())
    {
        fixture.executable("sha256sum", "[[ $# == 0 ]] || exit 91; exec shasum -a 256");
    }
    // Supply a finite workspace metadata seam so the real planner never
    // launches Cargo or downloads dependencies during this local contract.
    fixture.executable(
        "planner",
        r#"jq -c '. + {workspace_packages: [{name: "mesh-llm", path: "mesh/crates/mesh-llm"}]}' | "$MESH_FIXTURE_XTASK" --repo-root "$GITHUB_WORKSPACE" "$@""#,
    );
    let changed = fixture.path().join("changed.txt");
    fs::write(&changed, format!("{changed_file}\n")).unwrap();
    // Isolate the action's fixed file transport; the planner and projection
    // script themselves remain the checked-in implementations.
    let script = support::step(&support::action("plan-ci"), "run")
        .replace("/tmp/changed_files.txt", "\"$MESH_FIXTURE_CHANGED_FILES\"");
    let path = std::env::join_paths(
        std::iter::once(fixture.path().join("bin")).chain(std::env::split_paths(&inherited_path)),
    )
    .unwrap();
    let mut command = Command::new("bash");
    command
        .env_clear()
        .current_dir(fixture.path())
        .env("PATH", path)
        .env("HOME", fixture.path())
        .env("TMPDIR", fixture.path())
        .env("RUNNER_TEMP", fixture.path())
        .env("GITHUB_WORKSPACE", support::root())
        .env("GITHUB_OUTPUT", fixture.path().join("outputs"))
        .env("GITHUB_STEP_SUMMARY", fixture.path().join("summary"))
        .env("MESH_CI_CONTROLLER_BIN", fixture.path().join("bin/planner"))
        .env("MESH_FIXTURE_XTASK", env!("CARGO_BIN_EXE_xtask"))
        .env("MESH_FIXTURE_CHANGED_FILES", &changed)
        .env("EVENT_NAME", "workflow_dispatch")
        .env("SOURCE_SHA", SOURCE)
        .env("BASE_SHA", BASE)
        .env("DRAFT", "false")
        .env("CHANGED_FILES_HINT", "")
        .env("AFFECTED_CRATES", r#"["mesh-llm"]"#)
        .env("UPLOAD_ARTIFACT", "false")
        .env("ARTIFACT_NAME", "")
        .args(["-c", &script]);
    let output = fixture.run_with_timeout(command, Duration::from_secs(30));
    assert!(output.status.success(), "{output:?}");
    fixture.outputs()
}

fn workflow(file: &str) -> Node {
    let source = fs::read_to_string(support::root().join(".github/workflows").join(file)).unwrap();
    workflow_yaml::parse_resolved_aliases(&source).unwrap()
}

fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}

fn accepts_manual_opt_in(lane: &str, job: &str) -> bool {
    matches!(
        (lane, job),
        ("quality", "quality")
            | (
                "linux",
                "hosts" | "native_runtimes" | "static_abi" | "kotlin_sdk_input"
            )
    )
}

fn check_selection(
    lane: &str,
    job: &str,
    dispatch: &Value,
    forced: bool,
    manual_allowed: bool,
    expected_depot: bool,
) {
    let inputs = &dispatch["inputs"];
    let fixture = Fixture::new();
    let output = fixture.selector(&[
        ("INPUT_EVENT_NAME", "workflow_dispatch"),
        (
            "INPUT_ORIGINAL_EVENT_NAME",
            inputs["original_event_name"].as_str().unwrap(),
        ),
        ("INPUT_REF", "refs/heads/main"),
        ("INPUT_HEAD_REPOSITORY", ""),
        ("INPUT_DEPOT_MAIN_ENABLED", "false"),
        ("INPUT_DEPOT_PR_ENABLED", "false"),
        (
            "INPUT_MANUAL_USE_DEPOT",
            if manual_allowed {
                inputs["use_depot"].as_str().unwrap()
            } else {
                "false"
            },
        ),
        ("INPUT_FORCE_HOSTED", if forced { "true" } else { "false" }),
        ("FIXTURE_DATE", "2026-11-03"),
    ]);
    assert!(output.status.success(), "{output:?}");
    let selected = fixture.outputs();
    assert_eq!(selected["depot_enabled"], expected_depot.to_string());
    let (runner, depot_label, hosted_label) = match (lane, job) {
        (_, "ui_artifact") | ("website", _) => ("runner_4", "depot-ubuntu-24.04-4", "ubuntu-24.04"),
        ("quality", _) | ("linux", "hosts" | "static_abi" | "kotlin_sdk_input") => {
            ("runner_8", "depot-ubuntu-24.04-8", "ubuntu-24.04")
        }
        ("linux", "native_runtimes") => ("runner_16", "depot-ubuntu-24.04-16", "ubuntu-24.04"),
        ("linux", "runtime_product" | "rust_tests") => {
            ("runner", "depot-ubuntu-24.04", "ubuntu-24.04")
        }
        ("macos", _) => ("runner_macos", "depot-macos-15", "macos-15"),
        ("windows", _) => ("runner_windows", "depot-windows-2022", "windows-2022"),
        _ => panic!("unexpected composed lane"),
    };
    assert_eq!(
        selected[runner],
        if expected_depot {
            depot_label
        } else {
            hosted_label
        },
        "{lane}"
    );
    assert_eq!(selected["allow_depot_remote_cache"], "false");
    assert_eq!(
        selected["allow_native_github_cache"],
        (!expected_depot).to_string()
    );
    match (lane, job) {
        ("linux", "static_abi") => {
            resolvers::check_native_resolver("static-abi-artifact.yml", &selected)
        }
        ("linux", "kotlin_sdk_input") => {
            resolvers::check_native_resolver("native-sdk-artifact.yml", &selected)
        }
        ("macos" | "windows", "platform_checks") => resolvers::check_platform_resolver(&selected),
        _ => {}
    }
}

#[test]
fn manual_dispatch_carries_real_plan_to_checked_in_runner_selectors() {
    let targets = [
        ("quality", "quality", "ci-quality-slice.yml"),
        ("website", "web", "ci-web-slice.yml"),
        ("linux", "ui_artifact", "ci-ui-artifact-slice.yml"),
        ("linux", "static_abi", "static-abi-artifact.yml"),
        ("linux", "rust_tests", "ci-rust-tests-slice.yml"),
        ("linux", "hosts", "ci-linux-host-slice.yml"),
        ("linux", "native_runtimes", "ci-linux-runtime-slice.yml"),
        ("linux", "runtime_product", "ci-linux-product-slice.yml"),
        ("linux", "kotlin_sdk_input", "native-sdk-artifact.yml"),
        ("macos", "ui_artifact", "ci-ui-artifact-slice.yml"),
        ("macos", "hosts", "ci-macos-host-slice.yml"),
        ("macos", "native_runtimes", "ci-macos-runtime-slice.yml"),
        ("macos", "runtime_product", "ci-macos-product-slice.yml"),
        ("macos", "platform_checks", "ci-platform-checks-slice.yml"),
        ("macos", "swift_sdk_input", "swift-sdk-artifact.yml"),
        ("windows", "ui_artifact", "ci-ui-artifact-slice.yml"),
        ("windows", "hosts", "ci-windows-host-slice.yml"),
        ("windows", "native_runtimes", "ci-windows-runtime-slice.yml"),
        ("windows", "runtime_product", "ci-windows-product-slice.yml"),
        ("windows", "platform_checks", "ci-platform-checks-slice.yml"),
    ];
    for (lane, job, slice) in targets {
        callers::check_forwarding(lane, job, slice);
    }
    for (changed, forced) in [
        ("README.md", false),
        (".github/workflows/main_linux.yml", true),
    ] {
        let action = plan_action(changed);
        let plan: Value = serde_json::from_str(action["plan_json"].as_str().unwrap()).unwrap();
        assert_eq!(plan["source_sha"], SOURCE);
        assert_eq!(plan["signals"]["force_hosted"], forced);
        let required = std::array::from_fn(|index| {
            let key = format!("{}_lane_plan", LANES[index]);
            let projection: Value = serde_json::from_str(action[key].as_str().unwrap()).unwrap();
            projection["required"].as_bool().unwrap()
        });
        assert!(
            required.iter().all(|selected| *selected),
            "all five lanes are selected"
        );
        for manual in [false, true] {
            let output = run_with_plan(required, "", "", json!(manual), Some(&action));
            common(&output, required, if manual { "true" } else { "false" });
            assert_eq!(output["canonical"], action["plan_json"]);
            assert_eq!(output["digest"], action["plan_digest"]);
            for lane in LANES {
                let dispatch = output["dispatches"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|dispatch| dispatch["workflow_id"] == format!("ci-{lane}-lane.yml"))
                    .unwrap();
                let projection: Value =
                    serde_json::from_str(dispatch["inputs"]["lane_plan_json"].as_str().unwrap())
                        .unwrap();
                assert_eq!(projection["signals"], plan["signals"]);
                assert_eq!(projection["required_slices"], plan["required_slices"]);
                if lane == "quality" {
                    assert_eq!(projection["matrices"]["clippy"], plan["matrices"]["clippy"]);
                } else if lane == "website" {
                    assert!(
                        projection["required_slices"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|slice| slice == "web")
                    );
                } else {
                    let hosts = plan["matrices"]["hosts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|row| row["platform"] == lane)
                        .cloned()
                        .collect::<Vec<_>>();
                    assert!(!hosts.is_empty());
                    assert_eq!(projection["matrices"]["hosts"], json!(hosts));
                    let runtimes = plan["matrices"]["runtime_products"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|row| row["platform"] == lane)
                        .cloned()
                        .collect::<Vec<_>>();
                    assert!(!runtimes.is_empty());
                    assert_eq!(projection["matrices"]["runtime_products"], json!(runtimes));
                    if lane == "linux" {
                        assert_eq!(
                            projection["matrices"]["rust_tests"],
                            plan["matrices"]["rust_tests"]
                        );
                        assert!(
                            !projection["matrices"]["rust_tests"]
                                .as_array()
                                .unwrap()
                                .is_empty()
                        );
                        assert!(
                            projection["required_slices"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .any(|slice| slice == "static-abi")
                        );
                    } else {
                        let checks = plan["matrices"]["platform_checks"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|row| row["platform"] == lane)
                            .cloned()
                            .collect::<Vec<_>>();
                        assert!(!checks.is_empty());
                        assert_eq!(projection["matrices"]["platform_checks"], json!(checks));
                    }
                    if lane != "windows" {
                        let sdks = plan["matrices"]["sdk"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|row| row["platform"] == lane)
                            .cloned()
                            .collect::<Vec<_>>();
                        assert!(sdks.iter().any(
                            |row| row["id"] == if lane == "linux" { "kotlin" } else { "swift" }
                        ));
                        assert_eq!(projection["matrices"]["sdk"], json!(sdks));
                    }
                }
                let forwarded_force =
                    projection["signals"]["force_hosted"].as_bool() != Some(false);
                assert_eq!(forwarded_force, forced);
                for (_, job, _) in targets
                    .into_iter()
                    .filter(|(target_lane, _, _)| *target_lane == lane)
                {
                    let accepts = accepts_manual_opt_in(lane, job);
                    check_selection(
                        lane,
                        job,
                        dispatch,
                        forwarded_force,
                        accepts,
                        accepts && manual && !forced,
                    );
                    if (lane, job) == ("linux", "native_runtimes") {
                        check_selection(lane, job, dispatch, true, false, false);
                    }
                    if lane == "quality" {
                        check_selection(lane, job, dispatch, forwarded_force, false, false);
                    }
                }
            }
        }
    }
}
