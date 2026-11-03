//! Compose the actual manual planner projection, dispatcher and selector.
//!
//! GitHub calls remain finite local seams; this does not execute a workflow or
//! authenticate a protected revision, runner, cache, or artifact.

use super::{LANES, common, run_with_plan};
use crate::support::{self, Fixture};
use crate::workflow_yaml::{self, Node};
use serde_json::{Value, json};
use std::{fs, process::Command, time::Duration};

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

fn check_forwarding(lane: &str, job: &str, slice: &str) {
    let caller = workflow(&format!("ci-{lane}-lane.yml"));
    let job = caller.get("jobs").unwrap().get(job).unwrap();
    assert_eq!(
        text(job, "uses"),
        Some(format!("./.github/workflows/{slice}").as_str())
    );
    let inputs = job.get("with").unwrap();
    for (key, expected) in [
        ("source_sha", "${{ inputs.source_sha }}"),
        ("original_event_name", "${{ inputs.original_event_name }}"),
        ("use_depot", "${{ inputs.use_depot }}"),
        (
            "force_hosted",
            "${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}",
        ),
    ] {
        assert_eq!(text(inputs, key), Some(expected), "{lane}/{key}");
    }
    match lane {
        "quality" => assert_eq!(
            text(inputs, "clippy_matrix"),
            Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.clippy) }}")
        ),
        "linux" => {
            assert_eq!(
                text(inputs, "hosts_matrix"),
                Some("${{ toJson(fromJson(inputs.lane_plan_json).matrices.hosts) }}")
            );
            assert_eq!(
                text(inputs, "ui_artifact_name"),
                Some("ci-ui-dist-linux-${{ github.run_id }}")
            );
        }
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
        ("manual_use_depot", "${{ inputs.use_depot }}"),
        ("force_hosted", "${{ inputs.force_hosted }}"),
    ] {
        assert_eq!(text(inputs, key), Some(expected), "{slice}/{key}");
    }
}

fn check_selection(dispatch: &Value, forced: bool, expected_depot: bool) {
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
            inputs["use_depot"].as_str().unwrap(),
        ),
        ("INPUT_FORCE_HOSTED", if forced { "true" } else { "false" }),
        ("FIXTURE_DATE", "2026-11-03"),
    ]);
    assert!(output.status.success(), "{output:?}");
    let selected = fixture.outputs();
    assert_eq!(selected["depot_enabled"], expected_depot.to_string());
    assert_eq!(
        selected["runner_8"],
        if expected_depot {
            "depot-ubuntu-24.04-8"
        } else {
            "ubuntu-24.04"
        }
    );
    assert_eq!(selected["allow_depot_remote_cache"], "false");
    assert_eq!(
        selected["allow_native_github_cache"],
        (!expected_depot).to_string()
    );
}

#[test]
fn manual_dispatch_carries_real_plan_to_checked_in_runner_selectors() {
    for (lane, job, slice) in [
        ("quality", "quality", "ci-quality-slice.yml"),
        ("linux", "hosts", "ci-linux-host-slice.yml"),
    ] {
        check_forwarding(lane, job, slice);
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
        assert!(required[0] && required[2], "quality and Linux are selected");
        for manual in [false, true] {
            let output = run_with_plan(required, "", "", json!(manual), Some(&action));
            common(&output, required, if manual { "true" } else { "false" });
            assert_eq!(output["canonical"], action["plan_json"]);
            assert_eq!(output["digest"], action["plan_digest"]);
            for lane in ["quality", "linux"] {
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
                } else {
                    let hosts = plan["matrices"]["hosts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|row| row["platform"] == "linux")
                        .cloned()
                        .collect::<Vec<_>>();
                    assert!(!hosts.is_empty());
                    assert_eq!(projection["matrices"]["hosts"], json!(hosts));
                }
                let forwarded_force =
                    projection["signals"]["force_hosted"].as_bool() != Some(false);
                assert_eq!(forwarded_force, forced);
                check_selection(dispatch, forwarded_force, manual && !forced);
            }
        }
    }
}
