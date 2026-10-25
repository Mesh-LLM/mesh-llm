//! Actual reporter JavaScript with finite inert API boundary, no GitHub access.
use super::{Node, support, text};
use serde_json::{Value, json};
use std::{fs, process::Command};
fn report(input: Value) -> Value {
    let action = support::action("report-ci-lane");
    let Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!("steps")
    };
    let selected = steps
        .iter()
        .filter(|s| text(s, "name") == Some("Complete correlated PR/main checks"))
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 1);
    let step = selected[0];
    assert_eq!(text(step, "if"), Some("${{ inputs.lane_check_id != '' }}"));
    for (key, value) in [
        ("SOURCE_SHA", "${{ inputs.source_sha }}"),
        ("CORRELATION_ID", "${{ inputs.correlation_id }}"),
        ("OVERALL_CHECK_ID", "${{ inputs.overall_check_id }}"),
    ] {
        assert_eq!(text(step.get("env").unwrap(), key), Some(value));
    }
    let fixture = support::Fixture::new();
    let actual = fixture.path().join("actual.js");
    let config = fixture.path().join("input.json");
    fs::write(&actual, super::input(step, "script").unwrap()).unwrap();
    fs::write(&config, serde_json::to_vec(&input).unwrap()).unwrap();
    let mut command = Command::new("node");
    command
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap())
        .env("HOME", fixture.path())
        .arg(support::root().join(
            "tools/xtask/tests/ci_runner_cache_contract/artifact_workflows/reporter_fixture.js",
        ))
        .arg(actual)
        .arg(config);
    let output = fixture.run(command);
    assert!(output.status.success(), "{output:?}");
    let value = serde_json::from_slice(&output.stdout).unwrap();
    fixture.0.close().unwrap();
    value
}
#[test]
fn graph_reporter_refuses_malformed_or_mismatched_lane_and_overall_before_update() {
    for input in [
        json!({"source":"bad"}),
        json!({"correlation":""}),
        json!({"expected":[3]}),
    ] {
        let output = report(input);
        assert!(output["error"].is_string());
        assert_eq!(output["updates"], json!([]));
        assert_eq!(output["pagination"], json!([]));
    }
    for id in [7, 9] {
        for key in ["name", "external_id", "head_sha"] {
            let output = report(json!({"mutation":{"id":id,"key":key,"value":"foreign"}}));
            assert!(
                output["error"]
                    .as_str()
                    .unwrap()
                    .contains("identity does not match")
            );
            assert_eq!(output["updates"], json!([]));
        }
    }
}
#[test]
fn graph_reporter_converges_only_correlated_completed_lanes_and_preserves_failure() {
    for peer in ["success", "failure"] {
        let output = report(json!({"peer_result":peer}));
        assert!(output["error"].is_null());
        assert_ne!(output["source"], output["controller_sha"]);
        let updates = output["updates"].as_array().unwrap();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0]["check_run_id"], 7);
        assert_eq!(updates[1]["check_run_id"], 9);
        assert_eq!(updates[1]["conclusion"], peer);
        let pages = output["pagination"].as_array().unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0]["ref"], output["source"]);
        assert_eq!(pages[0]["per_page"], 100);
        assert_eq!(pages[0]["filter"], "latest");
    }
}
#[test]
fn graph_reporter_unconverged_overall_stays_open_and_optional_overall_is_omitted() {
    for input in [json!({"incomplete":true}), json!({"missing":true})] {
        let output = report(input);
        assert!(output["error"].is_null());
        assert_eq!(output["updates"].as_array().unwrap().len(), 1);
        assert_eq!(output["pagination"].as_array().unwrap().len(), 6);
    }
    for input in [json!({"overall":false}), json!({"expected":[]})] {
        let output = report(input.clone());
        assert!(output["error"].is_null());
        assert_eq!(output["updates"].as_array().unwrap().len(), 1);
        assert_eq!(output["pagination"], json!([]));
        if input["overall"] == false {
            assert_eq!(output["gets"].as_array().unwrap().len(), 1);
        }
    }
}
