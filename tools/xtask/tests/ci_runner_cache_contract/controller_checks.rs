//! Execute the actual protected controller scalar against finite GitHub seams.
use super::{
    support::{Fixture, root},
    workflow_yaml::{self, Node},
};
use serde_json::{Value, json};
use std::{fs, process::Command};
const SCRIPT: &str = "actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd";
const LABELS: [&str; 5] = [
    "CI / Quality",
    "CI / Website",
    "CI / Linux",
    "CI / macOS",
    "CI / Windows",
];
const LANES: [&str; 5] = ["quality", "website", "linux", "macos", "windows"];
const INPUTS: [(&str, &str); 13] = [
    ("SOURCE_SHA", "source_sha"),
    ("DEFAULT_BRANCH", "default_branch"),
    ("ORIGINAL_EVENT_NAME", "original_event_name"),
    ("CORRELATION_ID", "correlation_id"),
    ("SUPERSESSION_KEY", "supersession_key"),
    ("USE_DEPOT", "use_depot"),
    ("PLAN_JSON", "plan_json"),
    ("PLAN_DIGEST", "plan_digest"),
    ("QUALITY_LANE_PLAN", "quality_lane_plan"),
    ("WEBSITE_LANE_PLAN", "website_lane_plan"),
    ("LINUX_LANE_PLAN", "linux_lane_plan"),
    ("MACOS_LANE_PLAN", "macos_lane_plan"),
    ("WINDOWS_LANE_PLAN", "windows_lane_plan"),
];
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn document() -> Node {
    workflow_yaml::parse(
        &fs::read_to_string(root().join(".github/workflows/ci-control.yml")).unwrap(),
    )
    .unwrap()
}
fn selected(doc: &Node) -> Result<&Node, String> {
    if text(doc, "name") != "CI · Manual Full" {
        return Err("manual controller name drift".into());
    }
    let on = doc.get("on").ok_or("controller trigger missing")?;
    if on
        .entries()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        != ["workflow_dispatch"]
    {
        return Err("dispatch-only controller required".into());
    }
    let dispatch = doc
        .get("jobs")
        .and_then(|jobs| jobs.get("dispatch"))
        .ok_or("dispatch job missing")?;
    if !dispatch
        .get("needs")
        .ok_or("needs missing")?
        .list()
        .contains(&"plan")
    {
        return Err("controller must need plan".into());
    }
    let Some(Node::Seq(steps)) = dispatch.get("steps") else {
        return Err("steps missing".into());
    };
    let matching = steps
        .iter()
        .filter(|step| text(step, "name") == "Create correlated checks and dispatch selected lanes")
        .collect::<Vec<_>>();
    let [step] = matching.as_slice() else {
        return Err("unique correlated check owner required".into());
    };
    if text(step, "uses") != SCRIPT {
        return Err("owned github-script pin changed".into());
    }
    let env = step.get("env").ok_or("env missing")?;
    for (key, output) in &INPUTS {
        let expected = if *key == "DEFAULT_BRANCH" {
            "${{ github.event.repository.default_branch }}".into()
        } else {
            format!("${{{{ needs.plan.outputs.{output} }}}}")
        };
        if text(env, key) != expected {
            return Err(format!("{key} must bind protected owner {expected}"));
        }
    }
    if step
        .get("with")
        .map(|with| text(with, "script"))
        .unwrap_or("")
        .is_empty()
    {
        return Err("actual scalar missing".into());
    }
    Ok(step)
}
fn run(required: [bool; 5], mutation: &str, fail_lane: &str) -> Value {
    let fixture = Fixture::new();
    let doc = document();
    let script = text(selected(&doc).unwrap().get("with").unwrap(), "script");
    let actual = fixture.path().join("actual-controller.js");
    fs::write(&actual, script).unwrap();
    let harness = fixture.path().join("controller-harness.js");
    fs::write(&harness, include_str!("controller_checks.js")).unwrap();
    let projections = LANES.iter().enumerate().map(|(index, lane)| json!({"lane":lane,"required":required[index],"required_slices":if required[index] {vec!["synthetic"]} else {vec![]},"matrices":{},"fixture_capability":{"cpu":true}})).collect::<Vec<_>>();
    let config = fixture.path().join("finite-input.json");
    fs::write(
        &config,
        serde_json::to_vec(&json!({"lanes":projections,"mutation":mutation,"fail_lane":fail_lane}))
            .unwrap(),
    )
    .unwrap();
    let mut command = Command::new("node");
    command
        .env_clear()
        .env(
            "PATH",
            std::env::var_os("PATH").expect("required finite Node component"),
        )
        .env("HOME", fixture.path())
        .arg(harness)
        .arg(actual)
        .arg(config);
    let output = fixture.run(command);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn checks(output: &Value) -> &[Value] {
    output["creates"].as_array().unwrap()
}
fn dispatches(output: &Value) -> &[Value] {
    output["dispatches"].as_array().unwrap()
}
fn common(output: &Value, required: [bool; 5]) {
    assert!(output["error"].is_null(), "{output}");
    assert_eq!(checks(output).len(), 6);
    assert_eq!(checks(output)[0]["request"]["name"], "CI Required");
    assert_eq!(checks(output)[0]["request"]["status"], "in_progress");
    for (index, check) in checks(output).iter().enumerate() {
        let request = &check["request"];
        assert_eq!(check["id"], 701 + index);
        assert_eq!(request["head_sha"], output["source"]);
        assert_eq!(request["external_id"], output["correlation"]);
        assert_eq!(request["owner"], "fixture-owner");
        assert_eq!(request["repo"], "fixture-repository");
        assert!(
            request["details_url"]
                .as_str()
                .unwrap()
                .ends_with("/actions/runs/91")
        );
        if index > 0 {
            assert_eq!(request["name"], LABELS[index - 1]);
            assert_eq!(
                request["status"],
                if required[index - 1] {
                    "queued"
                } else {
                    "completed"
                }
            );
            if required[index - 1] {
                assert!(request.get("conclusion").is_none());
            } else {
                assert_eq!(request["conclusion"], "success");
            }
        }
    }
    assert_eq!(
        dispatches(output).len(),
        required.iter().filter(|required| **required).count()
    );
    for dispatch in dispatches(output) {
        let index = LANES
            .iter()
            .position(|lane| dispatch["workflow_id"] == format!("ci-{lane}-lane.yml"))
            .unwrap();
        assert!(required[index]);
        assert_eq!(dispatch["ref"], "protected-main");
        let inputs = &dispatch["inputs"];
        assert_eq!(inputs["source_sha"], output["source"]);
        assert_eq!(inputs["plan_digest"], output["digest"]);
        assert_eq!(inputs["correlation_id"], output["correlation"]);
        assert_eq!(inputs["lane_check_id"], (702 + index).to_string());
        assert_eq!(inputs["overall_check_id"], "701");
        assert_eq!(inputs["original_event_name"], "workflow_dispatch");
        assert_eq!(inputs["supersession_key"], "finite-supersession");
        assert_eq!(
            serde_json::from_str::<Value>(inputs["expected_lane_checks"].as_str().unwrap())
                .unwrap(),
            json!(LABELS)
        );
        let projection: Value =
            serde_json::from_str(inputs["lane_plan_json"].as_str().unwrap()).unwrap();
        assert_eq!(projection, output["projections"][index]);
        assert_eq!(inputs.get("canonical_plan_json").is_some(), index == 0);
        if index == 0 {
            assert_eq!(inputs["canonical_plan_json"], output["canonical"]);
        }
        assert_eq!(inputs.get("use_depot").is_some(), index == 0 || index == 2);
        if index == 0 || index == 2 {
            assert_eq!(inputs["use_depot"], "false");
        }
    }
}
#[test]
fn controller_checks_actual_all_lanes_bind_returned_ids_and_protected_identity() {
    let required = [true; 5];
    common(&run(required, "", ""), required);
}
#[test]
fn controller_checks_absent_lanes_complete_without_dispatch() {
    for required in [[false; 5], [true, false, true, false, false]] {
        common(&run(required, "", ""), required);
    }
}
#[test]
fn controller_checks_dispatch_failure_updates_exact_created_lane_and_overall() {
    for (index, lane) in LANES.iter().enumerate() {
        let output = run([true; 5], "", &format!("ci-{lane}-lane.yml"));
        assert!(
            output["error"]
                .as_str()
                .unwrap()
                .contains("finite dispatch failure")
        );
        assert_eq!(checks(&output).len(), index + 2);
        assert_eq!(dispatches(&output).len(), index + 1);
        let updates = output["updates"].as_array().unwrap();
        assert_eq!(updates.len(), 2);
        for (update, id) in updates.iter().zip([702 + index, 701]) {
            assert_eq!(update["check_run_id"], id);
            assert_eq!(update["status"], "completed");
            assert_eq!(update["conclusion"], "failure");
        }
    }
}
#[test]
fn controller_checks_source_schema_digest_and_lane_mismatch_refuse_before_creation() {
    for (mutation, message) in [
        ("source", "protected source identity"),
        ("schema", "protected source identity"),
        ("digest", "digest mismatch"),
        ("lane", "lane identity mismatch"),
    ] {
        let output = run([true; 5], mutation, "");
        assert!(
            output["error"].as_str().unwrap().contains(message),
            "{output}"
        );
        assert!(checks(&output).is_empty());
        assert!(dispatches(&output).is_empty());
        assert_eq!(output["updates"], json!([]));
    }
}
fn change(node: &mut Node, key: &str, value: Node) {
    let Node::Map(entries) = node else {
        panic!("map")
    };
    if let Some((_, existing)) = entries.iter_mut().find(|(name, _)| name == key) {
        *existing = value;
    } else {
        entries.push((key.into(), value));
    }
}
#[test]
fn controller_checks_parsed_owner_bindings_reject_decoys_and_allow_ancillary_metadata() {
    selected(&document()).unwrap();
    for key in [
        "SOURCE_SHA",
        "CORRELATION_ID",
        "PLAN_DIGEST",
        "LINUX_LANE_PLAN",
    ] {
        let mut doc = document();
        let Node::Map(jobs) = &mut doc else {
            panic!("map")
        };
        let (_, Node::Map(job_entries)) = jobs.iter_mut().find(|(key, _)| key == "jobs").unwrap()
        else {
            panic!("jobs")
        };
        let (_, Node::Map(dispatch)) = job_entries
            .iter_mut()
            .find(|(key, _)| key == "dispatch")
            .unwrap()
        else {
            panic!("dispatch")
        };
        let (_, Node::Seq(steps)) = dispatch.iter_mut().find(|(key, _)| key == "steps").unwrap()
        else {
            panic!("steps")
        };
        let step = steps
            .iter_mut()
            .find(|step| {
                text(step, "name") == "Create correlated checks and dispatch selected lanes"
            })
            .unwrap();
        let Node::Map(fields) = step else {
            panic!("step")
        };
        let (_, env) = fields.iter_mut().find(|(key, _)| key == "env").unwrap();
        change(
            env,
            key,
            Node::Scalar("${{ needs.unowned.outputs.decoy }}".into()),
        );
        assert!(selected(&doc).unwrap_err().contains(key));
    }
    let mut doc = document();
    change(
        &mut doc,
        "run-name",
        Node::Scalar("harmless controller log label".into()),
    );
    selected(&doc).unwrap();
}
