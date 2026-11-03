#![cfg(feature = "automation-contracts")]
#[path = "ci_governance_fixtures/support.rs"]
mod support;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};
use support::{TestResult, denied, root, run, success, write_json};

const POLICY: &str = "ci/governance/policy.json";
const EXPECTED: &str = "ci/governance/expected-state.json";
const LINK: &str = "https://github.com/Mesh-LLM/mesh-llm/issues/2781";

fn digest(value: &Value) -> String {
    let mut value = value.clone();
    value.sort_all_objects();
    hex::encode(Sha256::digest(serde_json::to_vec(&value).unwrap()))
}

fn policy_digest() -> String {
    digest(&serde_json::from_slice(&fs::read(root().join(POLICY)).unwrap()).unwrap())
}

fn human(login: &str, id: u64) -> Value {
    json!({"login":login,"id":id,"kind":"human"})
}

fn desired_state() -> Value {
    let expected: Value =
        serde_json::from_slice(&fs::read(root().join(EXPECTED)).unwrap()).unwrap();
    let mut state = expected["state"].clone();
    // Emit canonical fixture arrays so the reviewed request digest identifies
    // the same settings regardless of the human-readable source file's order.
    for pointer in [
        "/reviewer_team/members",
        "/native_rule/bypass_actors",
        "/native_rule/scope/directory_prefixes",
        "/native_rule/scope/exact_paths",
        "/native_rule/scope/root_filename_prefixes",
        "/administration_environment/reviewers",
        "/administration_environment/allowed_refs",
        "/credentials/observer_permissions",
    ] {
        state
            .pointer_mut(pointer)
            .unwrap()
            .as_array_mut()
            .unwrap()
            .sort_by_key(Value::to_string);
    }
    state
}

/// Synthetic local evidence, never an assertion about GitHub's current state.
fn fixture() -> (Value, Value) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let desired_state = desired_state();
    let execution = json!({
        "repository":"Mesh-LLM/mesh-llm", "run_id":101,"run_attempt":1,
        "ref_name":"refs/heads/main","protected_revision":"a".repeat(40),
        "operator":human("micspiral",1),"triggerer":human("micspiral",1),"rerun_actor":human("micspiral",1)
    });
    let mut observed: Value = serde_json::from_str(include_str!(
        "../../../ci/governance/examples/observation.template.json"
    ))
    .unwrap();
    observed["complete"] = json!(true);
    observed["observed_at_unix"] = json!(now - 10);
    observed["protected_policy_revision"] = json!("a".repeat(40));
    observed["current_protected_main_revision"] = json!("a".repeat(40));
    observed["policy_sha256"] = json!(policy_digest());
    observed["readback_link"] = json!(LINK);
    observed["replay_ledger_complete"] = json!(true);
    observed["replay_ledger_link"] = json!(LINK);
    observed["execution"] = execution.clone();
    observed["observer"] = json!({
        "repository":"Mesh-LLM/mesh-llm", "run_id":101, "run_attempt":1,
        "ref_name":"refs/heads/main", "protected_revision":"a".repeat(40),
        "principal":human("micspiral",1)
    });
    observed["state"] = desired_state.clone();
    observed["state"]["native_rule"]["required_approvals"] = json!(0);
    let mut record: Value = serde_json::from_str(include_str!(
        "../../../ci/governance/examples/change-record.template.json"
    ))
    .unwrap();
    let request = &mut record["request"];
    request["operation_id"] = json!("ba2e26e4-76a4-462f-894b-1a33d80d01aa");
    request["policy_revision"] = observed["protected_policy_revision"].clone();
    request["policy_sha256"] = observed["policy_sha256"].clone();
    request["expected_before_sha256"] = json!(digest(&observed["state"]));
    request["before"] = observed["state"].clone();
    request["desired_after"] = desired_state;
    request["controls"] = json!(["ci-ruleset"]);
    request["execution"] = execution.clone();
    for field in ["owner", "author", "sponsor"] {
        request[field] = human("micspiral", 1);
    }
    request["rationale"] =
        json!("Finite local acceptance fixture for the reviewed governance plan");
    request["affected_jobs"] = json!(["CI policy / independent approval"]);
    request["tests"] = json!(["local CLI contracts only"]);
    request["rollout"] =
        json!("Authorized human operation and authenticated readback remain pending");
    request["rollback"] = json!("Separately reviewed restoration of the captured before state");
    request["requested_at_unix"] = json!(now - 100);
    request["expires_at_unix"] = json!(now + 600);
    let request_digest = digest(request);
    let approval = &mut record["approval"];
    approval["approver"] = human("ndizazzo", 2);
    approval["request_sha256"] = json!(request_digest);
    approval["policy_revision"] = observed["protected_policy_revision"].clone();
    approval["execution"] = execution;
    approval["approved_at_unix"] = json!(now - 20);
    approval["expires_at_unix"] = json!(now + 300);
    approval["link"] = json!(LINK);
    (observed, record)
}

fn check(observed: &Value, record: Option<&Value>) -> std::process::Output {
    let directory = tempfile::tempdir().unwrap();
    let observed_path = write_json(directory.path(), "observed fixture.json", observed);
    let mut args = vec![
        if record.is_some() {
            "validate-change"
        } else {
            "check-drift"
        },
        "--policy",
        POLICY,
        "--expected",
        EXPECTED,
        "--observed",
        observed_path.to_str().unwrap(),
    ];
    let record_path;
    if let Some(record) = record {
        record_path = write_json(directory.path(), "reviewed record.json", record);
        args.extend(["--record", record_path.to_str().unwrap()]);
    }
    run(&args)
}

#[test]
fn checked_in_desired_state_matches_synthetic_readback_without_claiming_live_evidence() {
    let (mut observed, _) = fixture();
    let expected: Value =
        serde_json::from_slice(&fs::read(root().join(EXPECTED)).unwrap()).unwrap();
    observed["state"] = expected["state"].clone();
    let result = success(check(&observed, None));
    assert_eq!(result["status"], "MATCH");
    assert_eq!(result["proposed_detection_target_seconds"], 900);
    for field in [
        "execution_authorized",
        "evidence_authenticated",
        "live_detection_latency_measured",
    ] {
        assert_eq!(result[field], false, "{field}");
    }
    observed["state"]["reviewer_team"]["members"] = json!(["micspiral", "ndizazzo", "unlisted"]);
    denied(check(&observed, None), "CI governance drift: reviewer-team");
}

#[test]
fn reordered_settings_match_but_duplicate_members_are_not_valid_readback() {
    let (mut observed, _) = fixture();
    observed["state"] = desired_state();
    observed["state"]["reviewer_team"]["members"]
        .as_array_mut()
        .unwrap()
        .reverse();
    observed["state"]["native_rule"]["scope"]["exact_paths"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert_eq!(success(check(&observed, None))["status"], "MATCH");
    let member = observed["state"]["reviewer_team"]["members"][0].clone();
    observed["state"]["reviewer_team"]["members"]
        .as_array_mut()
        .unwrap()
        .push(member);
    denied(check(&observed, None), "duplicate");
}

#[test]
fn read_only_service_drift_observation_needs_no_administration_execution() {
    let (mut observed, _) = fixture();
    observed["state"] = desired_state();
    observed["execution"] = Value::Null;
    observed["observer"]["principal"] = json!({"login":"governance-observer","id":99,"kind":"bot"});
    assert_eq!(success(check(&observed, None))["status"], "MATCH");
    let (_, record) = fixture();
    denied(check(&observed, Some(&record)), "execution");
}

#[test]
fn supplied_plan_needs_independent_digest_bound_approval_and_exact_before_state() {
    let (observed, record) = fixture();
    let result = success(check(&observed, Some(&record)));
    assert_eq!(result["status"], "VALIDATED_PLAN");
    assert_eq!(result["execution_authorized"], false);
    assert_eq!(result["evidence_authenticated"], false);
    for (pointer, value, reason) in [
        (
            "/approval/approver",
            human("micspiral", 1),
            "approver must differ",
        ),
        (
            "/approval/request_sha256",
            json!("b".repeat(64)),
            "request-digest",
        ),
        (
            "/request/expected_before_sha256",
            json!("b".repeat(64)),
            "before-state precondition",
        ),
        (
            "/request/execution/run_attempt",
            json!(2),
            "exact observed run/attempt/ref/actors",
        ),
    ] {
        let mut changed = record.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        denied(check(&observed, Some(&changed)), reason);
    }
}

#[test]
fn stale_incomplete_replayed_or_off_main_observations_fail() {
    let (observed, record) = fixture();
    for (pointer, value, reason) in [
        ("/complete", json!(false), "incomplete observation"),
        ("/observed_at_unix", json!(1), "stale"),
        (
            "/current_protected_main_revision",
            json!("b".repeat(40)),
            "protected",
        ),
        (
            "/execution/ref_name",
            json!("refs/heads/feature"),
            "exact current protected main",
        ),
        (
            "/consumed_operation_ids",
            json!([record["request"]["operation_id"]]),
            "replay rejected",
        ),
    ] {
        let mut changed = observed.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        denied(check(&changed, Some(&record)), reason);
    }
}

#[test]
fn unfilled_templates_and_omitted_operation_arguments_are_rejected() -> TestResult {
    denied(
        run(&[
            "check-drift",
            "--policy",
            POLICY,
            "--expected",
            EXPECTED,
            "--observed",
            "ci/governance/examples/observation.template.json",
        ]),
        "invalid governance JSON schema",
    );
    denied(
        run(&[
            "validate-change",
            "--policy",
            POLICY,
            "--expected",
            EXPECTED,
        ]),
        "required options",
    );
    let (mut observed, _) = fixture();
    observed
        .as_object_mut()
        .ok_or("observation object")?
        .remove("observer");
    denied(check(&observed, None), "invalid governance JSON schema");
    Ok(())
}
