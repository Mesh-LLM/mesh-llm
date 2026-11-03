use super::*;
use serde_json::{Value, json};
use std::collections::BTreeSet;

fn selection() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/decision-selector-trace.json"
    ))
    .unwrap()
}

fn request(trace: &str, context: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({"selection_trace": trace, "context": context})).unwrap()
}

#[test]
fn supplied_context_remains_unverified_and_runner_name_is_not_emitted() {
    let trace = selection().to_string();
    let input = request(
        &trace,
        json!({
            "role": "linux-native-16", "run_id": 42, "run_attempt": 2,
            "job_id": "private_job_id", "source_sha": "b".repeat(40),
            "policy_sha": "c".repeat(40), "started_at": 1234,
            "runner": {"name": "private-runner-name", "os": "Linux", "arch": "X64"}
        }),
    );
    let output = render(&input).unwrap();
    assert!(!output.contains("private-runner-name"));
    assert!(!output.contains("private_job_id"));
    let record: Value = serde_json::from_str(&output).unwrap();
    assert_eq!(record["kind"], "execution_context_claim");
    assert_eq!(record["authenticity"], "unverified");
    assert_eq!(record["selection_to_job_binding"], "unverified");
    assert_eq!(record["selection_sha256"], digest(trace.as_bytes()));
    assert_eq!(record["claimed_context"]["role"], "linux-native-16");
    assert_eq!(
        record["claimed_context"]["job_id_sha256"],
        digest(b"private_job_id")
    );
    assert_eq!(
        record["claimed_context"]["runner"]["name_sha256"],
        digest(b"private-runner-name")
    );
    assert_eq!(record["actual_runner"], Value::Null);
    assert_eq!(record["effective_cache_mode"], Value::Null);
    assert_eq!(record["verified_policy_revision"], Value::Null);
    assert_eq!(record["retention"], "not_recorded");
    assert_eq!(record["authorizes_execution"], false);
}

#[test]
fn absent_context_stays_unknown_and_digest_binds_exact_trace_bytes() {
    let trace = selection().to_string();
    let input = serde_json::to_vec(&json!({"selection_trace": trace})).unwrap();
    let record: Value = serde_json::from_str(&render(&input).unwrap()).unwrap();
    assert_eq!(record["claimed_context"]["runner"], Value::Null);
    assert_eq!(record["claimed_context"]["role"], Value::Null);
    assert_eq!(record["claimed_context"]["policy_sha"], Value::Null);
    let padded = format!(" {trace} ");
    let other: Value =
        serde_json::from_str(&render(&request(&padded, json!({}))).unwrap()).unwrap();
    assert_ne!(record["selection_sha256"], other["selection_sha256"]);
    assert_eq!(record["selection"], other["selection"]);
}

#[test]
fn malformed_or_forged_trace_fields_are_rejected_without_echoing_input() {
    let mut cases = Vec::new();
    let mut candidate = selection();
    candidate["actual_runner"] = json!("forged-runner");
    cases.push(candidate);
    let mut candidate = selection();
    candidate["selected_labels"]["linux"] = json!("private-value");
    cases.push(candidate);
    let mut candidate = selection();
    candidate["force_hosted"] = json!("true");
    cases.push(candidate);
    let mut candidate = selection();
    candidate["unknown"] = json!("private-value");
    cases.push(candidate);
    let mut candidate = selection();
    candidate
        .as_object_mut()
        .unwrap()
        .remove("declared_head_sha");
    cases.push(candidate);
    for candidate in cases {
        let error = render(&request(&candidate.to_string(), json!({}))).unwrap_err();
        assert_eq!(error, "invalid or oversized execution-record input");
        assert!(!error.contains("private-value"));
    }
}

#[test]
fn contradictory_provider_label_reason_and_cache_claims_are_rejected() {
    let mut depot = selection();
    depot["selected_provider"] = json!("depot");
    depot["force_hosted"] = json!(false);
    depot["reason"] = json!("eligible_depot_selection");
    depot["allow_native_github_cache"] = json!(false);
    depot["allow_trusted_sccache_seed"] = json!(false);
    depot["selected_labels"] = json!({
        "linux": "depot-ubuntu-24.04", "linux_4": "depot-ubuntu-24.04-4",
        "linux_8": "depot-ubuntu-24.04-8", "linux_16": "depot-ubuntu-24.04-16",
        "linux_arm": "depot-ubuntu-24.04-arm", "linux_arm_4": "depot-ubuntu-24.04-arm-4",
        "linux_arm_8": "depot-ubuntu-24.04-arm-8", "linux_arm_16": "depot-ubuntu-24.04-arm-16",
        "macos": "depot-macos-15", "windows": "depot-windows-2022"
    });
    assert!(render(&request(&depot.to_string(), json!({}))).is_ok());

    let mut cases = Vec::new();
    for (path, value) in [
        ("linux", "depot-ubuntu-24.04"),
        ("linux_4", "ubuntu-24.04-arm"),
        ("macos", "depot-macos-15"),
    ] {
        let mut candidate = selection();
        candidate["selected_labels"][path] = json!(value);
        cases.push(candidate);
    }
    for (field, value) in [
        ("reason", json!("eligible_depot_selection")),
        ("force_hosted", json!(false)),
        ("allow_depot_remote_cache", json!(true)),
        ("allow_native_github_cache", json!(false)),
        ("allow_trusted_sccache_seed", json!(false)),
        ("exception_active", json!(true)),
    ] {
        let mut candidate = selection();
        candidate[field] = value;
        cases.push(candidate);
    }
    for (field, value) in [
        ("force_hosted", json!(true)),
        ("reason", json!("not_eligible_or_disabled")),
        ("allow_native_github_cache", json!(true)),
        ("allow_trusted_sccache_seed", json!(true)),
    ] {
        let mut candidate = depot.clone();
        candidate[field] = value;
        cases.push(candidate);
    }
    for candidate in cases {
        assert_eq!(
            render(&request(&candidate.to_string(), json!({}))).unwrap_err(),
            "invalid or oversized execution-record input"
        );
    }
}

#[test]
fn invalid_claims_unknown_fields_and_oversized_input_are_rejected() {
    let trace = selection().to_string();
    for context in [
        json!({"role": "runtime\nprivate"}),
        json!({"role": "unclassified"}),
        json!({"run_id": 0}),
        json!({"run_attempt": 0}),
        json!({"source_sha": "main"}),
        json!({"policy_sha": "A".repeat(40)}),
        json!({"runner": {"name": "host\nsecret"}}),
        json!({"runner": {"os": "unknown"}}),
        json!({"unknown": "private-value"}),
    ] {
        assert_eq!(
            render(&request(&trace, context)).unwrap_err(),
            "invalid or oversized execution-record input"
        );
    }
    assert_eq!(
        render(&vec![b'x'; MAX_INPUT as usize + 1]).unwrap_err(),
        "invalid or oversized execution-record input"
    );
}

#[test]
fn admitted_role_names_follow_the_checked_in_slice_catalog() {
    fn collect(value: &Value, roles: &mut BTreeSet<String>) {
        match value {
            Value::Object(fields) => {
                if let Some(Value::String(role)) = fields.get("runner_role") {
                    roles.insert(role.clone());
                }
                for child in fields.values() {
                    collect(child, roles);
                }
            }
            Value::Array(items) => {
                for item in items {
                    collect(item, roles);
                }
            }
            _ => {}
        }
    }
    let catalog = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ci/slices.yml");
    let catalog: Value = serde_json::from_slice(&std::fs::read(catalog).unwrap()).unwrap();
    let mut actual = BTreeSet::new();
    collect(&catalog, &mut actual);
    assert_eq!(
        actual,
        SUPPORTED_ROLES
            .iter()
            .map(|role| (*role).to_owned())
            .collect()
    );
}
