use serde_json::{Value, json};
use std::{fs, process::Command};
fn green(result: &str) -> Value {
    json!({"result":result,"outputs":{"state":"green","green":"true","repairable":"false","package":"verified-package","identity":"b".repeat(64),"head":"a".repeat(40),"branch":"llama-canary/fixture"}})
}
fn invoke(args: &[String]) -> (std::process::Output, String) {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outputs");
    fs::write(&path, "prior=preserved\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "canary-receipts"])
        .args(args)
        .env("GITHUB_OUTPUT", &path)
        .output()
        .unwrap();
    (output, fs::read_to_string(path).unwrap())
}
fn attempt_args(repair: Value, verification: Value) -> Vec<String> {
    vec![
        "select-attempt".into(),
        "--changed".into(),
        "true".into(),
        "--repair-json".into(),
        repair.to_string(),
        "--verification-json".into(),
        verification.to_string(),
    ]
}
fn final_args(attempts: Value) -> Vec<String> {
    vec![
        "select-final".into(),
        "--certify".into(),
        "true".into(),
        "--changed".into(),
        "true".into(),
        "--mesh-source".into(),
        "".into(),
        "--preflight".into(),
        "success".into(),
        "--attempts-json".into(),
        attempts.to_string(),
    ]
}
#[test]
fn selection_cli_preserves_reconciled_green_and_emits_verified_identity() {
    let (output, bytes) = invoke(&attempt_args(green("failure"), green("failure")));
    assert!(output.status.success(), "{output:?}");
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["state"], "green");
    assert_eq!(document["head"], "a".repeat(40));
    assert!(bytes.starts_with("prior=preserved\n"));
    assert!(!bytes.contains("publish=true"));
}
#[test]
fn final_selection_cli_writes_publish_last_and_rejects_later_failed_slot() {
    let input = json!({"attempt_1":green("success"),"attempt_2":{"result":"skipped","outputs":{}},"attempt_3":{"result":"skipped","outputs":{}}});
    let (output, bytes) = invoke(&final_args(input.clone()));
    assert!(output.status.success(), "{output:?}");
    assert!(bytes.ends_with("publish=true\n"));
    for job in [green("failure"), json!({"result":"cancelled","outputs":{}})] {
        let mut changed = input.clone();
        changed["attempt_3"] = job;
        let (output, bytes) = invoke(&final_args(changed));
        assert!(!output.status.success(), "{output:?}");
        assert_eq!(bytes, "prior=preserved\n");
    }
}
#[test]
fn selection_cli_validates_complete_decision_before_any_output_append() {
    for (field, value) in [
        ("branch", "x\npublish=true"),
        ("identity", "invalid"),
        ("head", ""),
        ("package", ""),
    ] {
        let mut verifier = green("success");
        verifier["outputs"][field] = json!(value);
        let (output, bytes) = invoke(&attempt_args(green("success"), verifier));
        assert!(!output.status.success(), "{output:?}");
        assert_eq!(bytes, "prior=preserved\n");
    }
    for value in ["TRUE", "1", "true\n"] {
        let mut args = attempt_args(green("success"), green("success"));
        args[2] = value.into();
        let (output, bytes) = invoke(&args);
        assert!(!output.status.success());
        assert_eq!(bytes, "prior=preserved\n");
    }
}
#[test]
fn selection_cli_rejects_foreign_slot_and_non_string_flags_without_output_mutation() {
    let input = json!({"attempt_1":green("success"),"attempt_4":green("success")});
    let (output, bytes) = invoke(&final_args(input));
    assert!(!output.status.success());
    assert_eq!(bytes, "prior=preserved\n");
    let mut candidate = green("success");
    candidate["outputs"]["green"] = json!(true);
    let (output, bytes) = invoke(&attempt_args(candidate, green("success")));
    assert!(!output.status.success());
    assert_eq!(bytes, "prior=preserved\n");
}

#[test]
fn selection_cli_refuses_contradictory_producer_flags_without_output_mutation() {
    for verifier in [false, true] {
        let mut repair = green("failure");
        let mut verification = green("failure");
        if verifier {
            verification["outputs"]["repairable"] = json!("true");
        } else {
            repair["outputs"]["repairable"] = json!("true");
        }
        let (output, bytes) = invoke(&attempt_args(repair, verification));
        assert!(!output.status.success(), "{output:?}");
        assert_eq!(bytes, "prior=preserved\n");
    }
}

#[test]
fn final_selection_cli_accepts_real_needs_projection_and_uses_latest_verified_slot() {
    let mut latest = green("success");
    latest["outputs"]["package"] = json!("independent-verified-package-3");
    latest["outputs"]["identity"] = json!("c".repeat(64));
    let input = json!({
        "resolve":{"result":"success","outputs":{"source":"d".repeat(40),"changed":"true","certify":"true"}},
        "preflight":{"result":"success","outputs":{}},
        "attempt_1":green("success"),
        "attempt_2":{"result":"success","outputs":{"state":"repairable","repairable":"true","failure_class":"candidate"}},
        "attempt_3":latest,
    });
    let (output, bytes) = invoke(&final_args(input));
    assert!(output.status.success(), "{output:?}");
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["package"], "independent-verified-package-3");
    assert_eq!(document["identity"], "c".repeat(64));
    assert!(bytes.ends_with("publish=true\n"));
}
