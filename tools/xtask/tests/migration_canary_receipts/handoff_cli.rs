use super::support::{Fixture, PLAN};
use crate::canary_receipts::Digest;
use serde_json::{Value, json};
use std::{
    fs,
    process::{Command, Output},
};

fn package(pass: &str, bundle: bool) -> (Fixture, Value) {
    let fixture = Fixture::new();
    let package = fixture.0.join("package");
    fs::create_dir(&package).unwrap();
    let mut identity = json!({"schema":3,"platform":"macos-arm64-metal","candidate":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","base":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","controller":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","mesh_source":"","pass_id":pass,"branch":"llama-canary/fixture","run_id":"123","run_attempt":"2","bundle_sha256":null});
    for (name, field, bytes) in [
        ("plan.json", "plan_sha256", PLAN),
        (
            "binaries.tar",
            "binaries_sha256",
            b"fixture-binaries".as_slice(),
        ),
        (
            "workload-oracles.tar",
            "workload_oracles_sha256",
            b"fixture-workload".as_slice(),
        ),
        (
            "llama-source.bundle",
            "llama_bundle_sha256",
            b"fixture-native".as_slice(),
        ),
        (
            "llama-source.json",
            "llama_provenance_sha256",
            b"fixture-provenance".as_slice(),
        ),
        (
            "upstream-summary.md",
            "summary_sha256",
            b"Upstream fixture.\n".as_slice(),
        ),
    ] {
        fs::write(package.join(name), bytes).unwrap();
        identity[field] = Digest::of_bytes(bytes).as_str().into();
    }
    if bundle {
        fs::write(package.join("candidate.bundle"), b"fixture-candidate").unwrap();
        identity["bundle_sha256"] = Digest::of_bytes(b"fixture-candidate").as_str().into();
    }
    let bytes = serde_json::to_vec(&identity).unwrap();
    fs::write(package.join("identity.json"), &bytes).unwrap();
    let input = json!({"package":package,"identity_sha256":Digest::of_bytes(&bytes).as_str(),"run_id":"123","run_attempt":"4","controller_revision":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"});
    (fixture, input)
}
fn invoke(fixture: &Fixture, input: &Value, verb: &str) -> Output {
    let path = fixture.0.join("input.json");
    fs::write(&path, serde_json::to_vec(input).unwrap()).unwrap();
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "canary-receipts", verb, "--input"])
        .arg(path)
        .output()
        .unwrap()
}
#[test]
fn actual_receipt_cli_binds_current_attempt_and_rejects_unplanned_family() {
    let (fixture, mut input) = package("repair-1", false);
    let evidence = fixture.0.join("evidence");
    input["evidence"] = json!(evidence);
    input["family"] = json!("dense");
    input["outcome"] = json!("failure");
    input["runner"] = json!("fixture-runner");
    let output = invoke(&fixture, &input, "receipt");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&fs::read(evidence.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["run_attempt"], "4");
    assert_eq!(receipt["outcome"], "failure");
    assert_eq!(receipt["results_sha256"], Value::Null);
    assert_eq!(receipt["identity_sha256"], input["identity_sha256"]);
    input["family"] = json!("foreign");
    input["evidence"] = json!(fixture.0.join("foreign"));
    assert!(!invoke(&fixture, &input, "receipt").status.success());
    assert!(!fixture.0.join("foreign").exists());
}
#[test]
fn actual_publication_cli_requires_independent_verified_bundle_and_retains_summary() {
    for (pass, bundle, accepted) in [
        ("verify-1", true, true),
        ("repair-1", true, false),
        ("verify-1", false, false),
    ] {
        let (fixture, mut input) = package(pass, bundle);
        input["repository"] = json!("Mesh-LLM/mesh-llm");
        let output = invoke(&fixture, &input, "publication");
        assert_eq!(
            output.status.success(),
            accepted,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let path = fixture.0.join("package/pr-body.md");
        if accepted {
            let body = fs::read_to_string(path).unwrap();
            assert!(body.contains("actions/runs/123"));
            assert!(body.contains("verify-1; 2 families"));
            assert!(body.ends_with("Upstream fixture.\n"));
        } else {
            assert!(!path.exists());
        }
    }
}
#[test]
fn actual_handoff_rejects_foreign_run_and_corrupt_artifact_before_output() {
    for corrupt in [false, true] {
        let (fixture, mut input) = package("verify-1", true);
        input["repository"] = json!("Mesh-LLM/mesh-llm");
        if corrupt {
            fs::write(fixture.0.join("package/binaries.tar"), b"changed").unwrap();
        } else {
            input["run_id"] = json!("999");
        }
        assert!(!invoke(&fixture, &input, "publication").status.success());
        assert!(!fixture.0.join("package/pr-body.md").exists());
    }
}
