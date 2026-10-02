use serde_json::{Value, json};
use std::{fs, process::Command};

fn needs() -> Value {
    json!({
        "resolve":{"result":"success","outputs":{"certify":"true","changed":"true","mesh_source":""}},
        "preflight":{"result":"success","outputs":{}},
        "candidate":{"result":"success","outputs":{"green":"true","head":"candidate-head"}},
        "verification":{"result":"success","outputs":{"green":"true","head":"candidate-head","package":"verified-package","identity":"verified-identity","branch":"llama-canary/fixture"}}
    })
}

fn invoke(input: &Value) -> (std::process::Output, String) {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("outputs");
    fs::write(&path, "prior=preserved\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "canary-receipts", "result"])
        .env("NEEDS_JSON", serde_json::to_string(input).unwrap())
        .env("GITHUB_OUTPUT", &path)
        .output()
        .unwrap();
    (output, fs::read_to_string(path).unwrap())
}

#[test]
fn changed_pin_appends_only_independently_verified_identity() {
    let (output, bytes) = invoke(&needs());
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(
        bytes,
        "prior=preserved\npackage=verified-package\nidentity=verified-identity\nhead=candidate-head\nbranch=llama-canary/fixture\npublish=true\n"
    );
}

#[test]
fn approved_skip_unchanged_and_historical_runs_never_publish() {
    let mut skipped = needs();
    skipped["resolve"]["outputs"]["certify"] = json!("false");
    skipped["preflight"]["result"] = json!("skipped");
    skipped["candidate"]["outputs"] = json!({});
    skipped["verification"]["outputs"] = json!({});
    let mut unchanged = needs();
    unchanged["resolve"]["outputs"]["changed"] = json!("false");
    unchanged["verification"]["outputs"] = json!({});
    let mut historical = unchanged.clone();
    historical["resolve"]["outputs"]["mesh_source"] = json!("candidate-head");
    for input in [skipped, unchanged, historical] {
        let (output, bytes) = invoke(&input);
        assert!(output.status.success(), "{:?}", output);
        assert_eq!(bytes, "prior=preserved\n");
    }
}

#[test]
fn resolution_preflight_and_every_certification_failure_deny_publication() {
    for (job, field, value, error) in [
        (
            "resolve",
            "result",
            "failure",
            "trusted source resolution failed",
        ),
        (
            "preflight",
            "result",
            "failure",
            "canary environment preflight failed",
        ),
        (
            "verification",
            "green",
            "false",
            "independent family verification failed",
        ),
        (
            "verification",
            "head",
            "different-head",
            "independent family verification failed",
        ),
        (
            "candidate",
            "head",
            "",
            "independent family verification failed",
        ),
    ] {
        let mut input = needs();
        if field == "result" {
            input[job][field] = json!(value);
        } else {
            input[job]["outputs"][field] = json!(value);
        }
        assert_denied(&input, error);
    }
    let mut unchanged = needs();
    unchanged["resolve"]["outputs"]["changed"] = json!("false");
    unchanged["candidate"]["outputs"]["green"] = json!("false");
    assert_denied(&unchanged, "unchanged-pin family certification failed");
    let mut historical = needs();
    historical["resolve"]["outputs"]["mesh_source"] = json!("other-head");
    assert_denied(
        &historical,
        "selected MeshLLM revision certification failed",
    );
    historical["resolve"]["outputs"]["changed"] = json!("false");
    assert_denied(
        &historical,
        "selected MeshLLM revision certification failed",
    );
}

fn assert_denied(input: &Value, error: &str) {
    let (output, bytes) = invoke(input);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(error),
        "{:?}",
        output
    );
    assert_eq!(bytes, "prior=preserved\n");
}

#[test]
fn candidate_failure_class_and_stage_remain_visible() {
    let mut input = needs();
    input["candidate"]["outputs"]["green"] = json!("false");
    assert_denied(
        &input,
        "candidate failure during build-or-family-certification; publication denied",
    );
    input["candidate"]["outputs"]["failure_class"] = json!("environment");
    input["candidate"]["outputs"]["failure_stage"] = json!("native-toolchain");
    assert_denied(
        &input,
        "environment failure during native-toolchain; publication denied",
    );
}

#[test]
fn green_outputs_from_failed_skipped_or_cancelled_jobs_do_not_certify() {
    for result in ["failure", "skipped", "cancelled"] {
        for job in ["candidate", "verification"] {
            let mut input = needs();
            input[job]["result"] = json!(result);
            let (output, bytes) = invoke(&input);
            assert!(!output.status.success(), "{job}/{result}: {output:?}");
            assert_eq!(bytes, "prior=preserved\n");
        }
        for historical in [false, true] {
            let mut input = needs();
            input["resolve"]["outputs"]["changed"] = json!("false");
            if historical {
                input["resolve"]["outputs"]["mesh_source"] = json!("candidate-head");
            }
            input["candidate"]["result"] = json!(result);
            let (output, bytes) = invoke(&input);
            assert!(
                !output.status.success(),
                "{result}/{historical}: {output:?}"
            );
            assert_eq!(bytes, "prior=preserved\n");
        }
    }
}

#[test]
fn malformed_missing_and_injected_outputs_never_append_partial_publish() {
    for value in [json!(null), json!(true), json!(""), json!("unknown")] {
        let mut input = needs();
        input["resolve"]["outputs"]["certify"] = value;
        let (output, bytes) = invoke(&input);
        assert!(!output.status.success());
        assert_eq!(bytes, "prior=preserved\n");
    }
    for key in ["package", "identity", "head", "branch"] {
        let mut input = needs();
        input["verification"]["outputs"]
            .as_object_mut()
            .unwrap()
            .remove(key);
        let (output, bytes) = invoke(&input);
        assert!(!output.status.success());
        assert_eq!(bytes, "prior=preserved\n");
    }
    for value in ["x\npublish=true", "x\ry", "x\0y"] {
        let mut input = needs();
        input["verification"]["outputs"]["branch"] = json!(value);
        assert_denied(&input, "verified publication output branch");
    }
}

#[test]
fn output_file_is_required_only_when_publication_is_approved() {
    let run = |input: Value| {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "canary-receipts", "result"])
            .env("NEEDS_JSON", input.to_string())
            .env_remove("GITHUB_OUTPUT")
            .output()
            .unwrap()
    };
    let output = run(needs());
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("GITHUB_OUTPUT is required"));
    let mut input = needs();
    input["resolve"]["outputs"]["certify"] = json!("false");
    assert!(run(input).status.success());
}
