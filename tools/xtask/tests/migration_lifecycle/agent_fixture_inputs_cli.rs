use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn invoke(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "agent-fixture-inputs"])
        .args(args)
        .output()
        .unwrap()
}

fn soak(model: &str, target: &str, path: &Path) -> Output {
    invoke(&["soak", model, target, path.to_str().unwrap()])
}

#[test]
fn fixture_hash_reads_original_bytes_and_observes_actual_edits() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory
        .path()
        .join("original implementation with spaces.py");
    fs::write(&path, b"abc").unwrap();
    let output = invoke(&["sha256", path.to_str().unwrap()]);
    assert!(output.status.success());
    assert_eq!(
        output.stdout,
        b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
    );
    fs::write(&path, []).unwrap();
    let changed = invoke(&["sha256", path.to_str().unwrap()]);
    assert!(changed.status.success());
    assert_eq!(
        changed.stdout,
        b"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n"
    );
    assert_ne!(output.stdout, changed.stdout);
    for path in [directory.path(), &directory.path().join("missing")] {
        let failure = invoke(&["sha256", path.to_str().unwrap()]);
        assert!(!failure.status.success());
        assert!(failure.stdout.is_empty());
    }
}

#[test]
fn soak_request_preserves_model_identity_escaping_and_extraction_contract() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("request with spaces.json");
    let model = "org/model:\"quoted\"\\path\n雪";
    let output = soak(model, "65536", &path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let bytes = fs::read(&path).unwrap();
    let request: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(request["model"], model);
    assert_eq!(request["stream"], false);
    assert_eq!(request["max_tokens"], 64);
    assert_eq!(request["temperature"], 0);
    assert_eq!(request.as_object().unwrap().len(), 5);
    assert_eq!(request["messages"].as_array().unwrap().len(), 2);
    assert_eq!(
        request["messages"][0],
        json!({"role":"system","content":"You are a precise long-context extraction probe."})
    );
    assert_eq!(request["messages"][1]["role"], "user");
    let content = request["messages"][1]["content"].as_str().unwrap();
    assert!(content.starts_with("This is a long-context CI soak document."));
    assert!(
        content.contains("Return exactly LONG_SOAK=ALPHA-719|MID-482|OMEGA-503 and no extra text.")
    );
    assert!(content.ends_with("\nSENTINEL_END=OMEGA-503\n"));
    let start = content.find("SENTINEL_START=ALPHA-719").unwrap();
    let middle = content.find("SENTINEL_MIDDLE=MID-482").unwrap();
    let end = content.find("SENTINEL_END=OMEGA-503").unwrap();
    assert!(start < middle && middle < end);
    assert!((32000..33500).contains(&middle));
    assert!((65300..=65536).contains(&content.chars().count()));
    assert_eq!(content.matches("SENTINEL_START=").count(), 1);
    assert_eq!(content.matches("SENTINEL_MIDDLE=").count(), 1);
    assert_eq!(content.matches("SENTINEL_END=").count(), 1);
}

#[test]
fn small_positive_soak_keeps_both_filler_halves_and_all_sentinels() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("small.json");
    assert!(soak("local model", "1", &path).status.success());
    let request: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let content = request["messages"][1]["content"].as_str().unwrap();
    assert_eq!(
        content
            .matches("FILLER: mesh long prompt soak line")
            .count(),
        2
    );
    let halves = content.split("SENTINEL_MIDDLE=MID-482").collect::<Vec<_>>();
    assert_eq!(halves.len(), 2);
    assert!(
        halves
            .iter()
            .all(|half| half.contains("Do not use this filler as the answer."))
    );
}

#[test]
fn rejected_soak_inputs_preserve_existing_output_and_emit_no_success() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("existing.json");
    for (model, target) in [
        ("", "65536"),
        ("model", "0"),
        ("model", "-1"),
        ("model", "+1"),
        ("model", "x"),
        ("model", "8388609"),
        ("model", "99999999999999999999999999999999999"),
    ] {
        fs::write(&path, b"previous request").unwrap();
        let output = soak(model, target, &path);
        assert!(!output.status.success(), "{model}: {target}");
        assert!(output.stdout.is_empty());
        assert_eq!(fs::read(&path).unwrap(), b"previous request");
    }
    let model = "x".repeat(65537);
    assert!(!soak(&model, "1", &path).status.success());
    assert!(
        !soak("model", "1", &directory.path().join("absent/out.json"))
            .status
            .success()
    );
}

#[test]
fn surface_request_preserves_tool_schema_and_compatibility_fields() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("probe request with spaces.json");
    let model = "selected\"\\\n雪";
    let result = invoke(&["surface", model, output.to_str().unwrap()]);
    assert!(result.status.success(), "{result:?}");
    assert!(result.stdout.is_empty());
    let request: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(
        request,
        json!({
            "model":model,
            "messages":[{"role":"system","content":"You are a brief CI compatibility probe."},{"role":"user","content":"Reply with ok, or call the tool if needed."}],
            "tools":[{"type":"function","function":{"name":"get_fixture_fact","description":"Return one known fact from the smoke fixture.",
                "parameters":{"type":"object","properties":{"key":{"type":"string","enum":["codeword","checksum"]}},"required":["key"],"additionalProperties":false}}}],
            "tool_choice":"auto","parallel_tool_calls":true,"stream":false,"max_tokens":8,"temperature":0
        })
    );
    fs::write(&output, "existing request").unwrap();
    assert!(
        !invoke(&["surface", "", output.to_str().unwrap()])
            .status
            .success()
    );
    assert_eq!(fs::read(&output).unwrap(), b"existing request");
}
