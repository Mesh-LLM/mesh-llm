//! Resolver edge cases: output modes, integrity verification, path and
//! newline injection, SHA/size shape, cadence, selection and argv handling.

use crate::support::{
    RESOLVER, Stage, TestResult, assert_same, assert_streams, code, field, fixture, legacy,
    run_legacy, xtask,
};
use serde_json::Value;

/// serde_json's parser wording replaces Python's `JSONDecodeError` text;
/// the prefix and status stay identical.
const EXPLAINED: &[&str] = &["invalid-json"];

const PAYLOAD: &[u8] = b"immutable fixture\n";

fn stage_case(stage: &Stage, case: &Value) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    match (&case["manifest"]["json"], case["manifest"]["text"].as_str()) {
        (Value::Null, Some(text)) => stage.write("m.json", text.as_bytes())?,
        (json, _) => stage.write("m.json", &serde_json::to_vec(json)?)?,
    }
    stage.write("gh.out", b"preexisting=1\n")?;
    let args = case["args"].as_array().ok_or("args must be an array")?;
    args.iter()
        .map(|arg| {
            let arg = arg.as_str().ok_or("arg must be a string")?;
            Ok(stage
                .expand(arg)?
                .replace("{m}", "m.json")
                .replace("{out}", "gh.out"))
        })
        .collect()
}

#[test]
fn migration_models_resolver_edge_cases_match_legacy() -> TestResult {
    // Given: a verified fixture file and every captured legacy edge case.
    let stage = Stage::new("resolver")?;
    stage.write("fixture.bin", PAYLOAD)?;
    let cases = fixture("resolver-cases.json")?;
    let cases = cases.as_array().ok_or("cases must be an array")?;
    let python = legacy("MIGRATION_MODELS_LEGACY_PYTHON");
    let script = python
        .as_ref()
        .map(|_| stage.legacy_script(RESOLVER))
        .transpose()?;
    for case in cases {
        let name = field(case, "name")?;
        let args = stage_case(&stage, case)?;
        let args = args.iter().map(String::as_str).collect::<Vec<_>>();
        let mut ported_args = vec!["models", "resolve"];
        ported_args.extend_from_slice(&args);
        // When: the ported resolver runs the same argv.
        let ported = xtask(stage.path(), &ported_args)?;
        // Then: status, streams and the appended GitHub output match.
        let stdout = stage.expand(field(case, "stdout")?)?;
        let stderr = stage.expand(field(case, "stderr")?)?;
        if EXPLAINED.contains(&name) {
            assert_eq!(ported.status.code(), Some(code(case)?), "{name}: status");
            let actual = String::from_utf8_lossy(&ported.stderr);
            assert!(
                actual.starts_with("test-model manifest error: "),
                "{name}: {actual}"
            );
        } else {
            assert_streams(name, &ported, code(case)?, &stdout, &stderr);
        }
        assert_eq!(
            stage.read("gh.out")?,
            field(case, "github_output")?,
            "{name}: github output"
        );
        if let (Some(python), Some(script)) = (&python, &script) {
            stage_case(&stage, case)?;
            let original = run_legacy(python, script, stage.path(), &args)?;
            let legacy_output = stage.read("gh.out")?;
            stage_case(&stage, case)?;
            let ported = xtask(stage.path(), &ported_args)?;
            if !EXPLAINED.contains(&name) {
                assert_same(name, &original, &ported);
            }
            assert_eq!(
                legacy_output,
                stage.read("gh.out")?,
                "{name}: github output parity"
            );
        }
    }
    assert_eq!(cases.len(), 77);
    Ok(())
}

#[test]
fn migration_models_resolver_reuses_frozen_unauthorized_cadence_fixture() -> TestResult {
    // Given: the task-3 negative fixture against the checked-in manifest.
    let root = crate::support::repository_root();
    let case = serde_json::from_slice::<Value>(&std::fs::read(
        root.join("tools/xtask/tests/fixtures/migration/model-negative-unauthorized-cadence.json"),
    )?)?;
    let command = case["command"]
        .as_array()
        .ok_or("command must be an array")?;
    let mut args = vec!["models", "resolve"];
    for arg in &command[2..] {
        args.push(arg.as_str().ok_or("arg must be a string")?);
    }
    // When: the unauthorized cadence resolves.
    let rejected = xtask(&root, &args)?;
    // Then: the legacy child status and diagnostic hold; nothing is emitted.
    let observed = &case["observed"];
    let status = observed["child_exit_status"].as_i64().ok_or("status")?;
    assert_streams(
        "unauthorized",
        &rejected,
        i32::try_from(status)?,
        field(observed, "stdout")?,
        field(observed, "stderr")?,
    );
    // When: the positive control's authorized cadence resolves.
    let control = field(&case["positive_control"], "command")?;
    let mut args = vec!["models", "resolve"];
    args.extend(control.split(' ').skip(2));
    let accepted = xtask(&root, &args)?;
    // Then: the pinned identity is emitted byte-for-byte.
    assert_streams(
        "control",
        &accepted,
        0,
        field(&case["positive_control"], "stdout")?,
        "",
    );
    Ok(())
}
