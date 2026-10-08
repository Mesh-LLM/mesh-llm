//! `models restore-inputs` parity with the resolve step of
//! `.github/actions/restore-test-model/action.yml`: direct URL/file inputs,
//! manifest resolution and the no-model outputs that gate the cache step.

use crate::support::{
    Stage, TestResult, assert_streams, code, field, fixture, repository_root, run,
};
use serde_json::Value;
use std::path::Path;

const ACTION: &str = ".github/actions/restore-test-model/action.yml";
const INPUTS: [(&str, &str); 5] = [
    ("--model-url", "model_url"),
    ("--model-file", "model_file"),
    ("--model-manifest", "model_manifest"),
    ("--model-artifact-id", "model_artifact_id"),
    ("--model-cadence", "model_cadence"),
];

/// The `run: |` body of the `resolve-model` step, dedented, as captured.
fn resolve_step() -> Result<String, Box<dyn std::error::Error>> {
    let action = std::fs::read_to_string(repository_root().join(ACTION))?;
    let mut lines = action
        .lines()
        .skip_while(|line| !line.contains("id: resolve-model"));
    let mut lines = lines
        .by_ref()
        .skip_while(|line| line.trim() != "run: |")
        .skip(1);
    let mut block = String::new();
    for line in lines
        .by_ref()
        .take_while(|line| !line.starts_with("    - name:"))
    {
        block.push_str(line.get(8..).unwrap_or(""));
        block.push('\n');
    }
    Ok(block)
}

fn ported_args(case: &Value) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut args = vec!["models".to_owned(), "restore-inputs".to_owned()];
    for (flag, input) in INPUTS {
        args.extend([format!("{flag}={}", field(&case["inputs"], input)?)]);
    }
    args.extend(["--github-output".to_owned(), "gh.out".to_owned()]);
    Ok(args)
}

#[test]
fn migration_models_restore_action_uses_typed_resolution() -> TestResult {
    let block = resolve_step()?;
    assert!(block.contains("models restore-inputs"));
    for (flag, _) in INPUTS {
        assert!(block.contains(flag), "missing action input {flag}");
    }
    assert!(block.contains("--github-output \"$GITHUB_OUTPUT\""));
    assert!(!block.contains("python"));
    Ok(())
}

#[test]
fn migration_models_restore_outputs_match_the_action_for_every_suite_and_cadence() -> TestResult {
    // Given: frozen manifests and every captured action-step invocation.
    let stage = Stage::new("restore")?;
    stage.frozen_manifests()?;
    let cases = fixture("restore-step.json")?;
    let cases = cases.as_array().ok_or("cases must be an array")?;
    let mut resolved = 0;
    for case in cases {
        let name = field(case, "name")?;
        let args = ported_args(case)?;
        let args = args.iter().map(String::as_str).collect::<Vec<_>>();
        stage.write("gh.out", b"")?;
        // When: the ported step runs with the same inputs.
        let ported = run(Path::new(env!("CARGO_BIN_EXE_xtask")), stage.path(), &args)?;
        // Then: status, streams and every step output equal the action's.
        assert_streams(
            name,
            &ported,
            code(case)?,
            field(case, "stdout")?,
            field(case, "stderr")?,
        );
        let outputs = stage.read("gh.out")?;
        assert_eq!(outputs, field(case, "github_output")?, "{name}: outputs");
        if code(case)? == 0 && outputs.contains("sha256=") && !outputs.contains("sha256=\n") {
            resolved += 1;
        }
    }
    assert_eq!(cases.len(), 340);
    assert_eq!(
        resolved, 84,
        "manifest resolutions carrying a verifiable cache key"
    );
    Ok(())
}
