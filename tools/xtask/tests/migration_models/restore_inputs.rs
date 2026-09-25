//! `models restore-inputs` parity with the resolve step of
//! `.github/actions/restore-test-model/action.yml`: direct URL/file inputs,
//! manifest resolution and the no-model outputs that gate the cache step.

use crate::support::{
    RESOLVER, Stage, TestResult, assert_same, assert_streams, code, field, fixture, legacy,
    repository_root, run,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;

const ACTION: &str = ".github/actions/restore-test-model/action.yml";
/// SHA-256 of the dedented resolve-step block the goldens were captured from.
/// An action edit fails here instead of silently staling the goldens.
const RESOLVE_STEP_SHA256: &str =
    "64784a27b10eb7c45f3360881ee841542dc7b14fbfdfc82c89c4359413892981";
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

/// Runs the captured action block under the operator's bash 5 with the
/// legacy resolver as `python3`.
fn legacy_step(
    stage: &Stage,
    case: &Value,
    bash: &Path,
    python: &Path,
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let bin = stage.path().join("bin");
    if !bin.join("python3").exists() {
        std::fs::create_dir_all(&bin)?;
        std::os::unix::fs::symlink(python, bin.join("python3"))?;
    }
    stage.write("resolve-step.sh", resolve_step()?.as_bytes())?;
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut command = std::process::Command::new(bash);
    command
        .current_dir(stage.path())
        .args(["-e", "resolve-step.sh"])
        .env("PATH", path);
    command.env("GITHUB_OUTPUT", stage.path().join("gh.out"));
    for (env, input) in [
        ("INPUT_MODEL_URL", "model_url"),
        ("INPUT_MODEL_FILE", "model_file"),
        ("MODEL_MANIFEST", "model_manifest"),
        ("MODEL_ARTIFACT_ID", "model_artifact_id"),
        ("MODEL_CADENCE", "model_cadence"),
    ] {
        command.env(env, field(&case["inputs"], input)?);
    }
    Ok(command.stdin(std::process::Stdio::null()).output()?)
}

#[test]
fn migration_models_restore_step_source_is_the_captured_block() -> TestResult {
    // Given/When: the resolve step in the checked-in action.
    let digest = hex::encode(Sha256::digest(resolve_step()?.as_bytes()));
    // Then: it is the exact block every restore golden was captured from.
    assert_eq!(digest, RESOLVE_STEP_SHA256);
    Ok(())
}

#[test]
fn migration_models_restore_outputs_match_the_action_for_every_suite_and_cadence() -> TestResult {
    // Given: frozen manifests and every captured action-step invocation.
    let stage = Stage::new("restore")?;
    stage.frozen_manifests()?;
    let cases = fixture("restore-step.json")?;
    let cases = cases.as_array().ok_or("cases must be an array")?;
    let bash = legacy("MIGRATION_MODELS_LEGACY_BASH");
    let python = legacy("MIGRATION_MODELS_LEGACY_PYTHON");
    if python.is_some() {
        stage.legacy_script(RESOLVER)?;
    }
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
        if let (Some(bash), Some(python)) = (&bash, &python) {
            stage.write("gh.out", b"")?;
            let original = legacy_step(&stage, case, bash, python)?;
            assert_same(name, &original, &ported);
            assert_eq!(stage.read("gh.out")?, outputs, "{name}: output parity");
        }
    }
    assert_eq!(cases.len(), 340);
    assert_eq!(
        resolved, 84,
        "manifest resolutions carrying a verifiable cache key"
    );
    Ok(())
}
