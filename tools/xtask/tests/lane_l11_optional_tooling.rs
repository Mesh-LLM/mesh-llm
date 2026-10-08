use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn command(verb: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.args(["automation", "replay-matrix", verb]);
    command
}

fn matrix() -> Result<Value, serde_json::Error> {
    serde_json::from_str(include_str!(
        "../../../ci/agentic-replay-nightly/matrix.json"
    ))
}

fn pin_fixture(root: &Path, matrix: &Value) -> TestResult {
    fs::write(root.join("matrix.json"), serde_json::to_vec(matrix)?)?;
    let replay = &matrix["replay"];
    fs::write(
        root.join("canonical.json"),
        serde_json::to_vec(&json!({"thoughtworks":{"dataset":{
            "repo": replay["dataset"], "revision": replay["dataset_revision"],
            "filename": replay["dataset_file"], "sha256": replay["dataset_sha256"]
        }}}))?,
    )?;
    Ok(())
}

fn pins(root: &Path) -> Command {
    let mut command = command("pins");
    command.current_dir(root).args([
        "--matrix",
        "matrix.json",
        "--canonical",
        "canonical.json",
        "--models-output",
        "models.tsv",
        "--dataset-output",
        "dataset.tsv",
    ]);
    command
}

#[test]
fn pins_project_complete_roster_and_canonical_dataset() -> TestResult {
    let root = tempfile::tempdir()?;
    pin_fixture(root.path(), &matrix()?)?;
    let output = pins(root.path()).output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows = fs::read_to_string(root.path().join("models.tsv"))?;
    assert_eq!(rows.lines().count(), 3);
    assert!(rows.lines().all(|row| row.split('\t').count() == 5));
    assert_eq!(
        fs::read_to_string(root.path().join("dataset.tsv"))?
            .split('\t')
            .count(),
        4
    );
    Ok(())
}

#[test]
fn invalid_pins_fail_before_any_export() -> TestResult {
    for case in [
        "revision",
        "sha256",
        "duplicate",
        "control",
        "drift",
        "empty",
        "short-context",
    ] {
        let root = tempfile::tempdir()?;
        let mut document = matrix()?;
        pin_fixture(root.path(), &document)?;
        match case {
            "revision" => document["models"][0]["revision"] = json!("main"),
            "sha256" => document["models"][0]["sha256"] = json!(true),
            "duplicate" => {
                document["models"][1]["family"] = document["models"][0]["family"].clone()
            }
            "control" => document["models"][0]["file"] = json!("bad\tfile"),
            "drift" => document["replay"]["dataset_file"] = json!("different.parquet"),
            "empty" => document["models"] = json!([]),
            "short-context" => document["replay"]["minimum_context_tokens"] = json!(32768),
            _ => unreachable!(),
        }
        fs::write(
            root.path().join("matrix.json"),
            serde_json::to_vec(&document)?,
        )?;
        assert!(!pins(root.path()).output()?.status.success(), "{case}");
        assert!(!root.path().join("models.tsv").exists(), "{case}");
        assert!(!root.path().join("dataset.tsv").exists(), "{case}");
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn publication(root: &Path) -> TestResult {
    let patch = b"From candidate\nSubject: fix: fixture\n\npatch\n";
    let body = b"verified fixture\n";
    fs::write(root.join("repair.patch"), patch)?;
    fs::write(root.join("pr-body.md"), body)?;
    fs::write(
        root.join("status.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1, "base_sha": "a".repeat(40), "run_id": "123", "run_attempt": 2,
            "resolution": "fix-verified", "patch_bytes": patch.len(), "patch_sha256": digest(patch),
            "body_bytes": body.len(), "body_sha256": digest(body)
        }))?,
    )?;
    Ok(())
}

fn admit(root: &Path) -> Command {
    let mut command = command("publication-verify");
    command.args([
        "--publication-dir",
        root.to_str().unwrap(),
        "--base-sha",
        &"a".repeat(40),
        "--checkout-sha",
        &"a".repeat(40),
        "--run-id",
        "123",
        "--run-attempt",
        "2",
    ]);
    command
}

#[test]
fn publication_emits_bound_run_identity_without_mutation() -> TestResult {
    let root = tempfile::tempdir()?;
    publication(root.path())?;
    let before = fs::read(root.path().join("status.json"))?;
    let output = admit(root.path()).output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"123\t2\tfix-verified\n");
    assert_eq!(fs::read(root.path().join("status.json"))?, before);
    Ok(())
}

#[test]
fn publication_rejects_foreign_identity_schema_and_tampering() -> TestResult {
    for case in [
        "run",
        "attempt",
        "base",
        "bool",
        "unknown",
        "resolution",
        "patch",
        "body",
        "extra",
        "empty",
    ] {
        let root = tempfile::tempdir()?;
        publication(root.path())?;
        let status_path = root.path().join("status.json");
        let mut status: Value = serde_json::from_slice(&fs::read(&status_path)?)?;
        match case {
            "run" => status["run_id"] = json!("456"),
            "attempt" => status["run_attempt"] = json!(3),
            "base" => status["base_sha"] = json!("b".repeat(40)),
            "bool" => status["body_bytes"] = json!(true),
            "unknown" => status["extra"] = json!(1),
            "resolution" => status["resolution"] = json!("failed"),
            "patch" => fs::write(root.path().join("repair.patch"), "tampered")?,
            "body" => fs::write(root.path().join("pr-body.md"), "tampered")?,
            "extra" => fs::write(root.path().join("executable"), "extra")?,
            "empty" => fs::write(root.path().join("repair.patch"), "")?,
            _ => unreachable!(),
        }
        fs::write(status_path, serde_json::to_vec(&status)?)?;
        let output = admit(root.path()).output()?;
        assert!(!output.status.success(), "{case}");
        assert!(output.stdout.is_empty(), "{case}");
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn publication_rejects_symlinked_data() -> TestResult {
    let root = tempfile::tempdir()?;
    let target = tempfile::NamedTempFile::new()?;
    publication(root.path())?;
    fs::remove_file(root.path().join("pr-body.md"))?;
    std::os::unix::fs::symlink(target.path(), root.path().join("pr-body.md"))?;
    assert!(!admit(root.path()).output()?.status.success());
    Ok(())
}

#[test]
fn digest_verification_rejects_corrupt_bytes_and_malformed_pin() -> TestResult {
    let root = tempfile::tempdir()?;
    let file = root.path().join("model with spaces.gguf");
    fs::write(&file, b"model")?;
    for (expected, success) in [
        (digest(b"model"), true),
        (digest(b"wrong"), false),
        ("invalid".into(), false),
    ] {
        let output = command("verify-digest")
            .args(["--file", file.to_str().unwrap(), "--sha256", &expected])
            .output()?;
        assert_eq!(output.status.success(), success);
    }
    Ok(())
}

#[test]
fn workflow_preserves_offline_cutover_and_repair_admission() {
    let workflow = include_str!("../../../.github/workflows/agentic-replay-nightly.yml");
    assert!(!workflow.contains("scripts/agentic-replay-params.py"));
    assert!(!workflow.contains("  schedule:"));
    assert!(workflow.contains("steps.history.outputs.repair_required == 'true'"));
    assert!(workflow.contains("steps.baseline.outcome == 'success'"));
    assert!(workflow.contains("needs.replay.result == 'failure'"));
    assert!(workflow.contains("needs.replay.outputs.repair_prepared == 'true'"));
    assert!(workflow.contains("cargo xtool automation replay-matrix publication-verify"));
    assert!(workflow.contains("git -c core.hooksPath=/dev/null am --no-verify"));
    assert!(workflow.contains("git diff --quiet HEAD^ HEAD -- .github .agents scripts evals ci"));
    let repair = include_str!("../../../scripts/agentic-replay-repair.sh");
    assert!(repair.contains("--canonical skippy/evals/skippy-competitive-benchmark.json"));
    assert!(repair.contains("--reader \"${REPLAY_READER:?native replay reader required}\""));
    assert!(!repair.contains("--python"));
    assert!(!repair.contains("--canonical evals/skippy-competitive-benchmark.json"));
    let canonical = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../skippy/evals/skippy-competitive-benchmark.json");
    let canonical: serde_json::Value =
        serde_json::from_slice(&std::fs::read(canonical).unwrap()).unwrap();
    assert!(!canonical["models"].as_array().unwrap().is_empty());
    assert!(repair.contains("unset CANARY_REPAIR_TOKEN GH_TOKEN GITHUB_TOKEN HF_TOKEN"));
    assert!(!repair.contains("git push"));
    assert!(!repair.contains("gh pr create"));
    assert!(repair.contains("run_untrusted cargo xtool automation replay-matrix run-family"));
    assert!(repair.contains("if [[ \"$RESOLVED\" != \"1\" ]]"));
}
