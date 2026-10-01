use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

type TestResult = Result<(), Box<dyn Error>>;

struct Fixture {
    _directory: tempfile::TempDir,
    evidence: PathBuf,
    comparison_log: PathBuf,
    model: PathBuf,
    candidate: PathBuf,
    oracle: PathBuf,
}

impl Fixture {
    fn new() -> Result<Self, Box<dyn Error>> {
        let directory = tempfile::tempdir()?;
        let evidence = directory.path().join("evidence.json");
        let comparison_log = directory.path().join("comparison.txt");
        let model = directory.path().join("model.gguf");
        let candidate = directory.path().join("skippy-server");
        let oracle = directory.path().join("llama-server");
        fs::write(
            &comparison_log,
            "embedding local-monolithic oracle passed: fixture\n",
        )?;
        fs::write(&model, b"abc")?;
        fs::write(&candidate, b"")?;
        fs::write(&oracle, b"hello")?;
        Ok(Self {
            _directory: directory,
            evidence,
            comparison_log,
            model,
            candidate,
            oracle,
        })
    }

    fn write_args(&self) -> Result<Vec<String>, Box<dyn Error>> {
        Ok(vec![
            "--repo-root".into(),
            path_arg(&workspace_root())?,
            "automation".into(),
            "workload-oracle-evidence".into(),
            "write".into(),
            "--output".into(),
            path_arg(&self.evidence)?,
            "--comparison-log".into(),
            path_arg(&self.comparison_log)?,
            "--class".into(),
            "embedding".into(),
            "--smoke-lane".into(),
            "embedding-smoke".into(),
            "--model-id".into(),
            "fixture".into(),
            "--model-sha256".into(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            "--candidate-executable".into(),
            path_arg(&self.candidate)?,
            "--oracle-executable".into(),
            path_arg(&self.oracle)?,
            "--pinned-patch-sha".into(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            "--work-dir".into(),
            path_arg(self.evidence.parent().ok_or("evidence parent")?)?,
        ])
    }

    fn verify_args(&self, model_class: &str) -> Result<Vec<String>, Box<dyn Error>> {
        Ok(vec![
            "--repo-root".into(),
            path_arg(&workspace_root())?,
            "automation".into(),
            "workload-oracle-evidence".into(),
            "verify".into(),
            "--evidence".into(),
            path_arg(&self.evidence)?,
            "--class".into(),
            model_class.into(),
            "--smoke-lane".into(),
            "embedding-smoke".into(),
            "--oracle-lane".into(),
            "embedding-oracle".into(),
            "--model-id".into(),
            "fixture".into(),
            "--model-path".into(),
            path_arg(&self.model)?,
            "--candidate-executable".into(),
            path_arg(&self.candidate)?,
            "--oracle-executable".into(),
            path_arg(&self.oracle)?,
            "--pinned-patch-sha".into(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
        ])
    }

    fn run(&self, args: &[String]) -> Result<Output, Box<dyn Error>> {
        Ok(Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(workspace_root())
            .args(args)
            .output()?)
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask lives under tools/")
        .to_path_buf()
}

fn path_arg(path: &Path) -> Result<String, Box<dyn Error>> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| "non-UTF8 fixture path".into())
}

#[test]
fn workload_evidence_cli_writes_and_verifies_python_compatible_bytes() -> TestResult {
    // Given: local artifact bytes and an explicit class-specific comparator pass.
    let fixture = Fixture::new()?;

    // When: write then verify evidence through the xtask executable.
    let written = fixture.run(&fixture.write_args()?)?;
    let verified = fixture.run(&fixture.verify_args("embedding")?)?;

    // Then: the writer emits the source-derived payload and verify uses its legacy success line.
    assert!(
        written.status.success(),
        "{}",
        String::from_utf8_lossy(&written.stderr)
    );
    assert!(written.stdout.is_empty());
    assert!(written.stderr.is_empty());
    assert_eq!(
        fs::read(&fixture.evidence)?,
        include_bytes!("migration_workload_oracle_evidence/embedding-evidence.json")
    );
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    assert_eq!(
        String::from_utf8(verified.stdout)?,
        "verified embedding local-monolithic oracle evidence\n"
    );
    assert!(verified.stderr.is_empty());
    Ok(())
}

#[test]
fn workload_evidence_cli_rejection_preserves_existing_output() -> TestResult {
    // Given: valid write inputs except for a smoke-only final comparison line and an existing output.
    let fixture = Fixture::new()?;
    fs::write(
        &fixture.comparison_log,
        "embedding OpenAI HTTP smoke passed\n",
    )?;
    fs::write(&fixture.evidence, b"retained")?;

    // When: the evidence writer runs on the invalid comparison.
    let output = fixture.run(&fixture.write_args()?)?;

    // Then: it reports the Python-compatible failure and does not truncate the artifact.
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .starts_with("workload oracle evidence not written: oracle comparator did not emit")
    );
    assert_eq!(fs::read(&fixture.evidence)?, b"retained");
    Ok(())
}

#[test]
fn workload_evidence_cli_checks_projector_before_evidence_or_artifacts() -> TestResult {
    // Given: a projector-required class, malformed evidence, and a missing model.
    let fixture = Fixture::new()?;
    fs::write(&fixture.evidence, b"{")?;
    fs::remove_file(&fixture.model)?;

    // When: verification runs without its required projector path.
    let output = fixture.run(&fixture.verify_args("ocr")?)?;

    // Then: the first legacy rejection wins before JSON parsing or artifact hashing.
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).starts_with(
        "workload oracle evidence rejected: ocr oracle evidence requires a projector path"
    ));
    Ok(())
}

#[test]
fn workload_evidence_cli_invalid_verifier_class_is_argument_error() -> TestResult {
    // Given: otherwise complete verifier arguments with an unsupported class.
    let fixture = Fixture::new()?;
    let args = fixture.verify_args("unknown")?;

    // When: the verifier command parses them.
    let output = fixture.run(&args)?;

    // Then: the closed class set is rejected as a usage error before evidence IO.
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid choice: 'unknown'"));
    Ok(())
}
