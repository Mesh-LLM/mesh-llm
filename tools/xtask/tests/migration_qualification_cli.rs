use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Fixture {
    directory: tempfile::TempDir,
    selected: PathBuf,
    invocation: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let selected = repository(directory.path(), "selected", &"a".repeat(40));
        let invocation = repository(directory.path(), "invocation", &"b".repeat(40));
        Self {
            directory,
            selected,
            invocation,
        }
    }

    fn receipt(&self, root: &Path, sha: &str) -> PathBuf {
        let artifact = write_artifact(&self.directory.path().join("input"), b"fake evidence only");
        let contracts = write_artifact(&root.join("ci/automation-migration/contracts.json"), br#"{"qualification":{"schema_version":1,"required_roots":["build","test-all"],"required_backends":{"macos":["metal"]}}}"#);
        let execution = json!({"argv":["must-not-execute"],"exit_code":0,"case_count":2,"cleanup_complete":true,"evidence":artifact});
        let scenarios: Vec<_> = ["product-readiness", "protocol-pair", "corrupt-runtime", "readiness-timeout"].into_iter().map(|scenario| {
            let failure = match scenario {
                "corrupt-runtime" => Some("digest_mismatch"),
                "readiness-timeout" => Some("readiness_timeout"),
                _ => None,
            };
            let mut row = execution.clone();
            row["exit_code"] = json!(i32::from(failure.is_some()));
            json!({"scenario":scenario,"execution":row,"expected_failure_observed":failure.is_some(),"failure_kind":failure})
        }).collect();
        let models = ["smollm2-q8-inference", "family-granite-hybrid"];
        let receipt = json!({
            "schema_version":1,"platform":"macos","source_sha":sha,"source_snapshot":artifact,"contracts":contracts,
            "interpreters":{"path":"/fake/tools","attempts":0,"probes":(["path","absolute","shebang","versioned"].map(|kind| json!({"kind":kind,"candidates":["python3"],"found":[],"evidence":artifact})))},
            "roots":{"build":execution,"test-all":execution},
            "products":[{"backend":"metal","hardware_evidence":artifact,"host_manifest":artifact,"runtime_manifest":artifact,"product_manifest":artifact,"files":[artifact]}],
            "models":models.map(|id| json!({"artifact_id":id,"revision":"c".repeat(40),"files":[artifact]})),
            "protocol_cases":models.map(|id| json!({"backend":"metal","model_id":id,"execution":execution})),
            "scenarios":scenarios
        });
        let path = self.invocation.join("receipt.json");
        std::fs::write(&path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        path
    }

    fn replay(&self, receipt: &Path) -> Output {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(&self.invocation)
            .env("GIT_DIR", self.invocation.join(".git"))
            .args([
                "--repo-root",
                self.selected.to_str().unwrap(),
                "automation",
                "qualify-replay",
                "--receipt",
                receipt.to_str().unwrap(),
                "--scenario",
                "product-readiness",
                "--evidence",
                "not-created",
            ])
            .output()
            .unwrap()
    }
}

fn repository(parent: &Path, name: &str, sha: &str) -> PathBuf {
    let root = parent.join(name);
    std::fs::create_dir_all(root.join("tools/xtask")).unwrap();
    std::fs::create_dir_all(root.join("ci/automation-migration")).unwrap();
    std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    std::fs::write(root.join("tools/xtask/Cargo.toml"), "[package]\n").unwrap();
    assert!(
        Command::new("git")
            .args(["init", "--quiet"])
            .arg(&root)
            .status()
            .unwrap()
            .success()
    );
    std::fs::write(root.join(".git/HEAD"), format!("{sha}\n")).unwrap();
    root.canonicalize().unwrap()
}

fn write_artifact(path: &Path, bytes: &[u8]) -> Value {
    use sha2::{Digest, Sha256};
    std::fs::write(path, bytes).unwrap();
    json!({"path":path,"sha256":hex::encode(Sha256::digest(bytes))})
}

#[test]
#[cfg(target_os = "macos")]
fn replay_is_pending_when_selected_root_differs_from_invocation_and_git_environment() {
    let fixture = Fixture::new();
    let receipt = fixture.receipt(&fixture.selected, &"a".repeat(40));
    let output = fixture.replay(Path::new("receipt.json"));
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("qualification execution pending"),
        "{output:?}; receipt={receipt:?}"
    );
    assert!(!fixture.invocation.join("not-created").exists());
    assert!(!fixture.selected.join("not-created").exists());
}

#[test]
fn replay_rejects_invocation_source_when_selected_root_differs() {
    let fixture = Fixture::new();
    let receipt = fixture.receipt(&fixture.invocation, &"b".repeat(40));
    let output = fixture.replay(&receipt);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("candidate source SHA mismatch"),
        "{output:?}"
    );
}

#[test]
fn replay_rejects_invocation_contracts_when_source_matches_selected_root() {
    let fixture = Fixture::new();
    let receipt = fixture.receipt(&fixture.invocation, &"a".repeat(40));
    write_artifact(
        &fixture
            .selected
            .join("ci/automation-migration/contracts.json"),
        b"different selected contracts",
    );
    let output = fixture.replay(&receipt);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("receipt does not bind candidate frozen contracts"),
        "{output:?}"
    );
}

#[test]
fn help_lists_qualification_when_top_level_or_command_help_requested() {
    for args in [
        vec!["--help"],
        vec!["automation", "qualify", "--help"],
        vec!["automation", "qualify-replay", "--help"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("automation qualify --platform"));
        assert!(stdout.contains("automation qualify-replay --receipt"));
    }
}

#[test]
fn qualification_rejects_missing_arguments_without_creating_evidence() {
    let fixture = Fixture::new();
    for args in [
        vec!["automation", "qualify"],
        vec!["automation", "qualify", "--platform"],
        vec!["automation", "qualify-replay", "--evidence", "not-created"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(&fixture.invocation)
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&output.stderr).contains("qualification receipt rejected"));
        assert!(!fixture.invocation.join("not-created").exists());
    }
}
