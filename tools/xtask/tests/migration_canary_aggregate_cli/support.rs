use super::fixture_files;

use std::{
    error::Error,
    fs,
    path::PathBuf,
    process::{Command, Output},
};
pub(super) type TestResult = Result<(), Box<dyn Error>>;

pub(super) struct Fixture {
    root: tempfile::TempDir,
    pub(super) identity: String,
}

impl Fixture {
    pub(super) fn new() -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let package = root.path().join("package");
        let evidence = root.path().join("evidence");
        fs::create_dir(&package)?;
        fs::create_dir(&evidence)?;
        fs::write(
            package.join("plan.json"),
            include_bytes!("../migration_canary_receipts/fixtures/plan.json"),
        )?;
        fixture_files::write_artifacts(&package)?;
        let bytes = fixture_files::source_identity(&package)?;
        let identity = fixture_files::hash_bytes(&bytes);
        fs::write(package.join("identity.json"), bytes)?;
        fixture_files::write_worker(
            &evidence,
            "dense",
            "2",
            "success",
            include_bytes!("../migration_canary_receipts/fixtures/dense.jsonl"),
            &identity,
        )?;
        fixture_files::write_worker(
            &evidence,
            "hybrid",
            "2",
            "success",
            include_bytes!("../migration_canary_receipts/fixtures/hybrid.jsonl"),
            &identity,
        )?;
        Ok(Self { root, identity })
    }

    pub(super) fn path(&self, relative: &str) -> PathBuf {
        self.root.path().join(relative)
    }

    pub(super) fn run(&self, extra: &[&str]) -> Result<Output, Box<dyn Error>> {
        Ok(Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "canary-receipts", "aggregate", "--package"])
            .arg(self.path("package"))
            .args(["--identity", &self.identity, "--evidence"])
            .arg(self.path("evidence"))
            .args([
                "--run-id",
                "123",
                "--run-attempt",
                "4",
                "--controller-revision",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ])
            .args(extra)
            .current_dir(self.root.path())
            .env("GITHUB_STEP_SUMMARY", self.path("summary"))
            .env("GITHUB_OUTPUT", self.path("output"))
            .env("GITHUB_RUN_ID", "ambient-wrong-run")
            .env("GITHUB_RUN_ATTEMPT", "1")
            .env("CANARY_CONTROLLER_SHA", "ambient-wrong-controller")
            .env("CANARY_MESH_SOURCE", "ambient-wrong-source")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env_remove("PYTHONPATH")
            .output()?)
    }

    pub(super) fn report(&self) -> &'static str {
        "Canary repair-1: 2/2 family receipts passed for aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n"
    }
}
