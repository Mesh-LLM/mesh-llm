use super::rollout::{self, Issue};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) struct Fixture {
    pub(super) root: PathBuf,
    pub(super) repo: PathBuf,
    pub(super) input: PathBuf,
    pub(super) catalog: String,
    pub(super) protected: String,
    pub(super) source: String,
}

impl Fixture {
    pub(super) fn new() -> Self {
        let root = crate::command::unique_temp_dir("migration-rollout");
        let repo = root.join("repository");
        fs::create_dir_all(&repo).unwrap();
        let mut fixture = Self {
            input: root.join("observations.json"),
            root,
            repo,
            catalog: String::new(),
            protected: String::new(),
            source: String::new(),
        };
        fixture.git(&["init", "-q"]);
        for path in [
            "ci/ownership.yml",
            "ci/slices.yml",
            "ci/runner-images.json",
            ".github/actions/select-ci-runners/action.yml",
        ] {
            fixture.copy_repository_file(path);
        }
        fixture.catalog = fixture.commit("catalog baseline");
        for path in [
            ".github/actions/prepare-automation/action.yml",
            "just/ci.just",
            "tools/xtask/Cargo.toml",
            "tools/xtask/src/automation_bootstrap.rs",
            "tools/xtask/src/ci_plan/mod.rs",
        ] {
            fixture.copy_repository_file(path);
        }
        fixture.copy_fixture_directory("cases");
        fixture.copy_fixture_directory("expected");
        fixture.protected = fixture.commit("protected support");
        fixture.write_repo("candidate.txt", b"inert candidate change\n");
        fixture.source = fixture.commit("dependent source");
        fixture.write_evidence();
        fixture
    }

    pub(super) fn git(&self, args: &[&str]) -> String {
        let result = Command::new("git")
            .current_dir(&self.repo)
            .env("GIT_MASTER", "1")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env(
                "GIT_CONFIG_GLOBAL",
                if cfg!(windows) { "NUL" } else { "/dev/null" },
            )
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .args([
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=rollout-fixture",
                "-c",
                "user.email=rollout@example.invalid",
            ])
            .args(args)
            .output()
            .expect("fixture git");
        assert!(
            result.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).unwrap().trim().to_owned()
    }

    pub(super) fn commit(&self, message: &str) -> String {
        self.git(&["add", "--all"]);
        self.git(&["commit", "-q", "-m", message]);
        self.git(&["rev-parse", "HEAD"])
    }

    pub(super) fn write_repo(&self, relative: &str, bytes: &[u8]) {
        let path = self.repo.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    fn copy_repository_file(&self, relative: &str) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        self.write_repo(relative, &fs::read(root.join(relative)).unwrap());
    }

    fn copy_fixture_directory(&self, name: &str) {
        let directory = format!("tools/xtask/tests/fixtures/ci_plan/{name}");
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        for entry in fs::read_dir(root.join(&directory)).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                self.copy_repository_file(&format!(
                    "{directory}/{}",
                    entry.file_name().to_str().unwrap()
                ));
            }
        }
    }

    fn write_evidence(&self) {
        let target = self.root.join("target");
        fs::create_dir_all(target.join("debug")).unwrap();
        let target = target.canonicalize().unwrap();
        let binary = target.join(if cfg!(windows) {
            "debug/xtask.exe"
        } else {
            "debug/xtask"
        });
        fs::write(&binary, b"inert fixture bytes; never executed").unwrap();
        let report = format!(
            "binary_path={}\ntarget_directory={}\nhost=fixture-host\n",
            binary.display(),
            target.display()
        );
        fs::write(self.root.join("bootstrap.txt"), report).unwrap();
        let predecessors: Vec<_> = (11..=25)
            .map(|task| {
                let file = format!("task-{task}.txt");
                fs::write(
                    self.root.join(&file),
                    format!("Task {task}: availability fixture, not acceptance\n"),
                )
                .unwrap();
                json!({"task": task, "evidence": self.binding(&file)})
            })
            .collect();
        let request = json!({
            "repository": "repository", "catalog_sha": self.catalog,
            "support_sha": self.protected, "protected_sha": self.protected,
            "source_sha": self.source, "planner_sha": self.protected,
            "workspace_sha": self.protected, "runner_policy_sha": self.protected,
            "bootstrap": {"source_sha": self.protected, "report": self.binding("bootstrap.txt"),
                "binary": self.binding(binary.to_str().unwrap())},
            "predecessors": predecessors
        });
        fs::write(&self.input, serde_json::to_vec(&request).unwrap()).unwrap();
    }

    pub(super) fn binding(&self, path: &str) -> Value {
        let bytes = fs::read(self.root.join(path)).unwrap();
        json!({"path": path, "sha256": hex::encode(Sha256::digest(bytes))})
    }

    pub(super) fn change(&self, edit: impl FnOnce(&mut Value)) {
        let mut request: Value = serde_json::from_slice(&fs::read(&self.input).unwrap()).unwrap();
        edit(&mut request);
        fs::write(&self.input, serde_json::to_vec(&request).unwrap()).unwrap();
    }

    pub(super) fn rejected(&self, issue: Issue) {
        let failure = rollout::validate_file(&self.input).unwrap_err();
        assert_eq!(failure.issue, issue, "{failure}");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            eprintln!("rollout fixture cleanup: {error}");
        }
    }
}
