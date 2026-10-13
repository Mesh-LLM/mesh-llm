//! Actual maintained repair guard with finite local Git fixtures.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

fn executable(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|p| p.join(name))
        .find(|p| p.is_file())
        .unwrap_or_else(|| panic!("required {name} missing"))
}
fn run(root: &Path, command: &str, arguments: &[&str]) -> (i32, String, String) {
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: executable(command),
            cwd: root.to_owned(),
            arguments: arguments
                .iter()
                .map(|v| Value::Public((*v).into()))
                .collect(),
            environment: BTreeMap::from([
                (
                    "PATH".into(),
                    Value::Public(std::env::var_os("PATH").unwrap()),
                ),
                (
                    "HOME".into(),
                    Value::Public(root.join("home").into_os_string()),
                ),
                ("GIT_MASTER".into(), Value::Public("1".into())),
                ("GIT_CONFIG_NOSYSTEM".into(), Value::Public("1".into())),
                (
                    "GIT_CONFIG_GLOBAL".into(),
                    Value::Public("/dev/null".into()),
                ),
            ]),
        },
        &Limits {
            execution: Duration::from_secs(15),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(
        result.process.cleanup.complete && result.process.failure.is_none(),
        "{:?}",
        result.process
    );
    (
        result.process.status.unwrap().code().unwrap(),
        String::from_utf8(result.stdout.unwrap().as_bytes().to_vec()).unwrap(),
        String::from_utf8(result.stderr.unwrap().as_bytes().to_vec()).unwrap(),
    )
}
struct Fixture {
    temp: tempfile::TempDir,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir(temp.path().join("home")).unwrap();
        let fixture = Self { temp };
        fixture.git(&["init", "-b", "main"]);
        fixture.git(&["config", "user.name", "Fixture"]);
        fixture.git(&["config", "user.email", "fixture@example.invalid"]);
        for name in [
            ".github/workflow.yml",
            ".agents/rules.md",
            "scripts/guard.sh",
            ".gitattributes",
            "ci/ci.md",
            "ci/llama-canary/agent-repair-prompt.md",
        ] {
            fixture.write(name, "protected baseline\n");
        }
        fixture.git(&["add", "-A"]);
        fixture.git(&["commit", "-m", "baseline"]);
        let source = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../scripts/llama-canary-agent-repair.sh"),
        )
        .unwrap();
        let start = source
            .find("\nassert_agent_control_unchanged() {\n")
            .unwrap()
            + 1;
        let tail = &source[start..];
        let end = tail.find("\nrepair_source_inspection() {\n").unwrap();
        let guard = &tail[..end];
        fs::write(fixture.temp.path().join("actual-guard.sh"), format!(
            "#!/usr/bin/env bash\nset -euo pipefail\nsource snapshot.env\n{guard}\nassert_agent_control_unchanged\necho ACTUAL_GUARD_ACCEPTED\n"
        )).unwrap();
        let snapshot = r#"printf 'BASE_HEAD=%q\nBASE_REF=%q\nGIT_CONFIG_FINGERPRINT=%q\n' "$(git rev-parse HEAD)" "$(git symbolic-ref -q HEAD)" "$(git config --list --show-origin | shasum -a 256 | awk '{print $1}')" > snapshot.env"#;
        let (status, _, errors) = run(fixture.temp.path(), "bash", &["-c", snapshot]);
        assert_eq!(status, 0, "{errors}");
        fixture
    }
    fn git(&self, args: &[&str]) {
        let (status, _, errors) = run(self.temp.path(), "git", args);
        assert_eq!(status, 0, "{args:?}: {errors}");
    }
    fn write(&self, name: &str, text: &str) {
        let path = self.temp.path().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    fn assert_guard(&self, accepted: bool, diagnostic: &str) {
        let (status, output, errors) = run(self.temp.path(), "bash", &["actual-guard.sh"]);
        assert_eq!(status == 0, accepted, "{output}{errors}");
        assert_eq!(
            output.contains("ACTUAL_GUARD_ACCEPTED"),
            accepted,
            "{output}{errors}"
        );
        if !accepted {
            assert!(errors.contains(diagnostic), "{errors}");
        }
    }
}
#[test]
fn actual_agent_guard_accepts_unchanged_controls() {
    Fixture::new().assert_guard(true, "");
}
#[test]
fn actual_agent_guard_leaves_manifest_changes_for_typed_policy_admission() {
    let fixture = Fixture::new();
    fixture.write("ci/llama-canary/family-certified.json", "{}\n");
    fixture.write("skippy/docs/llama-parity-candidates.json", "{}\n");
    fixture.assert_guard(true, "");
}
#[test]
fn actual_agent_guard_rejects_commits_and_branch_switches() {
    let committed = Fixture::new();
    committed.git(&["commit", "--allow-empty", "-m", "agent commit"]);
    committed.assert_guard(false, "agent created commits");
    let switched = Fixture::new();
    switched.git(&["checkout", "-b", "agent-branch"]);
    switched.assert_guard(false, "agent switched branches");
}
#[test]
fn actual_agent_guard_rejects_git_configuration_changes() {
    let fixture = Fixture::new();
    fixture.git(&["config", "alias.agent-change", "status"]);
    fixture.assert_guard(false, "agent changed Git configuration");
}
#[test]
fn actual_agent_guard_rejects_each_tracked_protected_owner() {
    for name in [
        ".github/workflow.yml",
        ".agents/rules.md",
        "scripts/guard.sh",
        ".gitattributes",
        "ci/ci.md",
        "ci/llama-canary/agent-repair-prompt.md",
    ] {
        let fixture = Fixture::new();
        fixture.write(name, "agent change\n");
        fixture.assert_guard(false, "agent modified protected CI or verification file");
    }
}
#[test]
fn actual_agent_guard_rejects_untracked_startup_hooks_and_protected_files() {
    for name in [
        "scripts/sitecustomize.py",
        ".github/new-workflow.yml",
        ".agents/new-policy.md",
    ] {
        let fixture = Fixture::new();
        fixture.write(name, "untracked fixture\n");
        fixture.assert_guard(false, "agent modified protected CI or verification file");
    }
}
