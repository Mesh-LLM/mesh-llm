//! Real cleanup CLI boundary over exclusively registered finite temporary owners.
use crate::cleanup_owner as ownership;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};
struct Fixture {
    directory: tempfile::TempDir,
    workspace: PathBuf,
    temporary: PathBuf,
    environment: BTreeMap<OsString, OsString>,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        ownership::register(&directory);
        let root = directory.path().canonicalize().unwrap();
        let workspace = root.join("workspace");
        let temporary = root.join("temporary");
        for path in [&workspace, &temporary] {
            ownership::check(&directory, path).unwrap();
            fs::create_dir(path).unwrap();
        }
        let mut environment: BTreeMap<OsString, OsString> = [
            ("PATH", "/usr/bin:/bin"),
            ("GIT_MASTER", "1"),
            ("GIT_CONFIG_NOSYSTEM", "1"),
            ("GIT_CONFIG_GLOBAL", "/dev/null"),
            ("GIT_TERMINAL_PROMPT", "0"),
            ("GITHUB_RUN_ID", "123"),
            ("GITHUB_RUN_ATTEMPT", "2"),
            ("CANARY_PASS_ID", "repair-1"),
            ("CANARY_SHARD_INDEX", "3"),
            ("CLEANUP_ARTIFACT_PATH", "ci-artifacts/linux"),
            ("CLEANUP_BINARY_PATH", "target/release/mesh-llm"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
        for (key, path) in [
            ("GITHUB_WORKSPACE", &workspace),
            ("CANARY_SOURCE_ROOT", &workspace),
            ("RUNNER_TEMP", &temporary),
            ("HOME", &root),
        ] {
            environment.insert(key.into(), path.as_os_str().to_owned());
        }
        Self {
            directory,
            workspace,
            temporary,
            environment,
        }
    }
    fn seed(&self, path: &Path) {
        ownership::check(&self.directory, path).unwrap();
        fs::create_dir_all(path).unwrap();
        fs::write(path.join("payload"), b"retain").unwrap();
    }
    fn execute(&self, executable: PathBuf, args: Vec<OsString>) -> process::RawProcessReport {
        ownership::check(&self.directory, &self.workspace).unwrap();
        ownership::check(&self.directory, &self.temporary).unwrap();
        let report = process::supervise_raw(
            &ProcessSpec {
                executable,
                cwd: self.workspace.clone(),
                environment: self
                    .environment
                    .iter()
                    .map(|(key, value)| (key.clone(), Value::Public(value.clone())))
                    .collect(),
                arguments: args.into_iter().map(Value::Public).collect(),
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
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
        assert!(report.process.failure.is_none(), "{:?}", report.process);
        assert!(report.process.cleanup.complete);
        report
    }
    fn cleanup(
        &self,
        profile: &str,
        uploaded: bool,
        git: Option<&Path>,
    ) -> process::RawProcessReport {
        let mut args = [
            "ci-ops",
            "runner-cleanup",
            "--evidence-uploaded",
            if uploaded { "true" } else { "false" },
            "--job",
            profile,
            "--package-uploaded",
            if uploaded { "true" } else { "false" },
        ]
        .map(OsString::from)
        .to_vec();
        if let Some(git) = git {
            assert!(git.is_absolute());
            args.extend(["--git".into(), git.as_os_str().to_owned()]);
        }
        self.execute(env!("CARGO_BIN_EXE_xtask").into(), args)
    }
    fn git(&self, args: &[&str]) -> Vec<u8> {
        let executable = PathBuf::from(
            std::env::var_os("MIGRATION_TEST_GIT")
                .expect("set MIGRATION_TEST_GIT to an absolute Git executable"),
        );
        assert!(executable.is_absolute());
        let report = self.execute(
            executable,
            args.iter().map(|arg| OsString::from(*arg)).collect(),
        );
        assert!(report.process.success(), "{:?}", report.process);
        report.stdout.unwrap().as_bytes().to_vec()
    }
    fn repository(&self) -> PathBuf {
        self.git(&["init", "--template="]);
        self.git(&[
            "-c",
            "user.name=Cleanup Fixture",
            "-c",
            "user.email=cleanup@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=disabled-hooks",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
        ]);
        PathBuf::from(std::env::var_os("MIGRATION_TEST_GIT").unwrap())
    }
    fn add(&self, path: &Path) {
        ownership::check(&self.directory, path).unwrap();
        self.git(&[
            "worktree",
            "add",
            "--detach",
            path.to_str().unwrap(),
            "HEAD",
        ]);
    }
    fn registered(&self, path: &Path) -> bool {
        use std::os::unix::ffi::OsStrExt;
        self.git(&["worktree", "list", "--porcelain", "-z"])
            .split(|b| *b == 0)
            .any(|field| field.strip_prefix(b"worktree ") == Some(path.as_os_str().as_bytes()))
    }
}
#[test]
fn actual_cleanup_build_preserves_recovery_until_both_uploads_then_is_repeatable() {
    let f = Fixture::new();
    let disposable = f.workspace.join("target/debug");
    let state = f.workspace.join(".deps/llama-canary-state-123-2-repair-1");
    let package = f.temporary.join("canary-export-123-2-repair-1");
    let unrelated = f.workspace.join("source");
    for p in [&disposable, &state, &package, &unrelated] {
        f.seed(p);
    }
    assert!(f.cleanup("build", false, None).process.success());
    assert!(!disposable.exists());
    for p in [&state, &package, &unrelated] {
        assert!(p.join("payload").exists());
    }
    for _ in 0..2 {
        assert!(f.cleanup("build", true, None).process.success());
        assert!(!state.exists() && !package.exists());
        assert!(unrelated.join("payload").exists());
    }
}
#[test]
fn actual_cleanup_family_removes_selected_outputs_and_retains_controller_build() {
    let mut f = Fixture::new();
    let selected = f.workspace.join("canary-source");
    let output = selected.join("target/debug");
    let controller = f.workspace.join("target/debug");
    let evidence = f.workspace.join("target/canary-evidence-123-2/repair-1-3");
    for p in [&output, &controller, &evidence] {
        f.seed(p);
    }
    f.environment
        .insert("CANARY_SOURCE_ROOT".into(), selected.into_os_string());
    assert!(f.cleanup("family", false, None).process.success());
    assert!(!output.exists());
    assert!(controller.join("payload").exists() && evidence.join("payload").exists());
}
#[test]
fn actual_cleanup_smoke_rejects_parent_symlink_and_shallow_binary_before_any_deletion() {
    let mut f = Fixture::new();
    let artifact = f.workspace.join("ci-artifacts/linux");
    let outside = f.directory.path().join("outside");
    f.seed(&artifact);
    f.seed(&outside.join("release"));
    std::os::unix::fs::symlink(&outside, f.workspace.join("target")).unwrap();
    let report = f.cleanup("smoke", true, None);
    assert!(!report.process.success());
    assert!(report.stdout.unwrap().as_bytes().is_empty());
    assert!(artifact.join("payload").exists() && outside.join("release/payload").exists());
    fs::remove_file(f.workspace.join("target")).unwrap();
    f.seed(&f.workspace.join("target"));
    f.environment
        .insert("CLEANUP_BINARY_PATH".into(), "target".into());
    assert!(!f.cleanup("smoke", true, None).process.success());
    assert!(artifact.join("payload").exists() && f.workspace.join("target/payload").exists());
}
#[test]
fn actual_cleanup_replay_removes_owned_and_missing_worktrees_and_keeps_unrelated_registration() {
    let f = Fixture::new();
    let git = f.repository();
    let owned = f
        .temporary
        .join("agentic-replay-worktrees/owned space\nnewline");
    let missing = f.temporary.join("agentic-replay-worktrees/missing");
    let unrelated = f.temporary.join("agentic-replay-worktrees-other/keep");
    let stale = f.temporary.join("unrelated-missing");
    for p in [&owned, &missing, &unrelated, &stale] {
        f.add(p);
    }
    for p in [&missing, &stale] {
        ownership::check(&f.directory, p).unwrap();
        fs::remove_dir_all(p).unwrap();
    }
    let evidence = f.temporary.join("agentic-replay-artifacts");
    f.seed(&evidence);
    for _ in 0..2 {
        assert!(f.cleanup("replay", false, Some(&git)).process.success());
        assert!(!f.registered(&owned) && !f.registered(&missing));
        assert!(f.registered(&unrelated) && f.registered(&stale));
        assert!(!f.temporary.join("agentic-replay-worktrees").exists());
        assert!(unrelated.exists() && evidence.join("payload").exists());
    }
}
#[test]
fn actual_cleanup_locked_replay_worktree_refusal_preserves_file_targets_and_registration() {
    let f = Fixture::new();
    let git = f.repository();
    let owned = f.temporary.join("agentic-replay-worktrees/locked");
    f.add(&owned);
    f.git(&["worktree", "lock", owned.to_str().unwrap()]);
    let disposable = f.workspace.join("ci/agentic-replay-nightly/.venv");
    f.seed(&disposable);
    let result = f.cleanup("replay", true, Some(&git));
    assert!(!result.process.success());
    assert!(result.stdout.unwrap().as_bytes().is_empty());
    assert!(owned.exists() && f.registered(&owned) && disposable.join("payload").exists());
}
