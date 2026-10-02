use super::{Error, Git, Profile, admission, deletion, fixtures::Fixture};
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, ffi::OsString, num::NonZeroUsize, path::Path, time::Duration};

pub(super) struct Repository {
    pub(super) fixture: Fixture,
    pub(super) git: Git,
    executable: std::path::PathBuf,
    pub(super) environment: BTreeMap<OsString, OsString>,
}
impl Repository {
    pub(super) fn new() -> Self {
        let fixture = Fixture::new();
        let executable = std::path::PathBuf::from(
            std::env::var_os("MIGRATION_TEST_GIT")
                .expect("set MIGRATION_TEST_GIT to an absolute Git executable"),
        );
        assert!(executable.is_absolute());
        let mut environment: BTreeMap<OsString, OsString> =
            ["PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"]
                .into_iter()
                .filter_map(|name| std::env::var_os(name).map(|value| (name.into(), value)))
                .collect();
        for (name, value) in [
            ("HOME", fixture.root.path().as_os_str().to_owned()),
            ("GIT_CONFIG_NOSYSTEM", "1".into()),
            (
                "GIT_CONFIG_GLOBAL",
                fixture.root.path().join("empty-config").into_os_string(),
            ),
            ("GIT_TERMINAL_PROMPT", "0".into()),
            ("GIT_MASTER", "1".into()),
        ] {
            environment.insert(name.into(), value);
        }
        std::fs::write(fixture.root.path().join("empty-config"), b"").unwrap();
        let git = Git::new(executable.clone(), environment.clone()).unwrap();
        let repository = Self {
            fixture,
            git,
            executable,
            environment,
        };
        repository.command(&["init".into(), "--template=".into()]);
        repository.command(&[
            "-c".into(),
            "user.name=Cleanup Test".into(),
            "-c".into(),
            "user.email=cleanup@example.invalid".into(),
            "-c".into(),
            "commit.gpgsign=false".into(),
            "-c".into(),
            "core.hooksPath=disabled-hooks".into(),
            "commit".into(),
            "--allow-empty".into(),
            "-m".into(),
            "fixture".into(),
        ]);
        repository
    }
    pub(super) fn command(&self, args: &[OsString]) -> Vec<u8> {
        self.check_workspace();
        super::fixture_git_admission::check(&self.fixture, args);
        if args.first().is_none_or(|arg| arg != "init") {
            assert!(self.fixture.workspace.join(".git").is_dir());
        }
        let spec = ProcessSpec {
            executable: self.executable.clone(),
            arguments: args.iter().cloned().map(Value::Public).collect(),
            cwd: self.fixture.workspace.clone(),
            environment: self
                .environment
                .iter()
                .map(|(key, value)| (key.clone(), Value::Public(value.clone())))
                .collect(),
        };
        let limits = Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = process::supervise_raw(
            &spec,
            &limits,
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(1024 * 1024),
                stderr: None,
            },
        )
        .unwrap();
        assert!(report.process.success(), "{:?}", report.process);
        report.stdout.unwrap().as_bytes().to_vec()
    }
    pub(super) fn add(&self, path: &Path) {
        super::fixtures::ownership::check(&self.fixture.root, path).unwrap();
        self.command(&[
            "worktree".into(),
            "add".into(),
            "--detach".into(),
            path.as_os_str().to_owned(),
            "HEAD".into(),
        ]);
    }
    pub(super) fn list(&self) -> Vec<u8> {
        self.command(&[
            "worktree".into(),
            "list".into(),
            "--porcelain".into(),
            "-z".into(),
        ])
    }
    fn cleanup(&self) -> Result<Vec<u8>, super::error::Failure<Vec<u8>>> {
        self.check_workspace();
        assert!(self.fixture.workspace.join(".git").is_dir());
        let plan = self.fixture.plan(Profile::Replay, false);
        self.fixture.check_plan(&plan);
        let admitted = admission::admit(&plan)?;
        let interrupt = super::fixtures::interrupt().map_err(Error::Interrupt)?;
        let mut output = Vec::new();
        let result = deletion::execute(admitted, Some(&self.git), (&interrupt, &mut output));
        super::finalization::finalize(result.map(|()| output), interrupt.finish())
    }

    pub(super) fn check_workspace(&self) {
        super::fixture_git_admission::workspace(&self.fixture);
    }
}

fn registered(listing: &[u8], path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    listing
        .split(|byte| *byte == 0)
        .any(|field| field.strip_prefix(b"worktree ") == Some(path.as_os_str().as_bytes()))
}

#[test]
fn owned_and_stale_worktrees_when_cleanup_repeats_keep_unrelated_registrations() {
    let repository = Repository::new();
    let root = repository
        .fixture
        .temporary
        .join("agentic-replay-worktrees");
    let owned = root.join("owned worktree\nwith newline");
    let missing = root.join("already-missing");
    let unrelated = repository
        .fixture
        .temporary
        .join("agentic-replay-worktrees-other/keep");
    let stale = repository.fixture.temporary.join("unrelated-missing");
    for path in [&owned, &missing, &unrelated, &stale] {
        repository.add(path);
    }
    super::fixtures::ownership::check(&repository.fixture.root, &missing).unwrap();
    super::fixtures::ownership::check(&repository.fixture.root, &stale).unwrap();
    std::fs::remove_dir_all(&missing).unwrap();
    std::fs::remove_dir_all(&stale).unwrap();
    let evidence = repository
        .fixture
        .temporary
        .join("agentic-replay-artifacts");
    repository.fixture.seed(&evidence);
    for _ in 0..2 {
        repository.cleanup().unwrap();
        let listing = repository.list();
        assert!(!registered(&listing, &owned));
        assert!(!registered(&listing, &missing));
        assert!(registered(&listing, &unrelated));
        assert!(registered(&listing, &stale));
        assert!(!root.exists());
        assert!(unrelated.exists());
        assert!(evidence.join("payload").exists());
    }
}

#[test]
fn locked_owned_worktree_when_git_rejects_removal_preserves_all_file_targets() {
    let repository = Repository::new();
    let owned = repository
        .fixture
        .temporary
        .join("agentic-replay-worktrees/locked");
    repository.add(&owned);
    repository.command(&[
        "worktree".into(),
        "lock".into(),
        owned.as_os_str().to_owned(),
    ]);
    let venv = repository
        .fixture
        .workspace
        .join("ci/agentic-replay-nightly/.venv");
    repository.fixture.seed(&venv);
    assert!(matches!(
        repository.cleanup(),
        Err(super::error::Failure::Operation(Error::Git(_)))
    ));
    assert!(owned.exists());
    assert!(venv.join("payload").exists());
    assert!(registered(&repository.list(), &owned));
}

#[test]
fn replay_root_symlink_when_admitted_file_list_is_safe_rejects_before_git_and_deletion() {
    let fixture = Fixture::new();
    let outside = fixture.root.path().join("outside");
    fixture.seed(&outside);
    let root = fixture.temporary.join("agentic-replay-worktrees");
    std::os::unix::fs::symlink(&outside, &root).unwrap();
    let plan = fixture.plan(Profile::Replay, true);
    fixture.seed(&plan.targets[0].path);
    let interrupt = super::fixtures::interrupt().unwrap();
    fixture.check_plan(&plan);
    let result = deletion::execute(
        admission::admit(&plan).unwrap(),
        None,
        (&interrupt, &mut Vec::new()),
    );
    interrupt.finish().unwrap();
    assert!(matches!(result, Err(Error::ReplayRoot(_))));
    assert!(outside.join("payload").exists());
    assert!(plan.targets[0].path.join("payload").exists());
}

#[test]
fn full_worktree_admission_when_last_registered_path_is_symlink_rejects_all_removals() {
    use std::os::unix::ffi::OsStrExt;
    let fixture = Fixture::new();
    let root = fixture.temporary.join("agentic-replay-worktrees");
    let safe = root.join("safe");
    fixture.seed(&safe);
    let external = fixture.root.path().join("external");
    fixture.seed(&external);
    let link = root.join("linked");
    std::os::unix::fs::symlink(&external, &link).unwrap();
    let mut listing = Vec::new();
    for path in [&safe, &link] {
        listing.extend_from_slice(b"worktree ");
        listing.extend_from_slice(path.as_os_str().as_bytes());
        listing.push(0);
    }
    assert!(matches!(
        super::replay::owned_worktrees(&root, &listing),
        Err(Error::ReplaySymlink(_))
    ));
    assert!(safe.join("payload").exists());
    assert!(external.join("payload").exists());
}
