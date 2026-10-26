use crate::{
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
    real_git::Repository,
};
use serde_json::json;
use std::{
    collections::BTreeMap, ffi::OsString, fs, num::NonZeroUsize, os::unix::fs::PermissionsExt,
    path::PathBuf, time::Duration,
};
struct Fixture {
    repo: Repository,
    _tools: tempfile::TempDir,
    output: tempfile::TempDir,
    environment: BTreeMap<OsString, OsString>,
}
impl Fixture {
    fn new() -> Self {
        let repo = Repository::new();
        let tools = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        let executable = tools.path().join("gh");
        fs::write(&executable,"#!/bin/sh\nprintf invoked > \"$FIXTURE_LAUNCH\"\ncase \"$1\" in\nrelease) /bin/cat \"$FIXTURE_RELEASE\";;\npr) case \"$FIXTURE_MODE\" in\nmutate) printf 'mutated during collection' > \"$FIXTURE_TRACKED\";;\ncancel) sleep 30 &\nchild=$!\ntrap 'kill \"$child\" 2>/dev/null || :; wait \"$child\" 2>/dev/null || :; exit 143' TERM INT\nprintf '%s' \"$child\" > \"$FIXTURE_CHILD\"\nkill -TERM \"$PPID\"\nwait \"$child\";;\nfail) printf 'credential-private-source' >&2; exit 7;;\nesac\nprintf '[]';;\n*) exit 9;;\nesac\n").unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(tools.path().join("release.json"),serde_json::to_vec(&json!({"tagName":"v1.0.0","name":"retained name","publishedAt":"2026-10-01T00:00:00Z","url":"https://example.invalid/release","isDraft":false,"isPrerelease":false,"body":"full body","targetCommitish":"main","assets":[{"name":"archive","size":17}]})).unwrap()).unwrap();
        let mut environment: BTreeMap<_, _> = std::env::vars_os().collect();
        for name in ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"] {
            environment.remove(&OsString::from(name));
        }
        environment.insert(
            "PATH".into(),
            std::env::join_paths(
                std::iter::once(tools.path().to_path_buf())
                    .chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
            )
            .unwrap(),
        );
        for (key, path) in [
            ("FIXTURE_RELEASE", tools.path().join("release.json")),
            ("FIXTURE_LAUNCH", tools.path().join("launch")),
            ("FIXTURE_TRACKED", repo.root.join("tracked binary")),
            ("FIXTURE_CHILD", tools.path().join("child")),
            ("GIT_CONFIG_GLOBAL", tools.path().join("absent-config")),
        ] {
            environment.insert(key.into(), path.into());
        }
        environment.insert("GIT_CONFIG_NOSYSTEM".into(), "1".into());
        environment.insert("GH_TOKEN".into(), "fixture-private-auth".into());
        Self {
            repo,
            _tools: tools,
            output,
            environment,
        }
    }
    fn run(&self, args: &[&str], mode: &str, cwd: Option<PathBuf>) -> process::RawProcessReport {
        let mut environment = self
            .environment
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    if key.to_str() == Some("GH_TOKEN") {
                        Value::Secret(value.clone())
                    } else {
                        Value::Public(value.clone())
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        environment.insert("FIXTURE_MODE".into(), Value::Public(mode.into()));
        process::supervise_raw(
            &ProcessSpec {
                executable: PathBuf::from(env!("CARGO_BIN_EXE_xtask")),
                cwd: cwd.unwrap_or_else(|| self.repo.root.clone()),
                environment,
                arguments: ["release", "inventory"]
                    .into_iter()
                    .chain(args.iter().copied())
                    .map(|v| Value::Public(v.into()))
                    .collect(),
            },
            &Limits {
                execution: Duration::from_secs(30),
                graceful_shutdown: Duration::from_secs(2),
                forced_shutdown: Duration::from_secs(5),
                retained_bytes_per_stream: 1024 * 1024,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(4 * 1024 * 1024),
                stderr: NonZeroUsize::new(1024 * 1024),
            },
        )
        .unwrap()
    }
    fn child_gone(&self) {
        let pid: i32 = fs::read_to_string(self._tools.path().join("child"))
            .unwrap()
            .parse()
            .unwrap();
        // SAFETY: probes only the descendant PID recorded by this private owned fixture.
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
}
fn assert_exit(report: &process::RawProcessReport, code: i32) {
    assert!(
        report.process.cleanup.complete && report.process.failure.is_none(),
        "{:?}",
        report.process
    );
    assert_eq!(report.process.outcome, process::Outcome::Exited);
    assert_eq!(report.process.status.and_then(|s| s.code()), Some(code));
}
#[test]
fn release_inventory_publication_actual_cli_complete_schema_candidate_workspace_and_subdir() {
    let mut fixture = Fixture::new();
    let requested = fixture.repo.ok(&["rev-parse", "HEAD"]);
    fs::write(fixture.repo.root.join("tracked binary"), b"original\0\xff").unwrap();
    fixture.repo.ok(&["add", "-A"]);
    let working = fixture.repo.main_commit("feature: tracked source");
    fs::write(fixture.repo.root.join("tracked binary"), b"staged\0\xfe").unwrap();
    fixture.repo.ok(&["add", "-A"]);
    fs::write(fixture.repo.root.join("tracked binary"), b"unstaged\0\xfd").unwrap();
    fs::write(fixture.repo.root.join("untracked bytes"), b"actual\0\xff").unwrap();
    fs::create_dir(fixture.repo.root.join("subdir")).unwrap();
    fixture.repo.ok(&[
        "remote",
        "add",
        "origin",
        "https://operator:credential-private-source@example.invalid/Mesh-LLM/mesh-llm.git",
    ]);
    let report = fixture.run(
        &["--repo=Mesh-LLM/mesh-llm", "--head", "v1.0.0"],
        "success",
        Some(fixture.repo.root.join("subdir")),
    );
    assert_exit(&report, 0);
    let bytes = report.stdout.unwrap();
    let value: serde_json::Value = serde_json::from_slice(bytes.as_bytes()).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["candidate"]["sha"], requested);
    assert_eq!(value["candidate"]["working_tree_head"], working);
    assert_eq!(value["candidate"]["dirty"]["diff_base"], working);
    assert!(value["candidate"]["dirty"]["is_dirty"].as_bool().unwrap());
    assert!(
        value["candidate"]["dirty"]["staged_against_head"]["bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(value["previous_release"]["assets"][0]["name"], "archive");
    assert!(
        value["comparison"]["merged_prs_since_release"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        value["comparison"]["merged_pr_query_scope"]
            .as_str()
            .unwrap()
            .contains("not a complete enumeration")
    );
    assert!(value["collected_at"].as_str().unwrap().contains('T'));
    assert!(!String::from_utf8_lossy(bytes.as_bytes()).contains("credential-private-source"));
    assert!(
        !String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
            .contains("fixture-private-auth")
    );
}
#[test]
fn release_inventory_publication_actual_cli_atomic_output_inside_worktree_and_help() {
    let fixture = Fixture::new();
    let output = fixture.repo.root.join("inventory report.json");
    fs::write(&output, b"prior report").unwrap();
    let report = fixture.run(&["--output", output.to_str().unwrap()], "success", None);
    assert_exit(&report, 0);
    assert!(report.stdout.unwrap().as_bytes().is_empty());
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert!(
        value["candidate"]["dirty"]["observation_scope"]
            .as_str()
            .unwrap()
            .contains("before inventory publication")
    );
    assert_eq!(
        fs::read_dir(&fixture.repo.root)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e
                .file_name()
                .to_string_lossy()
                .starts_with(".release-inventory-"))
            .count(),
        0
    );
    fs::remove_file(fixture._tools.path().join("launch")).unwrap();
    let help = fixture.run(&["--help"], "fail", None);
    assert_exit(&help, 0);
    assert!(!fixture._tools.path().join("launch").exists());
    assert!(String::from_utf8_lossy(help.stdout.unwrap().as_bytes()).contains("release inventory"));
}
#[test]
fn release_inventory_publication_actual_cli_source_mutation_or_request_failure_preserves_old_output()
 {
    let mut fixture = Fixture::new();
    fs::write(fixture.repo.root.join("tracked binary"), b"original source").unwrap();
    fixture.repo.ok(&["add", "-A"]);
    fixture.repo.main_commit("feature: source");
    let output = fixture.output.path().join("inventory.json");
    for mode in ["mutate", "fail"] {
        fs::write(&output, b"previous complete output").unwrap();
        let report = fixture.run(&["--output", output.to_str().unwrap()], mode, None);
        assert_exit(&report, 1);
        assert_eq!(fs::read(&output).unwrap(), b"previous complete output");
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        assert!(
            !String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
                .contains("credential-private-source")
        );
    }
}
#[test]
fn release_inventory_publication_actual_cli_signal_cancels_gh_child_preserves_output() {
    let fixture = Fixture::new();
    let output = fixture.output.path().join("inventory.json");
    fs::write(&output, b"prior complete report").unwrap();
    let report = fixture.run(&["--output", output.to_str().unwrap()], "cancel", None);
    assert_exit(&report, 1);
    assert_eq!(fs::read(&output).unwrap(), b"prior complete report");
    assert!(report.stdout.unwrap().as_bytes().is_empty());
    assert!(String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains("cancel"));
    fixture.child_gone();
}
