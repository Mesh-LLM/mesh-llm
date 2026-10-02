//! Execute the actual precheckout guard with finite uname/Git/xcrun boundaries.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use crate::workflow_yaml;
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn guard() -> String {
    let source =
        fs::read_to_string(root().join(".github/workflows/agentic-replay-nightly.yml")).unwrap();
    let tree = workflow_yaml::parse(&source).unwrap();
    let replay = tree.get("jobs").unwrap().get("replay").unwrap();
    let workflow_yaml::Node::Seq(steps) = replay.get("steps").unwrap() else {
        panic!("steps")
    };
    assert_eq!(
        steps[0].get("name").and_then(workflow_yaml::Node::text),
        Some("Verify pinned replay runner")
    );
    steps[0].get("run").unwrap().text().unwrap().to_owned()
}
fn executable(path: &Path, source: &str) {
    fs::write(path, format!("#!/bin/sh\nset -eu\n{source}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn bash() -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|dir| dir.join("bash"))
        .find(|path| {
            path.is_file() && fs::metadata(path).unwrap().permissions().mode() & 0o111 != 0
        })
        .unwrap()
}
struct GuardResult {
    success: bool,
    calls: String,
}
fn execute_guard(source: &str, arch: &str, runner: &str) -> GuardResult {
    let directory = tempfile::tempdir().unwrap();
    let bin = directory.path().join("bin");
    fs::create_dir(&bin).unwrap();
    executable(
        &bin.join("uname"),
        "[ \"$*\" = '-m' ] || exit 90\nprintf '%s\\n' \"$FIXTURE_ARCH\"",
    );
    executable(
        &bin.join("git"),
        "[ \"$*\" = '--version' ] || exit 91\nprintf 'git-version\\n' >> \"$FIXTURE_CALLS\"",
    );
    executable(
        &bin.join("xcrun"),
        "[ \"$*\" = '--find clang' ] || exit 92\nprintf 'xcrun-find-clang\\n' >> \"$FIXTURE_CALLS\"",
    );
    let calls = directory.path().join("calls");
    let environment = [
        ("PATH", format!("{}:/usr/bin:/bin", bin.display())),
        ("FIXTURE_ARCH", arch.into()),
        ("RUNNER_NAME", runner.into()),
        ("EXPECTED_REPLAY_RUNNER_NAME", "micstudio".into()),
        ("FIXTURE_CALLS", calls.display().to_string()),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), Value::Public(value.into())))
    .collect::<BTreeMap<_, _>>();
    let spec = ProcessSpec {
        executable: bash(),
        cwd: directory.path().to_owned(),
        arguments: vec![Value::Public("-c".into()), Value::Public(source.into())],
        environment,
    };
    let limits = Limits {
        execution: Duration::from_secs(4),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 8192,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let result = process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(8192),
            stderr: std::num::NonZeroUsize::new(8192),
        },
    )
    .unwrap();
    assert!(
        result.process.cleanup.complete && result.process.failure.is_none(),
        "{:?}",
        result.process
    );
    assert_eq!(result.process.outcome, process::Outcome::Exited);
    GuardResult {
        success: result.process.status.unwrap().success(),
        calls: fs::read_to_string(calls).unwrap_or_default(),
    }
}
#[test]
fn actual_precheckout_guard_admits_only_pinned_native_runner_before_tool_probes() {
    let source = guard();
    let admitted = execute_guard(&source, "arm64", "micstudio");
    assert!(admitted.success);
    assert_eq!(admitted.calls, "git-version\nxcrun-find-clang\n");
    for (arch, runner) in [
        ("x86_64", "micstudio"),
        ("arm64", "studio54"),
        ("x86_64", "studio54"),
    ] {
        let rejected = execute_guard(&source, arch, runner);
        assert!(!rejected.success, "{arch}/{runner}");
        assert!(
            rejected.calls.is_empty(),
            "tool probes must not precede runner admission"
        );
    }
}
#[test]
fn runner_guard_mutations_are_detected_by_actual_failure_cases() {
    let source = guard();
    for (from, to, arch, runner) in [
        (
            "\"$RUNNER_NAME\" != \"$EXPECTED_REPLAY_RUNNER_NAME\"",
            "1 != 1",
            "arm64",
            "studio54",
        ),
        (
            "\"$(uname -m)\" != \"arm64\"",
            "1 != 1",
            "x86_64",
            "micstudio",
        ),
    ] {
        assert!(source.contains(from));
        let changed = execute_guard(&source.replace(from, to), arch, runner);
        assert!(changed.success, "mutant must demonstrate loss of rejection");
        assert_eq!(changed.calls, "git-version\nxcrun-find-clang\n");
    }
}
