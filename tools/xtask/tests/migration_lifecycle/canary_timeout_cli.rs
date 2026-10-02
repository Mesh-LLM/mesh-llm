use crate::process::{
    Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness, Value,
    supervise_raw,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};

fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(8),
        graceful_shutdown: Duration::from_secs(30),
        forced_shutdown: Duration::from_secs(5),
        retained_bytes_per_stream: 4096,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}
fn spec(executable: &str, args: Vec<String>, cwd: &Path) -> ProcessSpec {
    ProcessSpec {
        executable: executable.into(),
        cwd: cwd.into(),
        arguments: args
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect(),
        environment: BTreeMap::from([
            (
                "PATH".into(),
                Value::Public("/usr/bin:/bin:/opt/homebrew/bin".into()),
            ),
            (
                "CANARY_TIMEOUT_FIXTURE".into(),
                Value::Public("space and\nnewline".into()),
            ),
        ]),
    }
}
fn raw(spec: &ProcessSpec, token: &Cancellation) -> crate::process::RawProcessReport {
    supervise_raw(
        spec,
        &limits(),
        token,
        RawCaptureOptions {
            stdout: NonZeroUsize::new(32768),
            stderr: NonZeroUsize::new(32768),
        },
    )
    .unwrap()
}
fn input(root: &Path, script: &str, seconds: u64) -> PathBuf {
    let input = root.join("command.json");
    fs::write(
        &input,
        serde_json::to_vec(&json!({"label":"CLI fixture","seconds":seconds,
        "cwd":root,"executable":"/bin/sh","arguments":["-c",script]}))
        .unwrap(),
    )
    .unwrap();
    input
}
fn cli(input: &Path, cwd: &Path) -> ProcessSpec {
    spec(
        env!("CARGO_BIN_EXE_xtask"),
        vec![
            "automation".into(),
            "canary-timeout".into(),
            "--input".into(),
            input.to_str().unwrap().into(),
        ],
        cwd,
    )
}

#[test]
fn actual_cli_input_preserves_nonzero_and_native_signal_status() {
    let root = tempfile::tempdir().unwrap();
    for (script, code) in [("exit 7", 7), ("kill -TERM $$", 143)] {
        let report = raw(
            &cli(&input(root.path(), script, 2), root.path()),
            &Cancellation::default(),
        );
        assert!(report.process.cleanup.complete);
        assert_eq!(report.process.status.unwrap().code(), Some(code));
    }
}

#[test]
fn actual_wrapper_jq_to_cli_preserves_full_arguments_and_environment() {
    let root = tempfile::tempdir().unwrap();
    let wrapper = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/llama-canary-agent-repair.sh");
    let source = fs::read_to_string(wrapper).unwrap();
    let start = source.find("run_for() {").unwrap();
    let end = source[start..].find("\n}\n").unwrap() + start + 3;
    assert!(
        source[start..end].contains("automation canary-timeout"),
        "fixture requires the Rust production caller, never a Python oracle"
    );
    let script = root.path().join("wrapper-function.sh");
    fs::write(
        &script,
        format!(
            "set -euo pipefail\n{}\nrun_for 'argument fixture' 2 \"$@\"\n",
            &source[start..end]
        ),
    )
    .unwrap();
    let arguments = vec![
        "with spaces".to_owned(),
        String::new(),
        "multiple\nlines".into(),
        "x".repeat(8192),
    ];
    let mut args = vec![
        script.to_str().unwrap().into(),
        "/bin/sh".into(),
        "-c".into(),
        "printf '%s\\0' \"$@\"; printf '%s\\0' \"$CANARY_TIMEOUT_FIXTURE\" \"$PWD\" >&2; exit 7"
            .into(),
        "fixture".into(),
    ];
    args.extend(arguments.clone());
    let mut command = spec("/bin/bash", args, root.path());
    for (key, value) in [
        ("HARNESS_MODE", "pinned-build"),
        ("RUNNER_TEMP", root.path().to_str().unwrap()),
        ("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask")),
    ] {
        command
            .environment
            .insert(key.into(), Value::Public(value.into()));
    }
    let report = raw(&command, &Cancellation::default());
    assert!(report.process.cleanup.complete);
    assert_eq!(report.process.status.unwrap().code(), Some(7));
    let expected: Vec<u8> = arguments
        .iter()
        .flat_map(|arg| arg.bytes().chain(std::iter::once(0)))
        .collect();
    assert_eq!(report.stdout.unwrap().as_bytes(), expected);
    let expected = format!(
        "space and\nnewline\0{}\0",
        root.path().canonicalize().unwrap().display()
    );
    assert_eq!(report.stderr.unwrap().as_bytes(), expected.as_bytes());
}

#[test]
fn nested_outer_cancellation_leaves_inner_cleanup_margin_and_unrelated_sentinel() {
    let root = tempfile::tempdir().unwrap();
    let input = input(
        root.path(),
        "trap '' TERM; /bin/sleep 30 & echo $! > descendant; echo ready > ready; wait",
        30,
    );
    let mut sentinel = Command::new("/bin/sleep").arg("40").spawn().unwrap();
    let token = Cancellation::default();
    let trigger = token.clone();
    let directory = root.path().to_path_buf();
    let cancel = thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(5);
        while !directory.join("ready").is_file() && Instant::now() < until {
            thread::sleep(Duration::from_millis(10));
        }
        let ready = directory.join("ready").is_file();
        trigger.cancel();
        ready
    });
    let started = Instant::now();
    let report = raw(&cli(&input, root.path()), &token);
    let ready = cancel.join().unwrap();
    let alive = sentinel.try_wait().unwrap().is_none();
    sentinel.kill().unwrap();
    sentinel.wait().unwrap();
    assert!(ready && alive);
    assert_eq!(report.process.outcome, Outcome::Cancelled);
    assert!(report.process.cleanup.complete && !report.process.cleanup.forced);
    assert_eq!(report.process.status.unwrap().code(), Some(143));
    assert!(
        started.elapsed() < Duration::from_secs(25),
        "inner owner must finish inside outer30s grace"
    );
}
