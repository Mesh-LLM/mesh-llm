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

#[test]
fn actual_cli_closes_piped_manifest_input_before_command_launch() {
    let root = tempfile::tempdir().unwrap();
    let command_input = input(
        root.path(),
        "if IFS= read -r row; then printf '%s' \"$row\" > consumed-row; exit 9; fi; exit 0",
        2,
    );
    let invocation = spec(
        "/bin/sh",
        vec![
            "-c".into(),
            "printf 'a later manifest row\\n' | \"$@\"".into(),
            "stdin fixture".into(),
            env!("CARGO_BIN_EXE_xtask").into(),
            "automation".into(),
            "canary-timeout".into(),
            "--input".into(),
            command_input.to_str().unwrap().into(),
        ],
        root.path(),
    );
    let report = raw(&invocation, &Cancellation::default());
    assert!(
        report.process.failure.is_none(),
        "{:?}",
        report.process.failure
    );
    assert!(report.process.cleanup.complete);
    assert_eq!(report.process.status.unwrap().code(), Some(0));
    assert!(!root.path().join("consumed-row").exists());
}

#[test]
fn actual_cli_refuses_special_and_oversize_requests_before_child_launch() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let valid = input(root.path(), "printf launched > launched", 2);
    let fifo = root.path().join("request.fifo");
    let fifo_name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo_name.as_ptr(), 0o600) }, 0);
    let link = root.path().join("request.link");
    symlink(&valid, &link).unwrap();
    let oversized = root.path().join("oversized.json");
    fs::File::create(&oversized)
        .unwrap()
        .set_len(16 * 1024 * 1024 + 1)
        .unwrap();
    let invalid = root.path().join("invalid.json");
    fs::write(&invalid, b"{malformed").unwrap();
    for path in [&fifo, &link, &oversized, &invalid] {
        let report = raw(&cli(path, root.path()), &Cancellation::default());
        assert_eq!(
            report.process.outcome,
            Outcome::Exited,
            "{:?}",
            report.process.outcome
        );
        assert!(report.process.failure.is_none());
        assert!(report.process.cleanup.complete);
        assert!(!report.process.status.unwrap().success());
        assert!(!root.path().join("launched").exists());
    }
}

fn raw_with_cleanup_budget(spec: &ProcessSpec) -> crate::process::RawProcessReport {
    let mut budget = limits();
    budget.execution = Duration::from_secs(25);
    supervise_raw(
        spec,
        &budget,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(32768),
            stderr: NonZeroUsize::new(32768),
        },
    )
    .unwrap()
}

#[test]
fn actual_cli_completed_stubborn_writer_stops_before_workspace_handoff() {
    let root = tempfile::tempdir().unwrap();
    let command = input(
        root.path(),
        "(trap '' TERM; echo ready > writer-ready; i=0; while [ \"$i\" -lt 1000 ]; do printf '%s' \"$i\" > writes; i=$((i+1)); /bin/sleep 0.02; done) >/dev/null 2>&1 & while [ ! -f writer-ready ] || [ ! -f writes ]; do /bin/sleep 0.01; done; exit 0",
        20,
    );
    let report = raw_with_cleanup_budget(&cli(&command, root.path()));
    assert!(report.process.cleanup.complete);
    assert_eq!(report.process.status.unwrap().code(), Some(0));
    let before = fs::read(root.path().join("writes")).unwrap();
    thread::sleep(Duration::from_millis(120));
    assert_eq!(
        fs::read(root.path().join("writes")).unwrap(),
        before,
        "completed command must not hand back a workspace while a descendant can still write"
    );
}

fn repeated_signal_case(first: i32, second: i32) {
    let root = tempfile::tempdir().unwrap();
    let command = input(
        root.path(),
        "trap 'echo observed > stop-seen' TERM; echo ready > ready; i=0; while [ \"$i\" -lt 1000 ]; do printf '%s' \"$i\" > writes; i=$((i+1)); /bin/sleep 0.02; done",
        20,
    );
    let invocation = spec(
        "/bin/sh",
        vec![
            "-c".into(),
            "echo $$ > wrapper-pid; exec \"$@\"".into(),
            "signal launcher".into(),
            env!("CARGO_BIN_EXE_xtask").into(),
            "automation".into(),
            "canary-timeout".into(),
            "--input".into(),
            command.to_str().unwrap().into(),
        ],
        root.path(),
    );
    let directory = root.path().to_path_buf();
    let signals = thread::spawn(move || {
        let wait_for = |name: &str| {
            let until = Instant::now() + Duration::from_secs(5);
            while !directory.join(name).is_file() && Instant::now() < until {
                thread::sleep(Duration::from_millis(5));
            }
            directory.join(name).is_file()
        };
        if !wait_for("ready") {
            return false;
        }
        let pid: i32 = fs::read_to_string(directory.join("wrapper-pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert!(pid > 0);
        // SAFETY: the still-unreaped outer owned child reserves this exact PID.
        if unsafe { libc::kill(pid, first) } != 0 {
            return false;
        }
        if !wait_for("stop-seen") {
            return false;
        }
        // SAFETY: first signal triggered inner cleanup; the wrapper remains owned
        // until the raw supervisor returns, and its PID cannot be recycled.
        unsafe { libc::kill(pid, second) == 0 }
    });
    let report = raw_with_cleanup_budget(&invocation);
    let delivered = signals.join().unwrap();
    assert!(report.process.cleanup.complete);
    assert!(
        delivered,
        "both signals must be observed at their causal boundaries"
    );
    assert_eq!(report.process.status.unwrap().code(), Some(128 + first));
    assert!(
        String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
            .contains(&format!("CLI fixture received signal {first}"))
    );
    let before = fs::read(root.path().join("writes")).unwrap();
    thread::sleep(Duration::from_millis(120));
    assert_eq!(fs::read(root.path().join("writes")).unwrap(), before);
}

#[test]
fn actual_cli_deadline_retains_caller_label_and_stops_owned_command() {
    let root = tempfile::tempdir().unwrap();
    for label in ["fixture", "agent developer task"] {
        let request = input(root.path(), "exec /bin/sleep 30", 1);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&request).unwrap()).unwrap();
        value["label"] = label.into();
        fs::write(&request, serde_json::to_vec(&value).unwrap()).unwrap();
        let report = raw_with_cleanup_budget(&cli(&request, root.path()));
        assert!(report.process.cleanup.complete);
        assert_eq!(report.process.status.unwrap().code(), Some(124));
        assert!(
            String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
                .contains(&format!("{label} timed out after 1s"))
        );
    }
}

#[test]
fn actual_cli_repeated_signal_during_cleanup_preserves_first_status_and_reaps_writer() {
    repeated_signal_case(libc::SIGTERM, libc::SIGINT);
    repeated_signal_case(libc::SIGINT, libc::SIGTERM);
}
