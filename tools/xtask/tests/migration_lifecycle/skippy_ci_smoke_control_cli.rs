use crate::process::{
    Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    supervise_raw,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
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
fn source() -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/skippy-ci-smoke.sh"),
    )
    .unwrap()
}
fn function(source: &str, name: &str) -> String {
    let start = source.find(&format!("{name}() {{")).unwrap();
    let end = source[start..].find("\n}\n").unwrap() + start + 3;
    source[start..end].into()
}
fn selector(source: &str) -> &str {
    source
        .split("# Frozen automation selection begins.\n")
        .nth(1)
        .unwrap()
        .split("# Frozen automation selection ends.")
        .next()
        .unwrap()
}
fn caller(root: &Path, label: &str, seconds: u64, arguments: Vec<String>) -> ProcessSpec {
    let source = source();
    let script = root.join("smoke timeout caller.sh");
    fs::write(&script,format!("set -euo pipefail\n{}\nWORK_DIR=\"$RUNNER_TEMP\"\n{}\nrun_with_timeout \"$LABEL\" \"$@\" <\"$INPUT_FILE\"\n",selector(&source),function(&source,"run_with_timeout"))).unwrap();
    let mut args = vec![script.display().to_string()];
    args.extend(arguments);
    let mut command = spec("/bin/bash", args, root);
    for (key, value) in [
        ("ROOT", root.display().to_string()),
        ("RUNNER_TEMP", root.display().to_string()),
        (
            "MESH_LLM_AUTOMATION_BIN",
            env!("CARGO_BIN_EXE_xtask").into(),
        ),
        ("SMOKE_COMMAND_TIMEOUT_SECS", seconds.to_string()),
        ("LABEL", label.into()),
        ("INPUT_FILE", root.join("stdin.txt").display().to_string()),
    ] {
        command
            .environment
            .insert(key.into(), Value::Public(value.into()));
    }
    command
}
fn executable(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn empty_input(root: &Path) {
    fs::write(root.join("stdin.txt"), b"").unwrap();
}
#[test]
fn actual_smoke_relative_executable_preserves_redirected_stdin_argv_env_and_native_failure() {
    let root = tempfile::tempdir().unwrap();
    let stdin = [
        b"original prompt\n\0raw input\xff\n".as_slice(),
        &vec![b'x'; 8192],
    ]
    .concat();
    fs::write(root.path().join("stdin.txt"), &stdin).unwrap();
    fs::create_dir(root.path().join("relative bin")).unwrap();
    executable(
        &root.path().join("relative bin/fixture"),
        "#!/bin/sh\nprintf '%s\\0' \"$@\"\n/bin/cat\nprintf '%s\\0' \"$CANARY_TIMEOUT_FIXTURE\" \"$PWD\" >&2\nexit 7\n",
    );
    let arguments = vec![
        "space arg".into(),
        String::new(),
        "multiple\nlines".into(),
        "x".repeat(8192),
    ];
    let mut args = vec!["relative bin/fixture".into()];
    args.extend(arguments.clone());
    let report = raw(
        &caller(root.path(), "prompt binary smoke", 2, args),
        &Cancellation::default(),
    );
    assert!(report.process.cleanup.complete);
    assert_eq!(report.process.status.unwrap().code(), Some(7));
    let mut expected: Vec<u8> = arguments
        .iter()
        .flat_map(|arg| arg.bytes().chain(std::iter::once(0)))
        .collect();
    expected.extend(stdin);
    assert_eq!(report.stdout.unwrap().as_bytes(), expected);
    assert_eq!(
        report.stderr.unwrap().as_bytes(),
        format!(
            "space and\nnewline\0{}\0",
            root.path().canonicalize().unwrap().display()
        )
        .as_bytes()
    );
    assert!(!fs::read_dir(root.path()).unwrap().any(|row| {
        row.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("command-timeout.")
    }));
}
#[test]
fn actual_timeout_cli_itself_inherits_a_redirected_file() {
    let root = tempfile::tempdir().unwrap();
    fs::write(
        root.path().join("stdin.txt"),
        b"actual CLI input\nsecond line\0",
    )
    .unwrap();
    let input = root.path().join("input.json");
    fs::write(&input,serde_json::to_vec(&serde_json::json!({"label":"stdin regression","seconds":2,"cwd":root.path(),"executable":"/bin/cat","arguments":[]})).unwrap()).unwrap();
    let command = spec(
        "/bin/bash",
        vec![
            "-c".into(),
            "\"$1\" automation canary-timeout --input \"$2\" <\"$3\"".into(),
            "fixture".into(),
            env!("CARGO_BIN_EXE_xtask").into(),
            input.display().to_string(),
            root.path().join("stdin.txt").display().to_string(),
        ],
        root.path(),
    );
    let report = raw(&command, &Cancellation::default());
    assert!(report.process.cleanup.complete && report.process.status.unwrap().success());
    assert_eq!(
        report.stdout.unwrap().as_bytes(),
        b"actual CLI input\nsecond line\0"
    );
}
#[test]
fn actual_smoke_timeout_preserves_signal_timeout_and_fail_closed_infrastructure() {
    let root = tempfile::tempdir().unwrap();
    empty_input(root.path());
    for (script, code) in [
        ("kill -TERM $$", 143),
        ("/bin/sleep 30 & echo $! > descendant; wait", 124),
    ] {
        let report = raw(
            &caller(
                root.path(),
                "dense binary smoke",
                1,
                vec!["/bin/sh".into(), "-c".into(), script.into()],
            ),
            &Cancellation::default(),
        );
        assert!(report.process.cleanup.complete);
        assert_eq!(report.process.status.unwrap().code(), Some(code));
        if code == 124 {
            let pid = fs::read_to_string(root.path().join("descendant")).unwrap();
            let probe = raw(
                &spec(
                    "/bin/kill",
                    vec!["-0".into(), pid.trim().into()],
                    root.path(),
                ),
                &Cancellation::default(),
            );
            assert!(
                !probe.process.status.unwrap().success(),
                "owned descendant survived"
            );
        }
    }
    for (seconds, command) in [(0, "/bin/echo"), (2, "missing-executable-fixture")] {
        let report = raw(
            &caller(
                root.path(),
                "invalid command",
                seconds,
                vec![command.into(), "must-not-execute".into()],
            ),
            &Cancellation::default(),
        );
        assert!(report.process.cleanup.complete);
        assert_eq!(report.process.status.unwrap().code(), Some(125));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
    }
}
#[test]
fn actual_smoke_nested_cancellation_cleans_owned_tree_and_leaves_unrelated_sentinel() {
    use std::os::unix::process::ExitStatusExt;
    use std::{process::Command, thread, time::Instant};
    let root = tempfile::tempdir().unwrap();
    empty_input(root.path());
    let args = vec![
        "/bin/sh".into(),
        "-c".into(),
        "trap '' TERM; /bin/sleep 30 & echo $! > descendant; echo ready > ready; wait".into(),
    ];
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
    let report = raw(&caller(root.path(), "smoke cancellation", 30, args), &token);
    let ready = cancel.join().unwrap();
    let alive = sentinel.try_wait().unwrap().is_none();
    sentinel.kill().unwrap();
    sentinel.wait().unwrap();
    assert!(ready && alive);
    assert!(report.process.cleanup.complete && !report.process.cleanup.forced);
    let status = report.process.status.unwrap();
    assert!(status.code() == Some(143) || status.signal() == Some(libc::SIGTERM));
    assert_eq!(report.process.outcome, crate::process::Outcome::Cancelled);
    assert!(
        started.elapsed() < Duration::from_secs(25),
        "inner20s cleanup must fit outer30s grace"
    );
    let pid = fs::read_to_string(root.path().join("descendant")).unwrap();
    let probe = raw(
        &spec(
            "/bin/kill",
            vec!["-0".into(), pid.trim().into()],
            root.path(),
        ),
        &Cancellation::default(),
    );
    assert!(!probe.process.status.unwrap().success());
}
#[test]
fn actual_smoke_pair_callers_select_distinct_ports_in_one_typed_batch() {
    let root = tempfile::tempdir().unwrap();
    let source = source();
    for (prefix, first, second) in [
        ("CHAIN_PORTS=", "CHAIN_PORT_1", "CHAIN_PORT_2"),
        ("OPENAI_PORTS=", "OPENAI_PORT", "OPENAI_STAGE_PORT"),
    ] {
        let mut lines = source.lines();
        let select = lines
            .find(|line| line.trim_start().starts_with(prefix))
            .unwrap();
        let read = lines.next().unwrap();
        let script = format!(
            "set -euo pipefail\n{}\n{select}\n{read}\nprintf '%s,%s' \"${first}\" \"${second}\"\n",
            selector(&source)
        );
        let mut command = spec("/bin/bash", vec!["-c".into(), script], root.path());
        command
            .environment
            .insert("ROOT".into(), Value::Public(root.path().into()));
        command.environment.insert(
            "MESH_LLM_AUTOMATION_BIN".into(),
            Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        );
        let report = raw(&command, &Cancellation::default());
        assert!(report.process.cleanup.complete && report.process.status.unwrap().success());
        let bytes = report.stdout.unwrap();
        let ports: Vec<u16> = std::str::from_utf8(bytes.as_bytes())
            .unwrap()
            .split(',')
            .map(|port| port.parse().unwrap())
            .collect();
        assert_eq!(ports.len(), 2);
        assert_ne!(ports[0], ports[1]);
        assert!(ports.iter().all(|port| *port > 0));
    }
    let command_text = format!(
        "set -euo pipefail\n{}\n{}\npick_port\n",
        selector(&source),
        function(&source, "pick_port")
    );
    let mut command = spec("/bin/bash", vec!["-c".into(), command_text], root.path());
    command
        .environment
        .insert("ROOT".into(), Value::Public(root.path().into()));
    command.environment.insert(
        "MESH_LLM_AUTOMATION_BIN".into(),
        Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
    );
    let report = raw(&command, &Cancellation::default());
    assert!(report.process.status.unwrap().success());
    assert!(
        std::str::from_utf8(report.stdout.unwrap().as_bytes())
            .unwrap()
            .trim()
            .parse::<u16>()
            .unwrap()
            > 0
    );
}
#[test]
fn actual_smoke_selector_absent_uses_just_and_configured_invalid_fails_closed() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("Justfile"), b"# fixture facade\n").unwrap();
    executable(
        &root.path().join("just"),
        "#!/bin/sh\n[ \"$1\" = --justfile ] && [ \"$2\" = \"$ROOT/Justfile\" ] && [ -f \"$2\" ] && [ \"$3\" = automation-run ] || exit 92\nprintf '%s\\0' \"$@\"\nexit 19\n",
    );
    let owner = root.path().join("frozen owner with spaces");
    executable(&owner, "#!/bin/sh\nprintf '%s\\0' \"$@\"\nexit 17\n");
    let not_exec = root.path().join("not-executable");
    fs::write(&not_exec, b"fixture").unwrap();
    let source = source();
    for configured in [
        None,
        Some(owner.to_str().unwrap()),
        Some(""),
        Some("relative"),
        Some(root.path().to_str().unwrap()),
        Some(not_exec.to_str().unwrap()),
    ] {
        let script = format!(
            "set -euo pipefail\n{}\n\"${{automation[@]}}\" automation local-ports 'with spaces' '' $'line1\\nline2'\n",
            selector(&source)
        );
        let mut command = spec("/bin/bash", vec!["-c".into(), script], root.path());
        command
            .environment
            .insert("ROOT".into(), Value::Public(root.path().into()));
        command.environment.insert(
            "PATH".into(),
            Value::Public(format!("{}:/usr/bin:/bin", root.path().display()).into()),
        );
        if let Some(value) = configured {
            command.environment.insert(
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(value.into()),
            );
        }
        let report = raw(&command, &Cancellation::default());
        assert!(report.process.cleanup.complete);
        let status = report.process.status.unwrap().code();
        if configured.is_none() {
            assert_eq!(status, Some(19));
            assert_eq!(report.stdout.unwrap().as_bytes(),format!("--justfile\0{}\0automation-run\0automation\0local-ports\0with spaces\0\0line1\nline2\0",root.path().join("Justfile").display()).as_bytes());
        } else if configured == owner.to_str() {
            assert_eq!(status, Some(17));
            assert_eq!(
                report.stdout.unwrap().as_bytes(),
                b"automation\0local-ports\0with spaces\0\0line1\nline2\0"
            );
        } else {
            assert_eq!(status, Some(1));
            assert!(report.stdout.unwrap().as_bytes().is_empty());
        }
    }
}
