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
fn caller(root: &Path, label: &str, seconds: u64, arguments: Vec<String>) -> ProcessSpec {
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../skippy/scripts/skippy-family-battery.sh"),
    )
    .unwrap();
    let start = source.find("run_battery_timeout() {").unwrap();
    let end = source[start..].find("\n}\n").unwrap() + start + 3;
    let script = root.join("battery timeout caller.sh");
    fs::write(&script,format!("set -euo pipefail\nautomation=(\"$MESH_LLM_AUTOMATION_BIN\")\nARTIFACT_DIR=\"$RUNNER_TEMP\"\n{}\nrun_battery_timeout \"$LABEL\" \"$SECONDS_LIMIT\" \"$@\"\n",&source[start..end])).unwrap();
    let mut args = vec![script.display().to_string()];
    args.extend(arguments);
    let mut command = spec("/bin/bash", args, root);
    for (key, value) in [
        ("RUNNER_TEMP", root.display().to_string()),
        (
            "MESH_LLM_AUTOMATION_BIN",
            env!("CARGO_BIN_EXE_xtask").into(),
        ),
        ("LABEL", label.into()),
        ("SECONDS_LIMIT", seconds.to_string()),
    ] {
        command
            .environment
            .insert(key.into(), Value::Public(value.into()));
    }
    command
}
#[test]
fn actual_battery_jq_timeout_caller_preserves_argument_bytes_environment_logs_and_status() {
    let root = tempfile::tempdir().unwrap();
    let arguments = vec![
        "with spaces".to_owned(),
        String::new(),
        "multiple\nlines".into(),
        "x".repeat(8192),
    ];
    for label in [
        "family certification llama split 8",
        "mmproj smoke llama",
        "workload certification fixture (embedding)",
    ] {
        let mut args=vec!["/bin/sh".into(),"-c".into(),"printf '%s\\0' \"$@\"; printf '%s\\0' \"$CANARY_TIMEOUT_FIXTURE\" \"$PWD\" >&2; exit 7".into(),"fixture".into()];
        args.extend(arguments.clone());
        let result = raw(
            &caller(root.path(), label, 2, args),
            &Cancellation::default(),
        );
        assert!(result.process.cleanup.complete);
        assert_eq!(result.process.status.unwrap().code(), Some(7));
        let expected: Vec<u8> = arguments
            .iter()
            .flat_map(|arg| arg.bytes().chain(std::iter::once(0)))
            .collect();
        assert_eq!(result.stdout.unwrap().as_bytes(), expected);
        assert_eq!(
            result.stderr.unwrap().as_bytes(),
            format!(
                "space and\nnewline\0{}\0",
                root.path().canonicalize().unwrap().display()
            )
            .as_bytes()
        );
        assert!(!fs::read_dir(root.path()).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("command-timeout.")
        }));
    }
}
#[test]
fn actual_battery_timeout_caller_retains_native_signal_deadline_and_infrastructure_codes() {
    let root = tempfile::tempdir().unwrap();
    for (script, code) in [
        ("kill -TERM $$", 143),
        ("/bin/sleep 30 & echo $! > descendant; wait", 124),
    ] {
        let result = raw(
            &caller(
                root.path(),
                "battery label",
                1,
                vec!["/bin/sh".into(), "-c".into(), script.into()],
            ),
            &Cancellation::default(),
        );
        assert!(result.process.cleanup.complete);
        assert_eq!(result.process.status.unwrap().code(), Some(code));
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
                "owned descendant survived timeout"
            );

            assert!(
                String::from_utf8_lossy(result.stderr.unwrap().as_bytes())
                    .contains("battery label timed out after 1s")
            );
        }
    }
    let result = raw(
        &caller(
            root.path(),
            "invalid deadline",
            0,
            vec!["/bin/sh".into(), "-c".into(), "echo should-not-run".into()],
        ),
        &Cancellation::default(),
    );
    assert_eq!(result.process.status.unwrap().code(), Some(125));
    assert!(result.stdout.unwrap().as_bytes().is_empty());
}

#[test]
fn actual_workload_dimensions_caller_consumes_typed_selected_file_projection() {
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }
    let root = tempfile::tempdir().unwrap();
    let mut gguf = b"GGUF".to_vec();
    gguf.extend(3u32.to_le_bytes());
    gguf.extend(0u64.to_le_bytes());
    gguf.extend(3u64.to_le_bytes());
    string(&mut gguf, "general.architecture");
    gguf.extend(8u32.to_le_bytes());
    string(&mut gguf, "llama");
    for (key, value) in [
        ("llama.block_count", 16u64),
        ("llama.embedding_length", 2048),
    ] {
        string(&mut gguf, key);
        gguf.extend(10u32.to_le_bytes());
        gguf.extend(value.to_le_bytes());
    }
    let model = root.path().join("selected model.gguf");
    fs::write(&model, &gguf).unwrap();
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../skippy/scripts/skippy-workload-certify.sh"),
    )
    .unwrap();
    let line = source
        .lines()
        .find(|line| line.starts_with("DIMENSIONS="))
        .unwrap();
    let script = root.path().join("dimensions.sh");
    fs::write(&script, format!("set -euo pipefail\nworkload_automation=(\"$OWNER\")\n{line}\nprintf '%s' \"$DIMENSIONS\"\n")).unwrap();
    let mut command = spec("/bin/bash", vec![script.display().to_string()], root.path());
    command.environment.insert(
        "OWNER".into(),
        Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
    );
    command
        .environment
        .insert("MODEL_PATH".into(), Value::Public(model.clone().into()));
    let result = raw(&command, &Cancellation::default());
    assert!(result.process.cleanup.complete && result.process.status.unwrap().success());
    let value: serde_json::Value =
        serde_json::from_slice(result.stdout.unwrap().as_bytes()).unwrap();
    assert_eq!(value["layer_count"], 16);
    assert_eq!(value["activation_width"], 2048);
    fs::write(model, b"invalid model").unwrap();
    let result = raw(&command, &Cancellation::default());
    assert!(!result.process.status.unwrap().success());
    assert!(result.stdout.unwrap().as_bytes().is_empty());
}
