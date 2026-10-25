//! Actual observation CLI against supplied files only; no receiver or inference.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, path::PathBuf, time::Duration};
#[test]
fn handoff_summary_actual_cli_renders_supplied_observation_and_refuses_partial_success() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("send-2.json");
    let report = serde_json::json!({"prompt_token_count":2,"state_bytes":1048576,"transfer_gbps":1.25,"source_prefill_ms":200,"state_export_ms":3,"transfer_ms":4,"receiver":{"kv_attach_ms":5},"ttft_disaggregated_ms":212,"matches":false});
    std::fs::write(&path, serde_json::to_vec(&report).unwrap()).unwrap();
    let spec = ProcessSpec {
        executable: PathBuf::from(env!("CARGO_BIN_EXE_xtask")),
        arguments: vec![
            Value::Public("automation".into()),
            Value::Public("remote-handoff-summary".into()),
            Value::Public("--reports-dir".into()),
            Value::Public(root.path().as_os_str().into()),
        ],
        cwd: root.path().to_owned(),
        environment: BTreeMap::new(),
    };
    let limits = Limits {
        execution: Duration::from_secs(5),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    for valid in [true, false] {
        if !valid {
            std::fs::write(root.path().join("send-bad.json"), b"invalid").unwrap();
        }
        let result = process::supervise_raw(
            &spec,
            &limits,
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        let p = &result.process;
        assert!(
            p.failure.is_none()
                && p.cleanup.complete
                && !p.cleanup.forced
                && !p.cleanup.graceful_signal_failed
                && p.cleanup.failure.is_none()
        );
        for (stream, raw) in [
            (&p.stdout, result.stdout.as_ref().unwrap()),
            (&p.stderr, result.stderr.as_ref().unwrap()),
        ] {
            assert!(stream.line_capture_complete && !stream.truncated);
            assert_eq!(stream.bytes_seen, raw.as_bytes().len() as u64);
        }
        assert_eq!(p.status.as_ref().unwrap().success(), valid);
        let text = std::str::from_utf8(result.stdout.as_ref().unwrap().as_bytes()).unwrap();
        if valid {
            assert_eq!(
                text.lines()
                    .nth(1)
                    .unwrap()
                    .split_whitespace()
                    .collect::<Vec<_>>(),
                [
                    "2", "1.0", "1.25", "200", "3", "4", "5", "212", "-", "-", "False"
                ]
            );
        } else {
            assert!(text.is_empty());
        }
    }
    root.close().unwrap();
}

use std::{fs, os::unix::fs::PermissionsExt, path::Path};
fn finite_sender(root: &Path) -> PathBuf {
    let tools = root.join("tools");
    fs::create_dir(&tools).unwrap();
    let sender = root.join("finite sender");
    fs::write(
        &sender,
        r#"#!/bin/bash
set -euo pipefail
[[ $# == 20 ]]
[[ "$1" == remote-handoff && "$2" == --role && "$3" == send ]]
[[ "$4" == --peer && "$5" == private-receiver:19081 ]]
[[ "$6" == --model && "$7" == 'private model.gguf' ]]
[[ "$8" == --layer-end && "$9" == 12 ]]
[[ "${10}" == --ctx-size && "${11}" == 16384 ]]
[[ "${12}" == --n-gpu-layers && "${13}" == 99 ]]
[[ "${14}" == --prefix-token-count && "${16}" == --decode-tokens && "${17}" == 32 ]]
[[ "${18}" == --baseline && "${19}" == --report-out ]]
prefix="${15}"
printf '%s\0' "$@" > "$FIXTURE_ROOT/args-${prefix}"
if [[ "$prefix" == 10 ]]; then echo 'finite sender failed' >&2; exit 23; fi
cat "$FIXTURE_ROOT/source-${prefix}.json" > "${20}"
echo 'finite sender observed'
"#,
    )
    .unwrap();
    fs::set_permissions(&sender, fs::Permissions::from_mode(0o700)).unwrap();
    let python = tools.join("python3");
    fs::write(
        &python,
        "#!/bin/bash\nprintf called > \"$FIXTURE_ROOT/python.called\"\nexit 99\n",
    )
    .unwrap();
    fs::set_permissions(python, fs::Permissions::from_mode(0o700)).unwrap();
    sender
}
fn supplied_reports(root: &Path, prefixes: &[u64]) {
    for &prefix in prefixes.iter().filter(|&&p| p != 10) {
        let report = serde_json::json!({"prompt_token_count":prefix,"state_bytes":1048576,"transfer_gbps":1.25,"source_prefill_ms":200,"state_export_ms":3,"transfer_ms":4,"receiver":{"kv_attach_ms":5},"ttft_disaggregated_ms":212,"matches":prefix!=7});
        fs::write(
            root.join(format!("source-{prefix}.json")),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
    }
}
fn wrapper_process(
    root: &Path,
    prefixes: &[u64],
    default_prefixes: bool,
) -> process::RawProcessReport {
    let sender = finite_sender(root);
    let tools = root.join("tools");
    let wrapper = root.join("wrapper.sh");
    fs::write(
        &wrapper,
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../scripts/remote-handoff-sweep.sh"
        )),
    )
    .unwrap();
    let out = root.join("reports");
    let mut args = vec![
        wrapper.into_os_string(),
        "private-receiver:19081".into(),
        "private model.gguf".into(),
        "12".into(),
        out.clone().into_os_string(),
    ];
    if !default_prefixes {
        args.extend(prefixes.iter().map(|p| p.to_string().into()));
    }
    process::supervise_raw(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.to_path_buf(),
            arguments: args.into_iter().map(Value::Public).collect(),
            environment: BTreeMap::from([
                (
                    "PATH".into(),
                    Value::Public(format!("{}:/usr/bin:/bin", tools.display()).into()),
                ),
                ("BIN".into(), Value::Public(sender.into_os_string())),
                (
                    "FIXTURE_ROOT".into(),
                    Value::Public(root.as_os_str().to_owned()),
                ),
                (
                    "MESH_LLM_AUTOMATION_BIN".into(),
                    Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
                ),
            ]),
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
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}
fn assert_wrapper(
    result: &process::RawProcessReport,
    root: &Path,
    prefixes: &[u64],
    default_prefixes: bool,
) {
    let out = root.join("reports");
    let p = &result.process;
    assert_eq!(p.outcome, process::Outcome::Exited);
    assert!(p.status.as_ref().unwrap().success());
    assert!(
        p.failure.is_none()
            && p.cleanup.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
    );
    for (stream, raw) in [
        (&p.stdout, result.stdout.as_ref().unwrap()),
        (&p.stderr, result.stderr.as_ref().unwrap()),
    ] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(stream.bytes_seen, raw.as_bytes().len() as u64);
    }
    let text = std::str::from_utf8(result.stdout.as_ref().unwrap().as_bytes()).unwrap();
    let table = text
        .lines()
        .skip_while(|line| !line.contains("ttft-pd"))
        .skip(1)
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let expected: Vec<_> = prefixes.iter().copied().filter(|&p| p != 10).collect();
    assert_eq!(
        table
            .iter()
            .map(|row| row[0].parse::<u64>().unwrap())
            .collect::<Vec<_>>(),
        expected
    );
    for row in &table {
        assert_eq!(
            &row[1..10],
            &["1.0", "1.25", "200", "3", "4", "5", "212", "-", "-"]
        );
        assert_eq!(row[10], if row[0] == "7" { "False" } else { "True" });
    }
    for &prefix in prefixes {
        let observed = fs::read(root.join(format!("args-{prefix}"))).unwrap();
        let path = out.join(format!("send-{prefix}.json"));
        let expected = [
            "remote-handoff",
            "--role",
            "send",
            "--peer",
            "private-receiver:19081",
            "--model",
            "private model.gguf",
            "--layer-end",
            "12",
            "--ctx-size",
            "16384",
            "--n-gpu-layers",
            "99",
            "--prefix-token-count",
            &prefix.to_string(),
            "--decode-tokens",
            "32",
            "--baseline",
            "--report-out",
            path.to_str().unwrap(),
            "",
        ]
        .join("\0");
        assert_eq!(observed, expected.as_bytes());
        let log = fs::read_to_string(out.join(format!("send-{prefix}.log"))).unwrap();
        assert!(log.contains(if prefix == 10 {
            "finite sender failed"
        } else {
            "finite sender observed"
        }));
    }
    if !default_prefixes {
        assert!(text.contains("prefix 10 FAILED"));
        assert!(!out.join("send-10.json").exists());
    }
    assert!(!root.join("python.called").exists());
}
#[test]
fn handoff_summary_whole_wrapper_preserves_sender_argv_fail_logs_defaults_and_table() {
    for default_prefixes in [true, false] {
        let owned = tempfile::tempdir().unwrap();
        let root = owned.path().canonicalize().unwrap();
        let prefixes: &[u64] = if default_prefixes {
            &[512, 2048, 4096, 8192]
        } else {
            &[10, 2, 7]
        };
        supplied_reports(&root, prefixes);
        let result = wrapper_process(&root, prefixes, default_prefixes);
        assert_wrapper(&result, &root, prefixes, default_prefixes);
        owned.close().unwrap();
    }
}
