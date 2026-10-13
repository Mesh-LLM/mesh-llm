use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use std::{
    collections::BTreeMap,
    fs,
    net::{Ipv4Addr, TcpListener},
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
fn cli(root: &Path, args: Vec<String>, cancellation: &Cancellation) -> process::RawProcessReport {
    let spec = ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: ["automation".to_owned(), "manual-smoke".into()]
            .into_iter()
            .chain(args)
            .map(|value| Value::Public(value.into()))
            .collect(),
        cwd: root.into(),
        environment: ["PATH", "SYSTEMROOT", "WINDIR"]
            .into_iter()
            .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
            .collect::<BTreeMap<_, _>>(),
    };
    let limits = Limits {
        execution: Duration::from_secs(12),
        graceful_shutdown: Duration::from_secs(4),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        cancellation,
        RawCaptureOptions {
            stdout: NonZeroUsize::new(1024 * 1024),
            stderr: NonZeroUsize::new(1024 * 1024),
        },
    )
    .unwrap();
    assert!(
        report.process.cleanup.complete
            && !report.process.cleanup.forced
            && !report.process.cleanup.graceful_signal_failed
            && report.process.cleanup.failure.is_none()
            && report.process.failure.is_none(),
        "{report:?}"
    );
    for (stream, raw) in [
        (&report.process.stdout, report.stdout.as_ref()),
        (&report.process.stderr, report.stderr.as_ref()),
    ] {
        assert!(
            !stream.truncated && stream.line_capture_complete,
            "{report:?}"
        );
        assert_eq!(stream.bytes_seen, raw.unwrap().as_bytes().len() as u64);
    }
    report
}
fn coverage(root: &Path, rows: &str, matrix: &str) -> process::RawProcessReport {
    fs::write(root.join("matrix.md"), matrix).unwrap();
    fs::write(root.join("manifest.tsv"), rows).unwrap();
    cli(
        root,
        [
            "coverage",
            "--matrix",
            "matrix.md",
            "--manifest",
            "manifest.tsv",
            "--required-evidence",
            "evidence.txt",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        &Cancellation::default(),
    )
}
const HEADER: &str = "key_path\tfixture_path\tstartup_apply_command\tmodel_identifier\tverification_command\texpected_result\tactual_evidence_path\tpass_fail_status\n";
fn row(key: &str, status: &str) -> String {
    format!(
        "{key}\tfixture.toml\towner --model-path $MESH_LLM_SMOKE_MODEL_PATH --draft-path $MESH_LLM_SMOKE_DRAFT_PATH\tlocal-model\tHTTP status/models/chat\tmeasured or blocked explicitly\tevidence.txt\t{status}\n"
    )
}
#[test]
fn coverage_admits_complete_portable_pass_and_blocked_rows_with_real_evidence() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("fixture.toml"), "version = 1").unwrap();
    fs::write(
        root.path().join("evidence.txt"),
        "COMMAND: inert fixture; Coverage check\n",
    )
    .unwrap();
    let rows = format!(
        "{HEADER}{}{}",
        row("speculative.enabled", "PASS"),
        row("multimodal.mmproj", "BLOCKED_ASSET_LOCAL")
    );
    let result = coverage(
        root.path(),
        &rows,
        "| A | B | `speculative.enabled`<br>`multimodal.mmproj` |\n",
    );
    assert!(result.process.success(), "{result:?}");
    assert!(
        std::str::from_utf8(result.stdout.unwrap().as_bytes())
            .unwrap()
            .contains("2 matrix keys")
    );
    root.close().unwrap();
}
#[test]
fn coverage_refuses_duplicate_missing_extra_nonportable_and_unproven_evidence() {
    for attack in [
        "duplicate",
        "missing",
        "extra",
        "status",
        "local",
        "placeholder",
        "model-env",
        "draft-env",
        "non-executable",
        "empty",
        "markers",
        "model-id-env",
    ] {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("fixture.toml"), "version = 1").unwrap();
        fs::write(
            root.path().join("evidence.txt"),
            "COMMAND: actual evidence\n",
        )
        .unwrap();
        let mut rows = format!("{HEADER}{}", row("speculative.enabled", "PASS"));
        match attack {
            "duplicate" => rows.push_str(&row("speculative.enabled", "PASS")),
            "missing" => rows = HEADER.into(),
            "extra" => rows.push_str(&row("extra.key", "PASS")),
            "status" => rows = rows.replace("\tPASS", "\tUNKNOWN"),
            "local" => rows = rows.replace("local-model", "/Users/operator/model"),
            "placeholder" => rows = rows.replace("local-model", "<model>"),
            "model-env" => {
                rows = rows.replace(
                    "--model-path $MESH_LLM_SMOKE_MODEL_PATH",
                    "--model-path /tmp/model",
                )
            }
            "draft-env" => {
                rows = rows.replace(
                    "--draft-path $MESH_LLM_SMOKE_DRAFT_PATH",
                    "--draft-path /tmp/draft",
                )
            }
            "non-executable" => rows = rows.replace("owner ", "not-run locally "),
            "empty" => fs::write(root.path().join("evidence.txt"), "").unwrap(),
            "markers" => fs::write(root.path().join("evidence.txt"), "unsupported claim").unwrap(),
            "model-id-env" => {
                rows = rows
                    .replace("local-model", "$MESH_LLM_SMOKE_MODEL_PATH")
                    .replace("\tPASS", "\tBLOCKED_NETWORK_LOCAL");
            }
            _ => unreachable!(),
        }
        let result = coverage(root.path(), &rows, "| A | B | `speculative.enabled` |\n");
        assert_eq!(
            result.process.status.unwrap().code(),
            Some(1),
            "{attack}: {result:?}"
        );
        root.close().unwrap();
    }
}
fn runtime_args(root: &Path, mode: &str) -> Vec<String> {
    fs::write(root.join("fixture.toml"),format!("version = 1\nfixture_marker = 'manual-smoke-fixture {mode}'\n[[models]]\nmodel = 'inert-model'\n[models.hardware]\nmodel_path = '__LOCAL_MODEL_PATH__'\ndraft_path = '__LOCAL_INCOMPATIBLE_DRAFT_PATH__'\n")).unwrap();
    fs::write(root.join("model quote\".gguf"), "inert bytes").unwrap();
    let sockets = [
        TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap(),
        TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap(),
    ];
    let ports = sockets
        .iter()
        .map(|socket| socket.local_addr().unwrap().port().to_string())
        .collect::<Vec<_>>();
    drop(sockets);
    let example = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples/l7_daemon_fixture");
    assert!(example.is_file(), "existing fixture example prerequisite");
    vec![
        "run".into(),
        "--binary".into(),
        example.to_str().unwrap().into(),
        "--fixture".into(),
        root.join("fixture.toml").to_str().unwrap().into(),
        "--model-path".into(),
        root.join("model quote\".gguf").to_str().unwrap().into(),
        "--native-runtime-root".into(),
        root.to_str().unwrap().into(),
        "--output".into(),
        root.join("evidence").to_str().unwrap().into(),
        "--api-port".into(),
        ports[0].clone(),
        "--console-port".into(),
        ports[1].clone(),
        "--max-wait".into(),
        "2".into(),
    ]
}
fn no_state(root: &Path) {
    assert!(fs::read_dir(root).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("manual-smoke.")
    }));
}
#[test]
fn runtime_executes_rewritten_config_status_models_chat_and_graceful_owned_cleanup() {
    let root = tempfile::tempdir().unwrap();
    let args = runtime_args(root.path(), "success");
    let result = cli(root.path(), args, &Cancellation::default());
    assert!(result.process.success(), "{result:?}");
    let receipt: Json =
        serde_json::from_slice(&fs::read(root.path().join("evidence/receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["status"], "PASS");
    assert_eq!(receipt["http"]["Ok"]["STATUS_JSON"]["llama_ready"], true);
    let config: toml::Value =
        toml::from_str(&fs::read_to_string(root.path().join("applied.toml")).unwrap()).unwrap();
    let model = root
        .path()
        .join("model quote\".gguf")
        .canonicalize()
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        config["models"][0]["hardware"]["model_path"].as_str(),
        Some(model.as_str())
    );
    assert_eq!(
        config["models"][0]["hardware"]["draft_path"].as_str(),
        Some(model.as_str())
    );
    let request: Json =
        serde_json::from_slice(&fs::read(root.path().join("request.json")).unwrap()).unwrap();
    assert_eq!(
        request,
        json!({"model":"inert-model","messages":[{"role":"user","content":"Say hello in exactly three words."}],"max_tokens":4,"temperature":0})
    );
    assert!(root.path().join("stopped").is_file());
    no_state(root.path());
    root.close().unwrap();
}
#[test]
fn runtime_refuses_early_exit_and_http_error_without_pass_receipt_or_state_leak() {
    for mode in ["early-exit", "error-chat"] {
        let root = tempfile::tempdir().unwrap();
        let args = runtime_args(root.path(), mode);
        let result = cli(root.path(), args, &Cancellation::default());
        assert_eq!(result.process.status.unwrap().code(), Some(1), "{result:?}");
        let receipt: Json =
            serde_json::from_slice(&fs::read(root.path().join("evidence/receipt.json")).unwrap())
                .unwrap();
        assert_eq!(receipt["status"], "FAILED");
        no_state(root.path());
        root.close().unwrap();
    }
}
#[test]
fn held_chat_cancellation_stops_server_before_temporary_state_deletion() {
    let root = tempfile::tempdir().unwrap();
    let args = runtime_args(root.path(), "held-chat");
    let cancel = Cancellation::default();
    std::thread::scope(|scope| {
        let token = cancel.clone();
        let marker = root.path().join("request.json");
        let observer = scope.spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(8);
            while !marker.exists() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            let observed = marker.exists();
            token.cancel();
            observed
        });
        let result = cli(root.path(), args, &cancel);
        let observed = observer.join().unwrap();
        assert!(observed, "actual POST causal barrier");
        assert_eq!(result.process.outcome, process::Outcome::Cancelled);
        assert!(root.path().join("stopped").is_file());
        assert!(root.path().join("evidence/receipt.json").is_file());
        no_state(root.path());
    });
    root.close().unwrap();
}
#[test]
fn coverage_static_fifo_evidence_is_refused_before_open_without_writer() {
    use std::os::unix::ffi::OsStrExt as _;
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("fixture.toml"), "version = 1").unwrap();
    let path = root.path().join("evidence.txt");
    let path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let result = coverage(
        root.path(),
        &format!("{HEADER}{}", row("key", "PASS")),
        "| A | B | `key` |\n",
    );
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    assert!(
        std::str::from_utf8(result.stderr.unwrap().as_bytes())
            .unwrap()
            .contains("regular file")
    );
    root.close().unwrap();
}
