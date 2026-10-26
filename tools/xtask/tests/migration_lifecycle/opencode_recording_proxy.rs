//! Actual OpenCode background proxy block, request capture and EXIT cleanup.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::TcpListener,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    thread,
    time::{Duration, Instant},
};
const SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/ci-opencode-smoke.sh"
));

#[test]
fn actual_adapter_starts_native_proxy_captures_upstream_and_reaps_it_on_exit_without_python() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    fs::create_dir(root.join("scripts")).unwrap();
    fs::create_dir(root.join("bin")).unwrap();
    fs::write(
        root.join("openai-surface-proxy.ready"),
        "http://127.0.0.1:1/v1",
    )
    .unwrap();
    fs::write(root.join("capture.jsonl"), "stale capture from prior run\n").unwrap();
    let tripwire = root.join("bin/python3");
    fs::write(
        &tripwire,
        "#!/bin/sh\nprintf 'unexpected Python\\n' >> \"$TRACE\"\nexit 92\n",
    )
    .unwrap();
    fs::set_permissions(&tripwire, fs::Permissions::from_mode(0o700)).unwrap();
    let selector = SOURCE
        .split_once("# Frozen automation selection ends.")
        .unwrap()
        .0;
    let start = SOURCE.find("CONFIG_BASE_URL=\"$MESH_BASE_URL\"").unwrap();
    let end = SOURCE[start..].find("prepare_opencode_config()").unwrap() + start;
    let block = &SOURCE[start..end];
    assert!(block.contains("automation agent-recording-proxy"));
    fs::write(root.join("scripts/proxy.sh"), format!("{selector}\n{block}\ncurl --silent --show-error --max-time 5 --noproxy '*' \"${{CONFIG_BASE_URL%/}}/models\" >\"$WORK_DIR/client.json\"\n")).unwrap();
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", upstream.local_addr().unwrap());
    let worker = thread::spawn(move || {
        upstream.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(8);
        let mut socket = loop {
            match upstream.accept() {
                Ok((socket, _)) => break socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("{error}"),
            }
        };
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut request = Vec::new();
        let mut chunk = [0; 4096];
        while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
            let count = socket.read(&mut chunk).unwrap();
            assert!(count > 0 && request.len() < 65536);
            request.extend_from_slice(&chunk[..count]);
        }
        assert!(request.starts_with(b"GET /v1/models HTTP/1.1"));
        let body = br#"{"data":[]}"#;
        write!(socket, "HTTP/1.1 200 fixture\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n", body.len()).unwrap();
        socket.write_all(body).unwrap();
    });
    let inherited = std::env::var_os("PATH").unwrap();
    let mut paths = vec![root.join("bin")];
    paths.extend(std::env::split_paths(&inherited));
    let mut environment = BTreeMap::from([
        (
            "PATH".into(),
            Value::Public(std::env::join_paths(paths).unwrap()),
        ),
        (
            "MESH_LLM_AUTOMATION_BIN".into(),
            Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
        ("MESH_BASE_URL".into(), Value::Public(base.into())),
        ("MODEL".into(), Value::Public("mesh/fixture".into())),
        ("SURFACE_CAPTURE".into(), Value::Public("true".into())),
        (
            "WORK_DIR".into(),
            Value::Public(root.clone().into_os_string()),
        ),
        (
            "SURFACE_LOG".into(),
            Value::Public(root.join("capture.jsonl").into_os_string()),
        ),
        (
            "SURFACE_PROXY_LOG".into(),
            Value::Public(root.join("proxy.log").into_os_string()),
        ),
        (
            "TRACE".into(),
            Value::Public(root.join("trace").into_os_string()),
        ),
        ("NO_PROXY".into(), Value::Public("*".into())),
        ("no_proxy".into(), Value::Public("*".into())),
    ]);
    if let Some(home) = std::env::var_os("HOME") {
        environment.insert("HOME".into(), Value::Public(home));
    }
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            arguments: vec![Value::Public(
                root.join("scripts/proxy.sh").into_os_string(),
            )],
            cwd: root.clone(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(20),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 16384,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(16384),
            stderr: NonZeroUsize::new(16384),
        },
    )
    .unwrap();
    assert_eq!(report.process.outcome, Outcome::Exited);
    assert!(
        report.process.success() && report.process.cleanup.complete,
        "{report:?}"
    );
    worker.join().unwrap();
    assert!(!root.join("trace").exists());
    assert!(!root.join("openai-surface-proxy.ready").exists());
    let response: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("client.json")).unwrap()).unwrap();
    assert_eq!(response, serde_json::json!({"data":[]}));
    let record: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("capture.jsonl")).unwrap()).unwrap();
    assert_eq!(record["method"], "GET");
    assert_eq!(record["path"], "/v1/models");
    assert_eq!(record["body"], serde_json::Value::Null);
}
