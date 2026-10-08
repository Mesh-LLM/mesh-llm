//! Fixed9337 actual native caller with an owned inert serve process. No model inference.
use crate::process;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn quote(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
}
fn model(path: &Path) {
    let string = |out: &mut Vec<u8>, text: &str| {
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(text.as_bytes());
    };
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(4_u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, "llama");
    for (key, value) in [
        ("llama.context_length", 512_u32),
        ("llama.block_count", 4),
        ("llama.embedding_length", 16),
    ] {
        string(&mut bytes, key);
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

fn wrapper(path: &Path, mode: &str, record: &Path) {
    use std::os::unix::fs::PermissionsExt as _;
    let script = format!(
        "#!/bin/bash\nset -eu\npid=''\ntrap 'if [[ -n \"$pid\" ]]; then wait \"$pid\" || true; fi; exit 0' TERM INT\n[[ \"$#\" == 5 && \"$1\" == serve && \"$2\" == --model && -f \"$3\" && \"$4\" == --log-format && \"$5\" == json ]] || exit 64\nexport KV_FIXTURE_BINARY=\"$0\"\nexport KV_FIXTURE_MODEL=\"$3\"\nexport KV_FIXTURE_MODE='{mode}'\nexport KV_FIXTURE_RECORD={}\n{} --exact kv_restart_full_cli::owned_default_server --ignored --nocapture &\npid=$!\nwait \"$pid\"\n",
        quote(record),
        quote(&std::env::current_exe().unwrap())
    );
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}
fn fixture(mode: &str) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let binary = root.path().join("host");
    let modelpath = root.path().join("model.gguf");
    model(&modelpath);
    wrapper(&binary, mode, &root.path().join("observations.jsonl"));
    let input = json!({"schema_version":1,"binary":binary,"model":modelpath,"turns":2,"turn_target_tokens":16,"system_tokens":16,"restore_repeats":3,"max_output_tokens":8,"request_timeout_secs":2.0,"ready_timeout_secs":2,"worker_timeout_secs":5,"timeout_secs":45});
    std::fs::write(
        root.path().join("input.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    root
}
fn invoke(root: PathBuf, cancel: process::Cancellation) -> process::RawProcessReport {
    process::supervise_raw(
        &process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: ["automation", "waiting-prefix", "kv-restart-run", "--input"]
                .into_iter()
                .map(|v| process::Value::Public(v.into()))
                .chain([
                    process::Value::Public(root.join("input.json").into_os_string()),
                    process::Value::Public("--output-directory".into()),
                    process::Value::Public(root.join("output").into_os_string()),
                ])
                .collect(),
            cwd: root,
            environment: BTreeMap::new(),
        },
        &process::Limits {
            execution: Duration::from_secs(25),
            graceful_shutdown: Duration::from_secs(10),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &cancel,
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}
fn cleanup(raw: &process::RawProcessReport) {
    let p = &raw.process;
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none(),
        "{p:?}"
    );
    assert!(p.stdout.line_capture_complete && p.stderr.line_capture_complete);
    assert_eq!(
        raw.stdout.as_ref().unwrap().as_bytes().len() as u64,
        p.stdout.bytes_seen
    );
    assert_eq!(
        raw.stderr.as_ref().unwrap().as_bytes().len() as u64,
        p.stderr.bytes_seen
    );
}
fn report(root: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(root.join("output/run.json")).unwrap()).unwrap()
}
fn observations(root: &Path) -> Vec<Value> {
    std::fs::read_to_string(root.join("observations.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}
fn endpoint_free() {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.set_reuseaddr(true).unwrap();
    socket
        .bind("127.0.0.1:9337".parse().unwrap())
        .expect("fixed9337 must be free; refusal is not qualification");
}
#[test]
fn kv_restart_full_cli_restores_same_state_and_exact_full_transcript() {
    let _serial = SERIAL.lock().unwrap();
    endpoint_free();
    let root = fixture("success");
    let binary_link = root.path().join("host-link");
    let model_link = root.path().join("model-link.gguf");
    std::os::unix::fs::symlink(root.path().join("host"), &binary_link).unwrap();
    std::os::unix::fs::symlink(root.path().join("model.gguf"), &model_link).unwrap();
    let mut input: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("input.json")).unwrap()).unwrap();
    input["binary"] = json!(binary_link);
    input["model"] = json!(model_link);
    std::fs::write(
        root.path().join("input.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    let raw = invoke(root.path().into(), process::Cancellation::default());
    cleanup(&raw);
    assert!(raw.process.success(), "{raw:?}");
    let run = report(root.path());
    assert!(run["error"].is_null(), "{run}");
    assert_eq!(run["terminal_complete"], true);
    assert!(run["terminal_error"].is_null());
    let start = run["started_at_unix_seconds"].as_u64().unwrap();
    let end = run["completed_at_unix_seconds"].as_u64().unwrap();
    assert!(start > 0 && end > 0);
    let rows = run["requests"].as_array().unwrap();
    assert_eq!(
        rows.iter()
            .map(|r| r["request_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["fill-1", "fill-2", "restore-1", "warm-1", "warm-2"]
    );
    assert!(
        rows.iter()
            .all(|r| r["error"].is_null() && r["messages_end_role"] == "user")
    );
    for row in &rows[2..] {
        assert_eq!(row["prompt_sha256"], rows[1]["prompt_sha256"]);
        assert_eq!(row["content_sha256"], rows[1]["content_sha256"]);
        assert_eq!(row["cached_tokens"], 5);
    }
    assert!(run["cohorts"][1]["ttft_p95_seconds"].is_null());
    assert_eq!(run["binary"]["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(
        run["runtime_policy"],
        "host-default-loader-authority-not-package-loaded-attestation"
    );
    assert!(run["restart"]["restart_to_ready_seconds"].as_f64().unwrap() >= 0.0);
    let seen = observations(root.path());
    let launches = seen
        .iter()
        .filter(|r| r["event"] == "launch")
        .collect::<Vec<_>>();
    assert_eq!(launches.len(), 2);
    for launch in &launches {
        assert_eq!(
            launch["binary_argv"],
            json!(root.path().join("host").canonicalize().unwrap())
        );
        assert_eq!(
            launch["model_argv"],
            json!(root.path().join("model.gguf").canonicalize().unwrap())
        );
    }
    assert_eq!(launches[0]["home"], launches[1]["home"]);
    assert_eq!(launches[1]["previous_clean_stop"], true);
    let requests = seen
        .iter()
        .filter(|r| r["event"] == "request")
        .collect::<Vec<_>>();
    assert_eq!(requests.len(), 5);
    for row in &requests[2..] {
        assert_eq!(row["messages"], requests[1]["messages"]);
    }
    assert_eq!(requests[0]["messages"].as_array().unwrap().len(), 2);
    assert_eq!(requests[1]["messages"].as_array().unwrap().len(), 4);
    for phase in ["fill", "replay"] {
        let life: Value = serde_json::from_slice(
            &std::fs::read(root.path().join(format!("output/{phase}/lifecycle.json"))).unwrap(),
        )
        .unwrap();
        assert!(
            life["members"]
                .as_array()
                .unwrap()
                .iter()
                .all(|r| r["cleanup_complete"] == true
                    && r["forced"] == false
                    && r["graceful_signal_failed"] == false
                    && r["cleanup_failure"].is_null()
                    && r["process_failure"].is_null()
                    && r["stdout_complete"] == true
                    && r["stderr_complete"] == true)
        );
    }
    assert!(root.path().join("output/report.md").is_file());
    endpoint_free();
    root.close().unwrap();
}
#[test]
fn kv_restart_full_cli_baseline_and_final_shutdown_refusals_retain_partial_rows() {
    let _serial = SERIAL.lock().unwrap();
    for mode in [
        "mismatch",
        "shutdown-failure",
        "truncated",
        "missing-prompt",
        "missing-cache",
    ] {
        endpoint_free();
        let root = fixture(mode);
        let raw = invoke(root.path().into(), process::Cancellation::default());
        cleanup(&raw);
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        let run = report(root.path());
        assert!(!run["error"].is_null());
        assert_eq!(run["terminal_complete"], false);
        let rows = run["requests"].as_array().unwrap();
        assert!(rows.len() >= 3);
        if mode == "mismatch" {
            assert!(rows[2]["error"].as_str().unwrap().contains("baseline"));
        } else if mode == "shutdown-failure" {
            assert!(rows.iter().all(|r| r["error"].is_null()));
            let members = run["sessions"][1]["members"].as_array().unwrap();
            assert!(members.iter().any(|m| m["forced"] == true));
        } else {
            assert!(!rows[2]["error"].is_null());
            assert!(rows[2].get("prompt_tokens").is_none());
        }
        endpoint_free();
        root.close().unwrap();
    }
}
#[test]
fn kv_restart_full_cli_occupied_and_nonempty_output_refuse_before_launch_or_mutation() {
    let _serial = SERIAL.lock().unwrap();
    endpoint_free();
    let listener = std::net::TcpListener::bind("127.0.0.1:9337").unwrap();
    let root = fixture("success");
    let raw = invoke(root.path().into(), process::Cancellation::default());
    cleanup(&raw);
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    assert!(!root.path().join("output").exists());
    assert!(!root.path().join("observations.jsonl").exists());
    listener.set_nonblocking(true).unwrap();
    assert!(matches!(listener.accept(),Err(e)if e.kind()==std::io::ErrorKind::WouldBlock));
    drop(listener);
    std::fs::create_dir(root.path().join("output")).unwrap();
    std::fs::write(root.path().join("output/keep"), b"sentinel").unwrap();
    let raw = invoke(root.path().into(), process::Cancellation::default());
    cleanup(&raw);
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    assert_eq!(
        std::fs::read(root.path().join("output/keep")).unwrap(),
        b"sentinel"
    );
    assert!(!root.path().join("observations.jsonl").exists());
    root.close().unwrap();
}
#[test]
fn kv_restart_full_cli_replay_inflight_cancel_retains_fill_and_owned_cleanup() {
    let _serial = SERIAL.lock().unwrap();
    endpoint_free();
    let root = fixture("hold");
    let cancellation = process::Cancellation::default();
    let scope = cancellation.clone();
    let path = root.path().to_path_buf();
    let worker = std::thread::spawn(move || invoke(path, scope));
    let until = std::time::Instant::now() + Duration::from_secs(15);
    while !root.path().join("held").exists()
        && std::time::Instant::now() < until
        && !worker.is_finished()
    {
        std::thread::park_timeout(Duration::from_millis(5));
    }
    let marker_seen = root.path().join("held").exists();
    cancellation.cancel();
    let raw = worker.join().unwrap();
    cleanup(&raw);
    assert!(
        marker_seen,
        "replay never held; supervisor was cancelled and joined before assertion"
    );
    assert!(cancellation.is_cancelled());
    let run = report(root.path());
    assert!(!run["error"].is_null());
    assert_eq!(run["terminal_complete"], false);
    assert!(run["requests"].as_array().unwrap().len() >= 2);
    assert!(run["requests"][0]["error"].is_null() && run["requests"][1]["error"].is_null());
    endpoint_free();
    root.close().unwrap();
}
async fn request(socket: &mut tokio::net::TcpStream) -> (String, Option<Value>) {
    let mut bytes = Vec::new();
    let boundary = loop {
        let mut buffer = [0; 1024];
        let n = socket.read(&mut buffer).await.unwrap();
        assert!(n > 0 && bytes.len() + n <= 65536);
        bytes.extend_from_slice(&buffer[..n]);
        if let Some(i) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let header = std::str::from_utf8(&bytes[..boundary]).unwrap();
    let line = header.lines().next().unwrap().to_owned();
    let length = header
        .lines()
        .find_map(|line| {
            let (k, v) = line.split_once(':')?;
            k.eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    assert!(length <= 32768);
    while bytes.len() - boundary < length {
        let mut buffer = [0; 1024];
        let n = socket.read(&mut buffer).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&buffer[..n]);
    }
    (
        line,
        (length > 0).then(|| serde_json::from_slice(&bytes[boundary..boundary + length]).unwrap()),
    )
}
static TERMINATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
extern "C" fn term(_signal: libc::c_int) {
    TERMINATED.store(true, std::sync::atomic::Ordering::SeqCst);
}
struct TermScope(libc::sigaction);
impl TermScope {
    fn install() -> Self {
        TERMINATED.store(false, std::sync::atomic::Ordering::SeqCst);
        // SAFETY: initialized sigaction values; handler only updates a lock-free atomic.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            let mut previous = std::mem::zeroed();
            action.sa_sigaction = term as *const () as usize;
            assert_eq!(libc::sigemptyset(&mut action.sa_mask), 0);
            assert_eq!(libc::sigaction(libc::SIGTERM, &action, &mut previous), 0);
            Self(previous)
        }
    }
}
impl Drop for TermScope {
    fn drop(&mut self) {
        // SAFETY: previous action was captured during installation in this isolated helper process.
        unsafe {
            assert_eq!(
                libc::sigaction(libc::SIGTERM, &self.0, std::ptr::null_mut()),
                0
            );
        }
    }
}

fn append(record: &Path, value: Value) {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(record)
        .unwrap();
    writeln!(file, "{value}").unwrap();
}
async fn respond(mut socket: tokio::net::TcpStream, record: PathBuf, session: u64, mode: String) {
    let (line, body) = request(&mut socket).await;
    let (kind, response) = if line.starts_with("GET /v1/models ") {
        (
            "application/json",
            json!({"data":[{"id":"inert-restart"}]}).to_string(),
        )
    } else {
        assert!(line.starts_with("POST /v1/chat/completions "));
        let body = body.unwrap();
        assert_eq!(body["model"], "inert-restart");
        assert_eq!(body["seed"], 42);
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["max_tokens"], 8);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(
            body["messages"].as_array().unwrap().last().unwrap()["role"],
            "user"
        );
        append(
            &record,
            json!({"event":"request","session":session,"messages":body["messages"]}),
        );
        if session == 2 && mode == "hold" {
            std::fs::write(
                record.parent().unwrap().join("held"),
                b"replay-request-inflight",
            )
            .unwrap();
            std::future::pending::<()>().await;
        }
        let content = if session == 2 && mode == "mismatch" {
            "changed"
        } else {
            "baseline"
        };
        let mut usage = json!({"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":if session==2{5}else{0}}});
        if session == 2 && mode == "missing-prompt" {
            usage.as_object_mut().unwrap().remove("prompt_tokens");
        }
        if session == 2 && mode == "missing-cache" {
            usage
                .as_object_mut()
                .unwrap()
                .remove("prompt_tokens_details");
        }
        let delta = json!({"choices":[{"delta":{"content":content},"finish_reason":"stop"}]});
        let done = if session == 2 && mode == "truncated" {
            ""
        } else {
            "data: [DONE]\n\n"
        };
        (
            "text/event-stream",
            format!(
                "data: {delta}\n\ndata: {}\n\n{done}",
                json!({"usage":usage})
            ),
        )
    };
    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",response.len()).as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}
async fn server() {
    let mode = std::env::var("KV_FIXTURE_MODE").unwrap();
    let record = PathBuf::from(std::env::var("KV_FIXTURE_RECORD").unwrap());
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let counter = home.join("cohort-session");
    let session = std::fs::read_to_string(&counter)
        .ok()
        .map(|v| v.parse::<u64>().unwrap() + 1)
        .unwrap_or(1);
    assert!(session <= 2);
    let previous = home.join("clean-stop").exists();
    if session == 2 {
        assert!(previous);
    }
    std::fs::write(&counter, session.to_string()).unwrap();
    append(
        &record,
        json!({"event":"launch","session":session,"home":home,"previous_clean_stop":previous,"binary_argv":std::env::var("KV_FIXTURE_BINARY").unwrap(),"model_argv":std::env::var("KV_FIXTURE_MODEL").unwrap()}),
    );
    let listener = TcpListener::bind("127.0.0.1:9337").await.unwrap();
    let mut tasks = tokio::task::JoinSet::new();
    let mut poll = tokio::time::interval(Duration::from_millis(5));
    loop {
        tokio::select! {_=poll.tick()=>{if TERMINATED.load(std::sync::atomic::Ordering::SeqCst)&&!(session==2&&mode=="shutdown-failure"){break;}},accepted=listener.accept()=>{let(socket,_)=accepted.unwrap();let record=record.clone();let mode=mode.clone();tasks.spawn(async move{tokio::time::timeout(Duration::from_secs(8),respond(socket,record,session,mode)).await.unwrap();});},complete=tasks.join_next(),if !tasks.is_empty()=>{complete.unwrap().unwrap();}}
    }
    tasks.abort_all();
    while let Some(value) = tasks.join_next().await {
        assert!(value.is_ok() || value.unwrap_err().is_cancelled());
    }
    std::fs::write(home.join("clean-stop"), session.to_string()).unwrap();
    append(&record, json!({"event":"clean-stop","session":session}));
}
#[test]
#[ignore = "isolated owned inert default server; invoked only by wrapper"]
fn owned_default_server() {
    let _signal = TermScope::install();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(server());
}

#[test]
fn kv_restart_full_cli_actual_bind_and_state_flags_refuse_before_output_or_launch() {
    let _serial = SERIAL.lock().unwrap();
    let root = fixture("success");
    let mut input: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("input.json")).unwrap()).unwrap();
    for extra in [
        vec!["--bind-ip", "0.0.0.0"],
        vec!["--bind-port=9999"],
        vec!["--listen-all"],
        vec!["--config=/other.toml"],
        vec!["--kv-cache-disk-dir=/other"],
        vec!["--native-serving-plugin-state=/other"],
        vec!["--gguf=/other.gguf"],
        vec!["--relay-auth=https://relay.invalid=dummy-secret"],
        vec!["--join", "dummy-secret"],
        vec!["--join-file=/invite"],
        vec!["-jdummy-secret"],
    ] {
        input["serve_extra_args"] = json!(extra);
        std::fs::write(
            root.path().join("input.json"),
            serde_json::to_vec(&input).unwrap(),
        )
        .unwrap();
        let raw = invoke(root.path().into(), process::Cancellation::default());
        cleanup(&raw);
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        assert!(
            !raw.stdout
                .as_ref()
                .unwrap()
                .as_bytes()
                .windows(b"dummy-secret".len())
                .any(|w| w == b"dummy-secret")
        );
        assert!(
            !raw.stderr
                .as_ref()
                .unwrap()
                .as_bytes()
                .windows(b"dummy-secret".len())
                .any(|w| w == b"dummy-secret")
        );
        assert!(!root.path().join("output").exists());
        assert!(!root.path().join("observations.jsonl").exists());
    }
    root.close().unwrap();
}
