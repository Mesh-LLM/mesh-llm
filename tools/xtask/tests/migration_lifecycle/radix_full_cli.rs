//! Actual full native radix dispatch with bounded inert standalone serve-openai processes.
use crate::process;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
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

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn wrapper(path: &Path, version: &str, mode: &str, record: &Path) {
    use std::os::unix::fs::PermissionsExt as _;
    let script = format!(
        "#!/bin/bash\nset -eu\npid=''\ntrap 'if [[ -n \"$pid\" ]]; then wait \"$pid\" || true; fi; exit 0' TERM INT\n[[ \"$#\" == 9 && \"$1\" == serve-openai && \"$2\" == --config && \"$4\" == --bind-addr && \"$6\" == --generation-concurrency && \"$7\" == 2 && \"$8\" == --telemetry-level && \"$9\" == debug ]] || exit 64\nexport RADIX_CONFIG=\"$3\"\nexport RADIX_BIND=\"$5\"\nexport RADIX_VERSION='{version}'\nexport RADIX_MODE='{mode}'\nexport RADIX_RECORD={}\n{} --exact radix_full_cli::owned_standalone_server --ignored --nocapture &\npid=$!\nwait \"$pid\"\n",
        quote(record),
        quote(&std::env::current_exe().unwrap())
    );
    let current = version == "new";
    let verb = if current { "serve" } else { "serve-openai" };
    let script = script.replace("serve-openai", verb).replace("set -eu\n", &format!("set -eu\nif [[ \"$#\" == 2 && \"$2\" == --help ]]; then [[ \"$1\" == {verb} ]] || exit 64; printf 'inert help\\n'; exit 0; fi\n"));
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}
fn fixture(mode: &str) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let model_path = root.path().join("model.gguf");
    model(&model_path);
    let native = root.path().join("native");
    std::fs::create_dir(&native).unwrap();
    std::fs::write(
        native.join("inert-library"),
        b"inert static build provenance",
    )
    .unwrap();
    let mut arms = vec![];
    for version in ["old", "new"] {
        let binary = root.path().join(version);
        wrapper(
            &binary,
            version,
            if version == "old" { "success" } else { mode },
            &root.path().join("observations.jsonl"),
        );
        arms.push(json!({"binary":binary,"binary_sha256":hash(&std::fs::read(&binary).unwrap()),"commit":"b".repeat(40),"model":model_path,"model_sha256":hash(&std::fs::read(&model_path).unwrap()),"model_id":"inert-radix","native_build":native,"native_build_sha256":"76c45a874d4dd1fc657e9e8a9eb736503ac2b725c2095a0314ae5768edcf3bb5","ctx_size":512,"layer_end":4,"payload":"resident-kv"}));
    }
    let input = json!({"schema_version":1,"cases":[{"key":"inert","family":"test","old":arms[0],"new":arms[1]}],"shape":{"rounds":2,"requests":2,"levels":[1,2],"prefix_blocks":2,"output_tokens":8,"lanes":2,"n_gpu_layers":999},"request_timeout_secs":2.0,"batch_timeout_secs":4,"cell_timeout_secs":24,"timeout_secs":180,"require_gates":true});
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
            arguments: ["automation", "waiting-prefix", "radix-run", "--input"]
                .into_iter()
                .map(|a| process::Value::Public(a.into()))
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
            execution: Duration::from_secs(60),
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

fn failure_evidence(root: &Path, raw: &process::RawProcessReport) -> String {
    let mut evidence = format!(
        "process={:?}\nstderr={}\n",
        raw.process,
        String::from_utf8_lossy(raw.stderr.as_ref().unwrap().as_bytes())
    );
    let mut pending = vec![root.to_path_buf()];
    let mut inspected = 0;
    while let Some(path) = pending.pop() {
        if inspected >= 100 || evidence.len() >= 128 * 1024 {
            break;
        }
        inspected += 1;
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&path) {
                let mut paths = entries
                    .flatten()
                    .map(|entry| entry.path())
                    .collect::<Vec<_>>();
                paths.sort();
                pending.extend(paths.into_iter().rev());
            }
        } else if metadata.is_file()
            && matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("json" | "log")
            )
        {
            use std::io::Read as _;
            if let Ok(file) = std::fs::File::open(&path) {
                let mut bytes = Vec::new();
                if file.take(8192).read_to_end(&mut bytes).is_ok() {
                    evidence.push_str(&format!(
                        "\n{}:\n{}\n",
                        path.strip_prefix(root).unwrap().display(),
                        String::from_utf8_lossy(&bytes)
                    ));
                }
            }
        }
    }
    evidence
}

fn read(root: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(root.join("output/comparison.json")).unwrap()).unwrap()
}
#[test]
fn radix_full_cli_alternating_cold_warm_matrix_waits_post_receipt_summaries_and_runs_concurrently()
{
    let root = fixture("slow-readiness");
    let raw = invoke(root.path().into(), process::Cancellation::default());
    cleanup(&raw);
    assert!(
        raw.process.success(),
        "{}",
        failure_evidence(root.path(), &raw)
    );
    let result = read(root.path());
    assert!(result["error"].is_null(), "{result}");
    let case = &result["cases"][0];
    assert_eq!(case["gate"]["passed"], true);
    let cells = case["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 8);
    assert_eq!(
        cells
            .iter()
            .map(|c| format!(
                "{}-{}-{}",
                c["round"],
                c["version"].as_str().unwrap(),
                c["cache"].as_str().unwrap()
            ))
            .collect::<Vec<_>>(),
        [
            "1-old-cold",
            "1-old-warm",
            "1-new-cold",
            "1-new-warm",
            "2-new-cold",
            "2-new-warm",
            "2-old-cold",
            "2-old-warm"
        ]
    );
    for cell in cells {
        let session_failure = cell["lifecycle"]
            .get("session_failure")
            .expect("successful cell must retain supervisor failure metadata");
        assert!(session_failure.is_null(), "{cell}");
        assert!(cell["error"].is_null());
        assert_eq!(cell["config"]["n_gpu_layers"], 999);
        assert_eq!(cell["observations"].as_array().unwrap().len(), 6);
        let warm = cell["cache"] == "warm";
        assert_eq!(cell["config"].get("kv_cache").is_some(), warm);
        for obs in cell["observations"].as_array().unwrap() {
            assert_eq!(obs["requests"].as_array().unwrap().len(), 2);
            assert_eq!(obs["summary"]["requests"], 2);
            assert_eq!(obs["summary"]["successful"], 2);
            assert_eq!(obs["summary"]["cache_hits"], if warm { 2 } else { 0 });
            for row in obs["requests"].as_array().unwrap() {
                assert!(row["error"].is_null());
                assert_eq!(row["prompt_tokens"], 10);
                assert_eq!(row["completion_tokens"], 2);
                assert_eq!(row["cached_tokens"], if warm { 5 } else { 0 });
                assert_eq!(row["content_sha256"].as_str().unwrap().len(), 64);
            }
        }
        assert!(
            cell["lifecycle"]["members"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m["clean"] == true
                    && m["forced"] == false
                    && m["cleanup_failure"].is_null()
                    && m["process_failure"].is_null())
        );
    }
    assert!(
        case["output_parity"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["identical_outputs"] == true)
    );
    assert!(
        case["cache_output_preservation"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["cache_preserves_output"] == true)
    );
    assert!(
        root.path().join("output/inert/table.md").is_file()
            && root
                .path()
                .join("output/inert/divergent-prefix.svg")
                .is_file()
    );
    let records = std::fs::read_to_string(root.path().join("observations.jsonl")).unwrap();
    assert!(records.contains("concurrent-second-arrived"));
    assert!(records.contains("summary-after-owned-receipt"));
    let delayed = records
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|v| v["event"] == "advertised-model-after-startup-delay")
        .collect::<Vec<_>>();
    assert!(!delayed.is_empty());
    assert!(records.contains("first-model-probe-denied"));
    assert!(
        delayed
            .iter()
            .all(|v| v["startup_ms"].as_u64().unwrap() >= 1500)
    );
    root.close().unwrap();
}
#[test]
fn radix_full_cli_missing_summary_stream_failure_and_suffix_regression_refuse_with_consumed_evidence()
 {
    for mode in ["missing-summary", "truncated", "suffix-regression"] {
        let root = fixture(mode);
        let raw = invoke(root.path().into(), process::Cancellation::default());
        cleanup(&raw);
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        let result = read(root.path());
        assert!(!result["error"].is_null());
        let case = &result["cases"][0];
        assert_eq!(case["gate"]["passed"], false);
        let cells = case["cells"].as_array().unwrap();
        assert!(cells.len() >= 4, "{}", failure_evidence(root.path(), &raw));
        assert!(cells[..3].iter().all(|c| c["error"].is_null()));
        if mode == "missing-summary" {
            assert!(
                cells[3]["lifecycle"]["rejection"]
                    .as_str()
                    .unwrap()
                    .contains("observed 0/1")
            );
        } else if mode == "truncated" {
            let row = &cells[3]["observations"][0]["requests"][0];
            assert!(!row["error"].is_null());
            assert!(row.get("completion_tokens").is_none());
        } else {
            assert_eq!(cells.len(), 8);
            assert!(
                case["gate"]["failures"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|v| v.as_str().unwrap().contains("suffix prefill exceeded"))
            );
        }
        root.close().unwrap();
    }
}
#[test]
fn radix_full_cli_marker_cancellation_after_warmup_retains_prior_cells_and_cleans_owners() {
    let root = fixture("hold");
    let cancel = process::Cancellation::default();
    let scope = cancel.clone();
    let path = root.path().to_path_buf();
    let worker = std::thread::spawn(move || invoke(path, scope));
    let until = std::time::Instant::now() + Duration::from_secs(30);
    while !root.path().join("held").exists()
        && std::time::Instant::now() < until
        && !worker.is_finished()
    {
        std::thread::park_timeout(Duration::from_millis(5));
    }
    let marker = root.path().join("held").exists();
    cancel.cancel();
    let raw = worker.join().unwrap();
    cleanup(&raw);
    assert!(
        marker,
        "owned supervisor cancelled and joined before missing-marker assertion: {}",
        failure_evidence(root.path(), &raw)
    );
    assert!(cancel.is_cancelled());
    let result = read(root.path());
    assert!(!result["error"].is_null());
    let cells = result["cases"][0]["cells"].as_array().unwrap();
    assert!(cells.len() >= 3);
    assert!(cells[..3].iter().all(|c| c["error"].is_null()));
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

fn append(path: &Path, value: Value) {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    writeln!(file, "{value}").unwrap();
}
fn position(warm: bool, count: usize) -> (usize, bool, bool) {
    if warm {
        let scenario = (count - 1) / 6;
        let offset = (count - 1) % 6;
        (
            scenario * 4
                + match offset {
                    0 => 0,
                    1 | 2 => 1,
                    3 => 2,
                    _ => 3,
                },
            offset == 4,
            offset == 5,
        )
    } else {
        let scenario = (count - 1) / 4;
        let offset = (count - 1) % 4;
        (
            scenario * 2 + usize::from(offset >= 2),
            offset == 2,
            offset == 3,
        )
    }
}
#[derive(Clone)]
struct Peer {
    config: Value,
    mode: String,
    record: PathBuf,
    directory: PathBuf,
    startup: std::sync::Arc<std::sync::Mutex<Option<std::time::Instant>>>,
    count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    release: std::sync::Arc<tokio::sync::Notify>,
}
async fn respond(mut socket: tokio::net::TcpStream, peer: Peer) {
    let Peer {
        config,
        mode,
        record,
        directory,
        startup,
        count,
        release,
    } = peer;
    let (line, body) = request(&mut socket).await;
    if line.starts_with("GET /v1/models ") {
        let startup_ms = {
            let mut origin = startup.lock().unwrap();
            let first = origin.is_none();
            let start = origin.get_or_insert_with(std::time::Instant::now);
            let elapsed = start.elapsed().as_millis() as u64;
            if first && mode == "slow-readiness" {
                append(&record, json!({"event":"first-model-probe-denied"}));
            }
            elapsed
        };
        let available = mode != "slow-readiness" || startup_ms >= 1500;
        if mode == "slow-readiness" && available {
            append(
                &record,
                json!({"event":"advertised-model-after-startup-delay", "startup_ms":startup_ms}),
            );
        }
        let response = if available {
            json!({"data":[{"id":"inert-radix"}]})
        } else {
            json!({"data":[]})
        }
        .to_string();
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",
                    response.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        return;
    }
    assert!(line.starts_with("POST /v1/chat/completions "));
    let body = body.unwrap();
    assert_eq!(body["model"], "inert-radix");
    assert_eq!(body["max_tokens"], 8);
    assert_eq!(body["seed"], 0);
    assert_eq!(body["stream_options"]["include_usage"], true);
    let index = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    let warm = config.get("kv_cache").is_some();
    let (batch, first, second) = position(warm, index);
    if first {
        release.notified().await;
    }
    if second {
        append(
            &record,
            json!({"event":"concurrent-second-arrived","cell":directory.file_name().unwrap().to_str().unwrap()}),
        );
        release.notify_one();
    }
    if mode == "hold" && warm && index == 2 {
        std::fs::write(
            record.parent().unwrap().join("held"),
            b"measured-request-after-completed-warmup",
        )
        .unwrap();
        std::future::pending::<()>().await;
    }
    let delta = json!({"choices":[{"delta":{"content":"baseline"},"finish_reason":"stop"}]});
    let usage = json!({"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":if warm{5}else{0}}}});
    let truncated = mode == "truncated" && warm && index == 2;
    let done = if truncated { "" } else { "data: [DONE]\n\n" };
    let response = format!("data: {delta}\n\ndata: {usage}\n\n{done}");
    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",response.len()).as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
    if mode == "missing-summary" && warm || truncated {
        return;
    }
    let receipt = directory.join(format!("batch-{batch}/receipt.json"));
    let until = std::time::Instant::now() + Duration::from_secs(6);
    while !receipt.exists() {
        if TERMINATED.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        assert!(
            std::time::Instant::now() < until,
            "owned batch receipt never published"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    append(
        &record,
        json!({"event":"summary-after-owned-receipt","batch":batch,"cell":directory.file_name().unwrap().to_str().unwrap()}),
    );
    eprintln!(
        "{}",
        json!({"event":"stage.openai_generation_summary","attributes":{"skippy.kv.status":if warm{"hit"}else{"disabled"},"skippy.kv.matched_prefix_tokens":if warm{100}else{0},"skippy.kv.suffix_prefill_tokens":if warm{if mode=="suffix-regression"{10}else{5}}else{20},"llama_stage.prompt_token_count":10,"skippy.kv.radix.nodes":2}})
    );
}
async fn server() {
    let config_path = PathBuf::from(std::env::var("RADIX_CONFIG").unwrap());
    let config: Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    assert_eq!(config["layer_start"], 0);
    assert_eq!(config["layer_end"], 4);
    assert_eq!(config["n_gpu_layers"], 999);
    assert_eq!(config["lane_count"], 2);
    assert_eq!(config["ctx_size"], 512);
    if let Some(cache) = config.get("kv_cache") {
        assert_eq!(cache["shared_prefix_record_limit"], 4);
        assert_eq!(cache["payload"], "resident-kv");
    }
    let addr: std::net::SocketAddr = std::env::var("RADIX_BIND").unwrap().parse().unwrap();
    assert!(addr.ip().is_loopback() && addr.port() != 0);
    let listener = TcpListener::bind(addr).await.unwrap();
    let record = PathBuf::from(std::env::var("RADIX_RECORD").unwrap());
    let directory = config_path.parent().unwrap().to_path_buf();
    append(
        &record,
        json!({"event":"launch","version":std::env::var("RADIX_VERSION").unwrap(),"cell":directory.file_name().unwrap().to_str().unwrap(),"config":config}),
    );
    let mode = std::env::var("RADIX_MODE").unwrap();
    let startup = std::sync::Arc::new(std::sync::Mutex::new(None));
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let release = std::sync::Arc::new(tokio::sync::Notify::new());
    let mut tasks = tokio::task::JoinSet::new();
    let mut poll = tokio::time::interval(Duration::from_millis(5));
    loop {
        tokio::select! {_=poll.tick()=>if TERMINATED.load(std::sync::atomic::Ordering::SeqCst){break;},accepted=listener.accept()=>{let(socket,_)=accepted.unwrap();let config=config.clone();let mode=mode.clone();let record=record.clone();let directory=directory.clone();let count=count.clone();let release=release.clone();let startup=startup.clone();tasks.spawn(async move{tokio::time::timeout(Duration::from_secs(8),respond(socket,Peer{config,mode,record,directory,startup,count,release})).await.unwrap();});},finished=tasks.join_next(),if !tasks.is_empty()=>{finished.unwrap().unwrap();}}
    }
    tasks.abort_all();
    while let Some(value) = tasks.join_next().await {
        assert!(value.is_ok() || value.unwrap_err().is_cancelled());
    }
}
#[test]
#[ignore = "isolated owned inert serve-openai fixture entry; selected only by wrapper"]
fn owned_standalone_server() {
    let _signal = TermScope::install();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(server());
}
