//! Actual native matrix dispatch with owned inert static-arm wrapper and bounded HTTP fixture.
//! This qualifies orchestration only; there is no real Skippy/runtime/model inference.
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
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
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
fn wrapper(path: &Path, mode: &str, marker: &Path) {
    use std::os::unix::fs::PermissionsExt as _;
    let test_binary = std::env::current_exe().unwrap();
    // Every interpolated path is an owned fixture path, shell quoted; mode is a closed test constant.
    let script = format!(
        "#!/bin/bash\nset -eu\npid=''\ntrap 'if [[ -n \"$pid\" ]]; then wait \"$pid\" || true; fi; exit 0' TERM INT\nconfig=''\nbind=''\ntarget=''\n[[ \"$1\" == serve-binary ]] || exit 64\nshift\nwhile (($#)); do\ncase \"$1\" in\n--config) config=\"$2\"; shift 2;;\n--openai-bind-addr) bind=\"$2\"; shift 2;;\n--openai-prefill-adaptive-target-ms) target=\"$2\"; shift 2;;\n--max-inflight) [[ \"$2\" == 2 ]] || exit 64; shift 2;;\n--reply-credit-limit) [[ \"$2\" == 1 ]] || exit 64; shift 2;;\n--async-prefill-forward) shift;;\n--telemetry-level) [[ \"$2\" == debug ]] || exit 64; shift 2;;\n--openai-generation-concurrency) [[ \"$2\" == 2 ]] || exit 64; shift 2;;\n--openai-default-max-tokens) [[ \"$2\" == 128 ]] || exit 64; shift 2;;\n--openai-prefill-chunk-policy) [[ \"$2\" == adaptive-ramp ]] || exit 64; shift 2;;\n--openai-prefill-chunk-size) [[ \"$2\" == 256 ]] || exit 64; shift 2;;\n--openai-prefill-adaptive-start) [[ \"$2\" == 256 ]] || exit 64; shift 2;;\n--openai-prefill-adaptive-step) [[ \"$2\" == 256 ]] || exit 64; shift 2;;\n--openai-prefill-adaptive-max) [[ \"$2\" == 256 ]] || exit 64; shift 2;;\n*) exit 64;;\nesac\ndone\nexport MIXED_FIXTURE_CONFIG=\"$config\"\nexport MIXED_FIXTURE_BIND=\"$bind\"\nexport MIXED_FIXTURE_TARGET=\"$target\"\nexport MIXED_FIXTURE_MODE='{mode}'\nexport MIXED_FIXTURE_MARKER={}\n{} --exact mixed_full_matrix_cli::owned_static_arm --ignored --nocapture &\npid=$!\nwait \"$pid\"\n",
        quote(marker),
        quote(&test_binary)
    );
    let current = path.file_name().is_some_and(|name| name == "arm-new");
    let mut script = script;
    if current {
        script = script.replace("serve-binary", "serve").replace("--openai-", "--").replace("--config)", "--stage-transport) [[ \"$2\" == binary ]] || exit 64; transport=1; shift 2;;\n--worker-only) worker=1; shift;;\n--config)");
        script = script.replace("target=''", "target=''\nworker=0\ntransport=0").replace("export MIXED_FIXTURE_CONFIG", "[[ \"$transport\" == 1 ]] || exit 64\n[[ ( -n \"$bind\" && \"$worker\" == 0 ) || ( -z \"$bind\" && \"$worker\" == 1 ) ]] || exit 64\nexport MIXED_FIXTURE_CONFIG");
    }
    let verb = if current { "serve" } else { "serve-binary" };
    script = script.replace("set -eu\n", &format!("set -eu\nif [[ \"$#\" == 2 && \"$2\" == --help ]]; then [[ \"$1\" == {verb} ]] || exit 64; printf 'inert help\\n'; exit 0; fi\n"));
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}
fn document(root: &Path, mode: &str) -> Value {
    let model_path = root.join("model.gguf");
    model(&model_path);
    let native = root.join("native-build");
    std::fs::create_dir(&native).unwrap();
    std::fs::write(
        native.join("inert-library"),
        b"inert static build provenance",
    )
    .unwrap();
    // Product tree contract golden for the single inert-library entry and exact bytes above.
    let tree_hash = "76c45a874d4dd1fc657e9e8a9eb736503ac2b725c2095a0314ae5768edcf3bb5";
    let mut arms = Vec::new();
    for version in ["old", "new"] {
        let binary = root.join(format!("arm-{version}"));
        wrapper(
            &binary,
            if version == "new" { mode } else { "success" },
            &root.join("measured-held"),
        );
        arms.push(json!({"schema_version":1,"round":1,"version":version,"binary":binary,"binary_sha256":hash(&std::fs::read(&binary).unwrap()),"commit":"b".repeat(40),"native_build":native,"native_build_sha256":tree_hash,"native_profile":"standalone-static-skippy-server","model":model_path,"model_sha256":hash(&std::fs::read(&model_path).unwrap()),"model_id":"inert-fixture","ctx_size":512,"split_layer":2,"layer_end":4,"n_gpu_layers":999,"adaptive_target_ms":10.0,"stage_ports":[12001,12002],"openai_port":12003}));
    }
    let manifest = json!({"metadata":{"revision":"pinned-inert"},"prompts":[{"family":"trace-1","prompt":"first fixture","source_id":"one"},{"family":"trace-2","prompt":"second fixture","source_id":"two"}]});
    let shape = json!({"rounds":2,"anchors":1,"prefills":1,"anchor_prompt_blocks":2,"prefill_prompt_blocks":4,"anchor_output_tokens":128,"prefill_output_tokens":8,"prefill_delay_ms":50.0,"prefill_stagger_ms":5.0,"lanes":2,"n_batch":1024,"n_ubatch":256,"prefill_adaptive_start":256,"prefill_adaptive_step":256,"prefill_adaptive_max":256,"adaptive_target_new_only":true});
    json!({"schema_version":1,"old":arms[0],"new":arms[1],"shape":shape,"split":true,"manifest":manifest,"request_timeout_secs":2.0,"worker_timeout_secs":4,"suppressed_token_ids":[7],"timeout_secs":60,"cell_timeout_secs":30,"startup_timeout_secs":2})
}
fn invoke(root: PathBuf, cancel: process::Cancellation) -> process::RawProcessReport {
    process::supervise_raw(
        &process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: ["automation", "waiting-prefix", "mixed-run", "--input"]
                .into_iter()
                .map(|s| process::Value::Public(s.into()))
                .chain([
                    process::Value::Public(root.join("input.json").into_os_string()),
                    process::Value::Public("--output-directory".into()),
                    process::Value::Public(root.join("matrix").into_os_string()),
                ])
                .collect(),
            cwd: root,
            environment: BTreeMap::new(),
        },
        &process::Limits {
            execution: Duration::from_secs(20),
            graceful_shutdown: Duration::from_secs(15),
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
fn unsupported_refusal(root: &Path) {
    let raw = process::supervise_raw(
        &process::ProcessSpec {
            executable: root.join("arm-old"),
            cwd: root.into(),
            arguments: vec![
                process::Value::Public("serve-binary".into()),
                process::Value::Public("--openai-port".into()),
                process::Value::Public("9337".into()),
            ],
            environment: BTreeMap::new(),
        },
        &process::Limits {
            execution: Duration::from_secs(1),
            graceful_shutdown: Duration::from_millis(250),
            forced_shutdown: Duration::from_millis(250),
            retained_bytes_per_stream: 4096,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(4096),
            stderr: NonZeroUsize::new(4096),
        },
    )
    .unwrap();
    cleanup(&raw);
    assert_eq!(raw.process.outcome, process::Outcome::Exited);
    assert_eq!(raw.process.status.as_ref().unwrap().code(), Some(64));
    assert!(raw.stdout.as_ref().unwrap().as_bytes().is_empty());
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
    serde_json::from_slice(&std::fs::read(root.join("matrix/comparison.json")).unwrap()).unwrap()
}
#[test]
fn mixed_full_matrix_cli_inert_success_and_later_arm_refusal_preserve_identity_and_reports() {
    for mode in ["success", "measured-error", "tail-drop"] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let input = document(root, mode);
        unsupported_refusal(root);
        std::fs::write(root.join("input.json"), serde_json::to_vec(&input).unwrap()).unwrap();
        let raw = invoke(root.into(), process::Cancellation::default());
        cleanup(&raw);
        assert_eq!(raw.process.outcome, process::Outcome::Exited);
        assert_eq!(
            raw.process.status.unwrap().code(),
            Some(if mode == "success" { 0 } else { 1 }),
            "{}",
            failure_evidence(root, &raw)
        );
        let output = read(root);
        let cells = output["cells"].as_array().unwrap();
        if mode == "success" {
            assert!(output["error"].is_null(), "{output}");
            assert_eq!(cells.len(), 4);
            assert_eq!(
                cells
                    .iter()
                    .map(|c| (c["round"].as_u64().unwrap(), c["version"].as_str().unwrap()))
                    .collect::<Vec<_>>(),
                vec![(1, "old"), (1, "new"), (2, "new"), (2, "old")]
            );
            // Two measured requests in each arm of each of two rounds form four
            // old/new pairs. Count all eight physical rows separately.
            assert_eq!(
                cells
                    .iter()
                    .map(|cell| cell["requests"].as_array().unwrap().len())
                    .sum::<usize>(),
                8
            );
            assert_eq!(
                output["comparison"]["output_parity"]["comparable_requests"],
                4
            );
            assert_eq!(output["comparison"]["output_parity"]["exact_matches"], 4);
            assert!(
                output["comparison"]["output_parity"]["mismatches"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(output["comparison"]["bootstrap_resamples"], 10000);
            assert!(
                std::fs::read_to_string(root.join("matrix/report.md"))
                    .unwrap()
                    .contains("Paired 95% CI")
            );
            for cell in cells {
                let session_failure = cell["lifecycle"]
                    .get("session_failure")
                    .expect("successful cell must retain supervisor failure metadata");
                assert!(session_failure.is_null(), "{cell}");
                let version = cell["version"].as_str().unwrap();
                assert_eq!(
                    cell["identity"]["admitted"]["binary_sha256"],
                    input[version]["binary_sha256"]
                );
                assert_eq!(
                    cell["identity"]["admitted"]["model_sha256"],
                    input[version]["model_sha256"]
                );
                let members = cell["lifecycle"]["members"].as_array().unwrap();
                assert_eq!(members.len(), 3);
                for member in members {
                    assert_eq!(member["status"], 0);
                    assert_eq!(member["cleanup_complete"], true);
                    assert_eq!(member["forced"], false);
                    assert_eq!(member["graceful_signal_failed"], false);
                    assert!(
                        member["cleanup_failure"].is_null() && member["process_failure"].is_null()
                    );
                    assert_eq!(member["stdout_complete"], true);
                    assert_eq!(member["stderr_complete"], true);
                    assert!(matches!(
                        member["disposition"].as_str(),
                        Some("intentional-stop" | "expected-exit")
                    ));
                }
                assert_eq!(cell["summary"]["successful_requests"], 2);
                assert_eq!(cell["summary"]["scheduler_phase_qualified"], true);
                assert_eq!(cell["summary"]["scheduler_iterations"], 1);
                assert_eq!(cell["requests"][0]["role"], "anchor");
                assert_eq!(cell["requests"][1]["role"], "prefill");
                assert_eq!(
                    cell["requests"][1]["prompt_provenance"]["source_id"],
                    if cell["round"] == 1 { "one" } else { "two" }
                );
                assert!(
                    cell["requests"][0]["completed_ms"].as_f64().unwrap()
                        >= cell["requests"][1]["submitted_ms"].as_f64().unwrap()
                );
                assert_eq!(
                    cell["requests"][0]["content_gaps_ms"]
                        .as_array()
                        .unwrap()
                        .len(),
                    1
                );
                assert_eq!(cell["worker_receipt"]["warmup"]["completion_tokens"], 2);
                assert_eq!(cell["configs"][0]["lane_count"], 2);
                assert_eq!(cell["configs"][0]["n_batch"], 1024);
                assert!(cell["configs"][0].get("kv_cache").is_none());
            }
        } else {
            assert!(output["error"].is_string());
            assert_eq!(cells.len(), 1);
            assert_eq!(cells[0]["version"], "old");
            assert_eq!(cells[0]["summary"]["successful_requests"], 2);
            assert!(output["comparison"].is_null());
            let failed: Value = serde_json::from_slice(
                &std::fs::read(root.join("matrix/round-1-new/requests.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(
                failed["successful_requests"],
                if mode == "tail-drop" { 2 } else { 1 }
            );
            assert_eq!(failed["requests"].as_array().unwrap().len(), 2);
            if mode == "tail-drop" {
                assert!(failed["error"].is_null());
                let phase: Value = serde_json::from_slice(
                    &std::fs::read(root.join("matrix/round-1-new/phase-observations.json"))
                        .unwrap(),
                )
                .unwrap();
                assert!(phase["error"].is_string());
                let life: Value = serde_json::from_slice(
                    &std::fs::read(root.join("matrix/round-1-new/lifecycle.json")).unwrap(),
                )
                .unwrap();
                for member in life["members"].as_array().unwrap() {
                    assert_eq!(member["cleanup_complete"], true);
                    assert_eq!(member["forced"], false);
                    assert_eq!(member["graceful_signal_failed"], false);
                    assert!(
                        member["process_failure"].is_null() && member["cleanup_failure"].is_null()
                    );
                    assert_eq!(member["stderr_complete"], true);
                }
            } else {
                assert!(failed["error"].is_string());
            }
        }
        directory.close().unwrap();
    }
}
#[test]
fn mixed_full_matrix_cli_inert_marker_driven_cancellation_retains_prior_cell_and_cleans_children() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path();
            let input = document(root, "hold");
            std::fs::write(root.join("input.json"), serde_json::to_vec(&input).unwrap()).unwrap();
            let cancel = process::Cancellation::default();
            let worker_cancel = cancel.clone();
            let worker_root = root.to_owned();
            let task = tokio::task::spawn_blocking(move || invoke(worker_root, worker_cancel));
            let marker = tokio::time::timeout(Duration::from_secs(15), async {
                let mut poll = tokio::time::interval(Duration::from_millis(10));
                loop {
                    poll.tick().await;
                    if root.join("measured-held").is_file() {
                        break;
                    }
                }
            })
            .await;
            cancel.cancel();
            let raw = task.await.unwrap();
            cleanup(&raw);
            assert!(
                marker.is_ok(),
                "owned anchor after excluded warmup marker absent: {}",
                failure_evidence(root, &raw)
            );
            assert_eq!(raw.process.outcome, process::Outcome::Cancelled);
            let output = read(root);
            assert!(output["error"].is_string());
            assert_eq!(output["cells"].as_array().unwrap().len(), 1);
            assert_eq!(output["cells"][0]["version"], "old");
            assert!(output["comparison"].is_null());
            directory.close().unwrap();
        });
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
fn nanos() -> u64 {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    )
    .unwrap()
}
fn telemetry(event: &str, start: u64, dropped: u64) {
    eprintln!(
        "{}",
        json!({"event":event,"start_time_unix_nanos":start,"end_time_unix_nanos":start+1,"attributes":{"skippy.otel_dropped_events":dropped,"skippy.otel_export_errors":0,"skippy.scheduler.token_count":7,"llama_stage.prefill_token_count":40,"llama_stage.prefill_chunk_count":2,"llama_stage.prefill_max_chunk_size":256,"skippy.kv.chain_cache_errors":0,"skippy.kv.stage0_cache_errors":0}})
    );
}
async fn serve(
    mut socket: tokio::net::TcpStream,
    config: Value,
    mode: String,
    release: std::sync::Arc<tokio::sync::Notify>,
    tail: std::sync::Arc<std::sync::Mutex<Option<u64>>>,
) {
    let (line, body) = request(&mut socket).await;
    let (kind, response) = if line.starts_with("GET /v1/models ") {
        (
            "application/json",
            json!({"data":[{"id":config["model_id"]}]}).to_string(),
        )
    } else {
        assert!(line.starts_with("POST /v1/chat/completions "));
        let body = body.unwrap();
        assert_eq!(body["model"], config["model_id"]);
        assert_eq!(body["logit_bias"]["7"], -100);
        let tokens = body["max_tokens"].as_u64().unwrap();
        assert!(matches!(tokens, 4 | 8 | 128));
        if tokens == 128 {
            if mode == "hold" {
                std::fs::write(
                    std::env::var("MIXED_FIXTURE_MARKER").unwrap(),
                    b"anchor-inflight-after-warmup",
                )
                .unwrap();
                std::future::pending::<()>().await;
            }
            release.notified().await;
        }
        if tokens == 8 {
            release.notify_one();
            *tail.lock().unwrap() = Some(nanos());
        }
        let error = mode == "measured-error" && tokens == 8;
        if !error {
            telemetry("stage.openai_prefill", nanos(), 0);
        }
        (
            "text/event-stream",
            if error {
                "data: {\"error\":{\"message\":\"inert mixed prefill refusal\"}}\n\n".into()
            } else {
                "data: {\"choices\":[{\"delta\":{\"content\":\"one\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"two\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n".into()
            },
        )
    };
    socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",response.len()).as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}
async fn stage(config: Value, bind: String, mode: String) {
    let _stage = TcpListener::bind(config["bind_addr"].as_str().unwrap())
        .await
        .unwrap();
    let upstream = config["stage_index"] == 0;
    let listener = if upstream {
        let address: std::net::SocketAddr = bind.parse().unwrap();
        assert_eq!(
            address.ip(),
            std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
        );
        assert_ne!(address.port(), 0);
        Some(TcpListener::bind(address).await.unwrap())
    } else {
        None
    };
    let release = std::sync::Arc::new(tokio::sync::Notify::new());
    let tail = std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut tasks = tokio::task::JoinSet::new();
    let mut poll = tokio::time::interval(Duration::from_millis(5));
    loop {
        tokio::select! {
            _=poll.tick()=>{if TERMINATED.load(std::sync::atomic::Ordering::SeqCst){break;}},
            accepted=async{match &listener{Some(listener)=>listener.accept().await,None=>std::future::pending().await}}=>{
                let(socket,_)=accepted.unwrap();let config=config.clone();let mode=mode.clone();let release=release.clone();let tail=tail.clone();tasks.spawn(async move{serve(socket,config,mode,release,tail).await;});
            },
            completed=tasks.join_next(),if !tasks.is_empty()=>{completed.unwrap().unwrap();}
        }
    }
    tasks.abort_all();
    while let Some(result) = tasks.join_next().await {
        assert!(result.is_ok() || result.unwrap_err().is_cancelled());
    }
    // Emitted only after TERM; producer timestamp still belongs to measured prefill admission.
    if let Some(start) = *tail.lock().unwrap() {
        telemetry(
            "stage.scheduler_feature_iteration",
            start,
            u64::from(mode == "tail-drop"),
        );
    }
}
#[test]
#[ignore = "owned inert mixed static-arm subprocess entry; selected only by wrapper"]
fn owned_static_arm() {
    let _signal = TermScope::install();
    let config: Value = serde_json::from_slice(
        &std::fs::read(std::env::var("MIXED_FIXTURE_CONFIG").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(config["ctx_size"], 512);
    assert_eq!(config["n_gpu_layers"], 999);
    assert_eq!(config["lane_count"], 2);
    assert_eq!(config["n_batch"], 1024);
    assert_eq!(config["n_ubatch"], 256);
    assert!(config.get("kv_cache").is_none());
    let bind = std::env::var("MIXED_FIXTURE_BIND").unwrap();
    let target = std::env::var("MIXED_FIXTURE_TARGET").unwrap();
    let version = if config["run_id"].as_str().unwrap().ends_with("new") {
        "new"
    } else {
        "old"
    };
    if config["stage_index"] == 0 {
        assert_eq!(target, if version == "new" { "10" } else { "" });
    } else {
        assert!(bind.is_empty());
    }
    let mode = std::env::var("MIXED_FIXTURE_MODE").unwrap();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(stage(config, bind, mode));
}
