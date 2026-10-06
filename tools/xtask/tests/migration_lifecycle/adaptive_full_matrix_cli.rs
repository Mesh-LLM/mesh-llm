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
        "#!/bin/bash\nset -eu\npid=''\ntrap 'if [[ -n \"$pid\" ]]; then wait \"$pid\" || true; fi; exit 0' TERM INT\nconfig=''\nbind=''\ntarget=''\n[[ \"$1\" == serve-binary ]] || exit 64\nshift\nwhile (($#)); do\ncase \"$1\" in\n--config) config=\"$2\"; shift 2;;\n--openai-bind-addr) bind=\"$2\"; shift 2;;\n--openai-prefill-adaptive-target-ms) target=\"$2\"; shift 2;;\n--max-inflight) [[ \"$2\" == 2 ]] || exit 64; shift 2;;\n--reply-credit-limit) [[ \"$2\" == 1 ]] || exit 64; shift 2;;\n--async-prefill-forward) shift;;\n--telemetry-level) [[ \"$2\" == debug ]] || exit 64; shift 2;;\n--openai-generation-concurrency) [[ \"$2\" == 1 ]] || exit 64; shift 2;;\n--openai-prefill-chunk-policy) [[ \"$2\" == adaptive-ramp ]] || exit 64; shift 2;;\n--openai-prefill-chunk-size) [[ \"$2\" == 256 ]] || exit 64; shift 2;;\n--openai-prefill-adaptive-start) [[ \"$2\" == 128 ]] || exit 64; shift 2;;\n--openai-prefill-adaptive-step) [[ \"$2\" == 128 ]] || exit 64; shift 2;;\n--openai-prefill-adaptive-max) [[ \"$2\" == 384 ]] || exit 64; shift 2;;\n*) exit 64;;\nesac\ndone\nexport ADAPTIVE_FIXTURE_CONFIG=\"$config\"\nexport ADAPTIVE_FIXTURE_BIND=\"$bind\"\nexport ADAPTIVE_FIXTURE_TARGET=\"$target\"\nexport ADAPTIVE_FIXTURE_MODE='{mode}'\nexport ADAPTIVE_FIXTURE_MARKER={}\n{} --exact adaptive_full_matrix_cli::owned_static_arm --ignored --nocapture &\npid=$!\nwait \"$pid\"\n",
        quote(marker),
        quote(&test_binary)
    );
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
    let manifest = json!({"metadata":{"revision":"pinned-inert"},"prompts":[{"family":"trace-1","prompt":"first fixture","source_id":"one"},{"family":"trace-2","prompt":"second fixture"}]});
    let worker = json!({"schema_version":1,"round":1,"version":"old","base_url":"http://127.0.0.1:12003/v1","model":"inert-fixture","output_tokens":2,"request_timeout_secs":2.0,"timeout_secs":4,"readiness_timeout_secs":0,"prompt_manifest_sha256":hash(&serde_json::to_vec(&manifest).unwrap()),"manifest":manifest,"provenance":{"fixture":"inert-static-arm"}});
    json!({"schema_version":1,"old":arms[0],"new":arms[1],"worker":worker,"rounds":2,"timeout_secs":60,"cell_timeout_secs":30,"startup_timeout_secs":2})
}
fn invoke(root: PathBuf, cancel: process::Cancellation) -> process::RawProcessReport {
    process::supervise_raw(
        &process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: ["automation", "waiting-prefix", "adaptive-run", "--input"]
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
fn read(root: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(root.join("matrix/comparison.json")).unwrap()).unwrap()
}
#[test]
fn adaptive_full_matrix_cli_inert_success_and_later_arm_refusal_preserve_identity_and_reports() {
    for mode in ["success", "measured-error"] {
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
            Some(if mode == "success" { 0 } else { 1 })
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
            assert_eq!(output["comparison"]["output_parity"]["exact_matches"], 4);
            assert_eq!(output["comparison"]["bootstrap_resamples"], 10000);
            assert!(
                std::fs::read_to_string(root.join("matrix/report.md"))
                    .unwrap()
                    .contains("Paired 95% CI")
            );
            for cell in cells {
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
                assert_eq!(cell["measured_prefill"].as_array().unwrap().len(), 2);
                assert_eq!(
                    cell["latest_calibration"]["llama_stage.prefill_calibration_observations"],
                    1.0
                );
                assert_eq!(
                    cell["requests"]["requests"][0]["prompt_provenance"]["source_id"],
                    "one"
                );
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
            assert_eq!(failed["successful_requests"], 1);
            assert_eq!(failed["requests"].as_array().unwrap().len(), 2);
            assert!(failed["error"].is_string());
        }
        directory.close().unwrap();
    }
}
#[test]
fn adaptive_full_matrix_cli_inert_marker_driven_cancellation_retains_prior_cell_and_cleans_children()
 {
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
            assert!(marker.is_ok(), "owned third measured request marker absent");
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
#[test]
#[ignore = "owned inert static-arm subprocess entry; selected only by wrapper"]
fn owned_static_arm() {
    let config: Value = serde_json::from_slice(
        &std::fs::read(std::env::var("ADAPTIVE_FIXTURE_CONFIG").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(config["ctx_size"], 512);
    assert_eq!(config["kv_cache"]["mode"], "disabled");
    assert_eq!(config["n_gpu_layers"], 999);
    let upstream = config["stage_index"] == 0;
    let bind = std::env::var("ADAPTIVE_FIXTURE_BIND").unwrap();
    let target = std::env::var("ADAPTIVE_FIXTURE_TARGET").unwrap();
    let version = if config["run_id"].as_str().unwrap().ends_with("new") {
        "new"
    } else {
        "old"
    };
    if upstream {
        assert_eq!(target, if version == "new" { "10" } else { "" });
    } else {
        assert!(bind.is_empty());
    }
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async{
        let _stage=TcpListener::bind(config["bind_addr"].as_str().unwrap()).await.unwrap();
        if !upstream {std::future::pending::<()>().await;}
        let address: std::net::SocketAddr = bind.parse().unwrap();assert_eq!(address.ip(),std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));assert_ne!(address.port(),0);let listener=TcpListener::bind(address).await.unwrap();let mut index=0;
        loop{
            let(mut socket,_)=listener.accept().await.unwrap();let(line,body)=request(&mut socket).await;
            let(content_type,response)=if line.starts_with("GET /v1/models "){("application/json",json!({"data":[{"id":config["model_id"]}]}).to_string())}else{
                assert!(line.starts_with("POST /v1/chat/completions "));let body=body.unwrap();assert_eq!(body["model"],config["model_id"]);assert_eq!(body["max_tokens"],2);
                let mode=std::env::var("ADAPTIVE_FIXTURE_MODE").unwrap();if index==2&&mode=="hold"{std::fs::write(std::env::var("ADAPTIVE_FIXTURE_MARKER").unwrap(),b"measured-inflight").unwrap();std::future::pending::<()>().await;}
                let error=index==2&&mode=="measured-error";
                if !error{eprintln!("{}",json!({"event":"stage.openai_prefill","attributes":{"llama_stage.prefill_chunk_count":3,"llama_stage.prefill_min_chunk_size":128,"llama_stage.prefill_max_chunk_size":384,"llama_stage.elapsed_ms":5.0,"skippy.kv.chain_cache_errors":0,"skippy.kv.stage0_cache_errors":0}}));if index==0{eprintln!("{}",json!({"event":"stage.openai_prefill_calibration","attributes":{"llama_stage.prefill_calibration_observations":1}}));}}
                index+=1;("text/event-stream",if error{"data: {\"error\":{\"message\":\"inert measured refusal\"}}\n\n".into()}else{"data: {\"choices\":[{\"delta\":{\"content\":\"inert measured output\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n".into()})
            };
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",response.len()).as_bytes()).await.unwrap();socket.shutdown().await.unwrap();
        }
    });
}

#[test]
fn adaptive_prepare_cli_binds_manifest_without_launch_and_refuses_provenance_drift() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let mut input = document(root, "success");
    input["worker"]
        .as_object_mut()
        .unwrap()
        .remove("prompt_manifest_sha256");
    input["worker"]["manifest"]["prompts"][0]["prompt"] = json!("long context row\n".repeat(3840));
    assert!(
        input["worker"]["manifest"]["prompts"][0]["prompt"]
            .as_str()
            .unwrap()
            .len()
            > 16 * 1024
    );
    for (name, expected) in [("prepared", 0), ("stale", 1)] {
        if name == "stale" {
            input["worker"]["prompt_manifest_sha256"] = json!("0".repeat(64));
        }
        std::fs::write(
            root.join("prepare-input.json"),
            serde_json::to_vec(&input).unwrap(),
        )
        .unwrap();
        let output = root.join(format!("{name}.json"));
        let raw = process::supervise_raw(
            &process::ProcessSpec {
                executable: PathBuf::from(env!("CARGO_BIN_EXE_xtask")),
                cwd: root.into(),
                arguments: [
                    "automation".into(),
                    "waiting-prefix".into(),
                    "adaptive-prepare".into(),
                    "--input".into(),
                    root.join("prepare-input.json").into_os_string(),
                    "--output".into(),
                    output.as_os_str().to_owned(),
                ]
                .into_iter()
                .map(process::Value::Public)
                .collect(),
                environment: BTreeMap::new(),
            },
            &process::Limits {
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
            &process::Cancellation::default(),
            process::RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        cleanup(&raw);
        assert_eq!(raw.process.outcome, process::Outcome::Exited);
        assert_eq!(raw.process.status.as_ref().unwrap().code(), Some(expected));
        if expected == 0 {
            let prepared: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
            assert_eq!(prepared["worker"]["manifest"], input["worker"]["manifest"]);
            assert_eq!(
                prepared["worker"]["provenance"],
                input["worker"]["provenance"]
            );
            assert_eq!(
                prepared["worker"]["prompt_manifest_sha256"],
                hash(&serde_json::to_vec(&prepared["worker"]["manifest"]).unwrap())
            );
            assert_eq!(prepared["old"], input["old"]);
            assert_eq!(prepared["new"], input["new"]);
            assert_eq!(prepared["rounds"], input["rounds"]);
        } else {
            assert!(!output.exists());
            assert!(
                std::str::from_utf8(raw.stderr.as_ref().unwrap().as_bytes())
                    .unwrap()
                    .contains("mismatched typed manifest pin")
            );
        }
        assert!(
            !root.join("matrix").exists(),
            "preparation must not launch an arm"
        );
    }
    directory.close().unwrap();
}
