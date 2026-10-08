//! Finite native acquisition/export and private helper-process seam; never compiled in production.
use super::*;
use std::{
    io::{Read as _, Write as _},
    net::TcpListener,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};
pub(crate) struct Peer {
    pub(crate) endpoint: String,
    stop: Arc<AtomicBool>,
    seen: Arc<Mutex<Vec<String>>>,
    task: Option<thread::JoinHandle<()>>,
}
impl Peer {
    pub(crate) fn new(files: BTreeMap<String, Vec<u8>>, revision: String) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let task = thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(60);
            while !stopped.load(Ordering::SeqCst) && Instant::now() < until {
                let mut socket = match listener.accept() {
                    Ok((s, _)) => s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(_) => break,
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_millis(500)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_millis(500)))
                    .unwrap();
                let mut raw = Vec::new();
                let mut block = [0; 1024];
                while raw.len() < 8192 && !raw.windows(4).any(|b| b == b"\r\n\r\n") {
                    match socket.read(&mut block) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => raw.extend_from_slice(&block[..n]),
                    }
                }
                let text = String::from_utf8(raw).unwrap();
                recorded.lock().unwrap().push(text.clone());
                let mut words = text.split_whitespace();
                let method = words.next().unwrap_or("");
                let path = words.next().unwrap_or("").split('?').next().unwrap();
                let (status, body) = match files.get(path) {
                    Some(b) => (200, b.as_slice()),
                    None => (404, b"missing".as_slice()),
                };
                let header = format!(
                    "HTTP/1.1 {status} fixture\r\nContent-Length: {}\r\nETag: \"fixture\"\r\nX-Repo-Commit: {revision}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                if socket.write_all(header.as_bytes()).is_ok() && method != "HEAD" {
                    let _ = socket.write_all(body);
                }
            }
        });
        Self {
            endpoint,
            stop,
            seen,
            task: Some(task),
        }
    }
    pub(crate) fn close(mut self) -> Vec<String> {
        self.stop.store(true, Ordering::SeqCst);
        self.task.take().unwrap().join().unwrap();
        self.seen.lock().unwrap().clone()
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(task) = self.task.take() {
            task.join().unwrap();
        }
    }
}
fn clients(endpoint: &str, explicit: bool) -> NativeClients {
    let mut builder = hf_hub::HFClient::builder()
        .endpoint(endpoint)
        .cache_enabled(false)
        .retry_max_attempts(0)
        .client(
            reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(2))
                .build()
                .unwrap(),
        );
    if explicit {
        builder = builder.token("finite-fixture-authorized");
    } else {
        assert_eq!(
            std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").as_deref(),
            Ok("true")
        );
    }
    NativeClients {
        client: builder.build().unwrap(),
        listing: listing::Listing::fixture(endpoint),
    }
}
pub(super) struct Fixture {
    pub(super) request: PathBuf,
    pub(super) config: Value,
    pub(super) files: BTreeMap<String, Vec<u8>>,
    pub(super) revision: String,
}
use std::path::PathBuf;
pub(super) fn fixture(root: &Path) -> Fixture {
    let revision = "a".repeat(40);
    let mut files = BTreeMap::new();
    let mut models = Vec::new();
    let mut exports = BTreeMap::new();
    let mut semantic = BTreeMap::new();
    let source_config=serde_json::to_vec(&json!({"auto_map":{"AutoTokenizer":"not-executed.py"},"bos_token":"[BOS]","chat_template":"{{ messages }}"})).unwrap();
    for key in [
        "llama32-dense",
        "deepseek-v2-moe",
        "falcon-h1-recurrent",
        "granite-h1-hybrid",
    ] {
        let filename = "fixture.gguf";
        let model = format!("finite-model-{key}").into_bytes();
        let repo = format!("owner/{key}");
        let mut source = BTreeMap::from([
            ("tokenizer.json".to_string(), tests::tokenizer()),
            ("tokenizer_config.json".into(), source_config.clone()),
            ("config.json".into(), b"{}".to_vec()),
            ("README.md".into(), b"omitted".to_vec()),
        ]);
        if key == "granite-h1-hybrid" {
            source.insert(
                "weights/model.safetensors".into(),
                b"full snapshot weight bytes".to_vec(),
            );
            source.insert("docs/README.md".into(), b"also omitted".to_vec());
        }
        let inventory: Vec<_> = source
            .iter()
            .map(|(name, b)| json!({"type":"file","path":name,"oid":"fixture","size":b.len()}))
            .collect();
        files.insert(
            format!("/api/models/{repo}/tree/{revision}"),
            serde_json::to_vec(&inventory).unwrap(),
        );
        for (name, b) in &source {
            files.insert(format!("/{repo}/resolve/{revision}/{name}"), b.clone());
        }
        files.insert(
            format!("/{repo}/resolve/{revision}/{filename}"),
            model.clone(),
        );
        let export = if key == "granite-h1-hybrid" {
            let pins = source
                .iter()
                .filter(|(n, _)| Path::new(n).file_name().unwrap() != "README.md")
                .map(|(n, b)| {
                    (
                        Path::new(n)
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .to_string(),
                        contract::digest(b),
                    )
                })
                .collect();
            contract::tree(&pins).unwrap()
        } else {
            semantic.insert(key.to_string(), tests::cases());
            tests::expected()
        };
        exports.insert(key.to_string(), export.clone());
        models.push(json!({"key":key,"artifact_id":key,"repo":repo,"revision":revision,"filename":filename,"sha256":contract::digest(&model),"tokenizer_sha256":export,"vllm_hf_config":{"repo":repo,"revision":revision,"sha256":contract::digest(b"{}")}}));
    }
    let dataset = b"inert protocol fixture; not actual parquet";
    files.insert(
        format!("/datasets/owner/trajectories/resolve/{revision}/sessions.parquet"),
        dataset.to_vec(),
    );
    let config = json!({"models":models,"thoughtworks":{"dataset":{"repo":"owner/trajectories","revision":revision,"filename":"sessions.parquet","sha256":contract::digest(dataset)},"selection":{"sources":["fixture"],"families":1,"requests_per_family":2,"min_isl":1,"max_isl_exclusive":20,"min_turns":1,"manifest_sha256":"0".repeat(64)}}});
    let config_path = root.join("config.json");
    let config_bytes = serde_json::to_vec(&config).unwrap();
    std::fs::write(&config_path, &config_bytes).unwrap();
    let request = root.join("request.json");
    std::fs::write(&request,serde_json::to_vec(&json!({"schema_version":1,"config":config_path,"config_sha256":contract::digest(&config_bytes),"model_keys":[],"output_directory":root.join("owned"),"timeout_seconds":30,"maximum_bytes":1024*1024,"credential_file":null,"export_sha256":exports,"semantic_cases":semantic})).unwrap()).unwrap();
    Fixture {
        request,
        config,
        files,
        revision,
    }
}
#[test]
fn native_full_four_family_acquisition_export_dataset_and_skip_groups_preserve_exact_bytes() {
    for skip in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let f = fixture(&root);
        let peer = Peer::new(f.files, f.revision);
        let mut request: Value =
            serde_json::from_slice(&std::fs::read(&f.request).unwrap()).unwrap();
        if skip {
            request["model_keys"] = json!(["granite-h1-hybrid"]);
            for key in ["skip_dataset", "skip_tokenizers", "skip_vllm_configs"] {
                request[key] = json!(true);
            }
        }
        std::fs::write(&f.request, serde_json::to_vec(&request).unwrap()).unwrap();
        run_using(
            &["--input".into(), f.request.to_str().unwrap().into()],
            Some(clients(&peer.endpoint, true)),
        )
        .unwrap();
        let acquired: Value =
            serde_json::from_slice(&std::fs::read(root.join("owned/acquisition.json")).unwrap())
                .unwrap();
        assert_eq!(acquired["status"], "ACQUIRED_EXPORTED");
        assert_eq!(
            acquired["families"].as_array().unwrap().len(),
            if skip { 1 } else { 4 }
        );
        verify::run(&f.request, "before-manifest").unwrap();
        verify::run(&f.request, "after-manifest").unwrap();
        if skip {
            assert!(!root.join("owned/thoughtworks").exists());
            assert!(acquired["families"][0]["export"].is_null());
            assert!(acquired["families"][0]["vllm_config"].is_null());
        } else {
            assert_eq!(
                std::fs::read(root.join("owned/tokenizers/granite-h1-hybrid/model.safetensors"))
                    .unwrap(),
                b"full snapshot weight bytes"
            );
            assert!(
                !root
                    .join("owned/tokenizers/granite-h1-hybrid/README.md")
                    .exists()
            );
            for row in f.config["models"].as_array().unwrap() {
                let key = row["key"].as_str().unwrap();
                assert!(
                    root.join("owned/tokenizers")
                        .join(key)
                        .join("tokenizer.json")
                        .is_file()
                );
            }
            assert_eq!(
                std::fs::read(root.join("owned/thoughtworks/sessions.parquet")).unwrap(),
                b"inert protocol fixture; not actual parquet"
            );
        }
        let requests = peer.close();
        assert!(!requests.is_empty());
        assert!(
            requests
                .iter()
                .all(|r| !r.to_ascii_lowercase().contains("bearer \r\n"))
        );
        if skip {
            assert!(requests.iter().all(|r| !r.contains("/api/models/")
                && !r.contains("/datasets/")
                && !r.contains("/config.json")));
        }
        temp.close().unwrap();
    }
}
#[test]
#[ignore = "private actual native helper process used only by the finite whole frontend fixture"]
fn fixture_native_helper_process() {
    let input = std::env::var("COMPETITIVE_FIXTURE_INPUT").unwrap();
    let phase = std::env::var("COMPETITIVE_FIXTURE_PHASE").unwrap_or_default();
    if phase.is_empty() {
        let endpoint = std::env::var("COMPETITIVE_FIXTURE_ENDPOINT").unwrap();
        let url = reqwest::Url::parse(&endpoint).unwrap();
        assert_eq!(url.scheme(), "http");
        assert_eq!(url.host_str(), Some("127.0.0.1"));
        run_using(&["--input".into(), input], Some(clients(&endpoint, false))).unwrap();
    } else {
        run(&[
            "verify-acquired".into(),
            "--input".into(),
            input,
            "--phase".into(),
            phase,
        ])
        .unwrap();
    }
}
#[test]
fn acquisition_terminal_retains_prior_failure_and_refuses_finish_cancel_or_deadline() {
    let future = Instant::now() + Duration::from_secs(10);
    assert!(terminal(Ok(()), future, || false).is_ok());
    assert!(
        terminal(Ok(()), future, || true)
            .unwrap_err()
            .to_string()
            .contains("cancellation")
    );
    assert!(terminal(Ok(()), Instant::now(), || false).is_err());
    assert_eq!(
        terminal(Err(anyhow::anyhow!("prior rows retained")), future, || true)
            .unwrap_err()
            .to_string(),
        "prior rows retained"
    );
}

#[test]
fn actual_isolated_signal_latch_refuses_success_and_retains_rows_before_final_publication() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (ok, _, error) = super::full_frontend::call(
        &std::env::current_exe().unwrap(),
        &root,
        &[
            "--exact".into(),
            "competitive_acquisition::full_chain::fixture_self_signal_terminal".into(),
            "--ignored".into(),
            "--nocapture".into(),
        ],
        "terminal",
    );
    assert!(ok, "isolated terminal assertion failed: {error}");
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(root.join("terminal.json")).unwrap()).unwrap();
    assert_eq!(receipt["status"], "FAILED");
    assert_eq!(receipt["families"].as_array().unwrap().len(), 1);
    assert!(receipt["error"].as_str().unwrap().contains("cancellation"));
    temp.close().unwrap();
}
#[test]
#[ignore = "private isolated self-signal fixture; normal owning attribute invokes this exact child"]
fn fixture_self_signal_terminal() {
    let latch = crate::snapshot_promotion::local_publisher::SignalLatch::install().unwrap();
    // SAFETY: raise targets only this isolated fixture process, with its owned callback installed.
    assert_eq!(unsafe { libc::raise(libc::SIGTERM) }, 0);
    let decision = terminal(Ok(()), Instant::now() + Duration::from_secs(10), || {
        latch.cancelled()
    });
    assert!(decision.is_err());
    let receipt = json!({"status":"FAILED","families":[{"measured":true}],"error":decision.unwrap_err().to_string()});
    std::fs::write("terminal.json", serde_json::to_vec(&receipt).unwrap()).unwrap();
}
