use super::super::progress::{PackageFileProgress, PackageProgress};
use super::*;
use hf_hub::progress::{ProgressEvent, ProgressHandler};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    sync::atomic::{AtomicBool, Ordering},
    thread::JoinHandle,
};

const COMMIT: &str = "0123456789012345678901234567890123456789";
fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn write_cached_package_snapshot(snapshot: &Path, layer_sha: String) {
    fs::create_dir_all(snapshot.join("shared")).unwrap();
    fs::create_dir_all(snapshot.join("layers")).unwrap();
    fs::write(snapshot.join("shared/metadata.gguf"), b"metadata").unwrap();
    fs::write(snapshot.join("layers/layer-000.gguf"), b"layer").unwrap();
    fs::write(
        snapshot.join("model-package.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "model_id": "model-a",
            "source_model": {
                "path": "model-a.gguf",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "files": [
                    {
                        "path": "model-a.gguf",
                        "size_bytes": 123,
                        "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    }
                ]
            },
            "format": "layer-package",
            "layer_count": 1,
            "activation_width": 4096,
            "shared": {
                "metadata": {
                    "path": "shared/metadata.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 8,
                    "sha256": sha256_hex(b"metadata")
                },
                "embeddings": {
                    "path": "shared/metadata.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 8,
                    "sha256": sha256_hex(b"metadata")
                },
                "output": {
                    "path": "shared/metadata.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 8,
                    "sha256": sha256_hex(b"metadata")
                }
            },
            "layers": [
                {
                    "layer_index": 0,
                    "path": "layers/layer-000.gguf",
                    "tensor_count": 1,
                    "tensor_bytes": 1,
                    "artifact_bytes": 5,
                    "sha256": layer_sha
                }
            ],
            "skippy_abi_version": "0.1.0",
        }))
        .unwrap(),
    )
    .unwrap();
}

// A bounded local HF endpoint exercises the real sync client/cache and joins on drop.
struct PackageServer {
    endpoint: String,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}
impl PackageServer {
    fn new(files: BTreeMap<String, Vec<u8>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let shutdown = stop.clone();
        let worker = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                let mut stream = stream.unwrap();
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&mut stream);
                let mut request = String::new();
                reader.read_line(&mut request).unwrap();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                        break;
                    }
                }
                let path = request.split_whitespace().nth(1).unwrap();
                let name = path.split_once("/resolve/main/").map(|(_, name)| name);
                let body = name.and_then(|name| files.get(name));
                if let Some(body) = body {
                    write!(stream, "HTTP/1.1 200 OK\r\nETag: \"{}\"\r\nX-Repo-Commit: {COMMIT}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", sha256_hex(body), body.len()).unwrap();
                    if !request.starts_with("HEAD ") {
                        stream.write_all(body).unwrap();
                    }
                } else {
                    stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                }
            }
        });
        Self {
            endpoint,
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for PackageServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.endpoint.trim_start_matches("http://"));
        self.worker.take().unwrap().join().unwrap();
    }
}

#[derive(Clone, Default)]
struct RecordingProgress(Arc<Mutex<Vec<String>>>);
impl PackageProgress for RecordingProgress {
    fn batch(&self, _: &str, total: usize) -> Arc<dyn PackageProgress> {
        self.0.lock().unwrap().push(format!("batch:{total}"));
        Arc::new(self.clone())
    }
    fn file(
        &self,
        _: &str,
        file: &str,
        _: Option<u64>,
        completed: usize,
    ) -> Arc<dyn PackageFileProgress> {
        self.0
            .lock()
            .unwrap()
            .push(format!("file:{file}:{completed}"));
        Arc::new(RecordingFile {
            file: file.into(),
            events: self.0.clone(),
        })
    }
}
struct RecordingFile {
    file: String,
    events: Arc<Mutex<Vec<String>>>,
}
impl ProgressHandler for RecordingFile {
    fn on_progress(&self, _: &ProgressEvent) {
        self.events
            .lock()
            .unwrap()
            .push(format!("progress:{}", self.file));
    }
}
impl PackageFileProgress for RecordingFile {
    fn ensuring(&self) {
        self.events
            .lock()
            .unwrap()
            .push(format!("ensuring:{}", self.file));
    }
    fn ready(&self, _: &Path) {
        self.events
            .lock()
            .unwrap()
            .push(format!("ready:{}", self.file));
    }
}
impl Drop for RecordingFile {
    fn drop(&mut self) {
        self.events
            .lock()
            .unwrap()
            .push(format!("drop:{}", self.file));
    }
}

fn fixture_files() -> BTreeMap<String, Vec<u8>> {
    let root = tempfile::tempdir().unwrap();
    write_cached_package_snapshot(root.path(), sha256_hex(b"layer"));
    [
        "model-package.json",
        "shared/metadata.gguf",
        "layers/layer-000.gguf",
    ]
    .into_iter()
    .map(|name| (name.to_string(), fs::read(root.path().join(name)).unwrap()))
    .collect()
}
fn acquisition(cache: &Path, endpoint: &str) -> PackageAcquisition {
    let client_cache = cache.to_path_buf();
    let endpoint = endpoint.to_string();
    PackageAcquisition::new(cache.to_path_buf(), move || {
        let _ = skippy_model_hf::configure_hf_tls_provider();
        hf_hub::HFClientSync::from_inner(
            hf_hub::HFClientBuilder::new()
                .cache_dir(client_cache.clone())
                .endpoint(endpoint.clone())
                .retry_max_attempts(0)
                .build()?,
        )
        .map_err(Into::into)
    })
}

#[test]
fn download_checks_integrity_reports_order_and_reuses_cache_without_client() {
    let server = PackageServer::new(fixture_files());
    let cache = tempfile::tempdir().unwrap();
    let progress = RecordingProgress::default();
    let mut acquisition = acquisition(cache.path(), &server.endpoint);
    acquisition.progress = Arc::new(progress.clone());
    let local = acquisition
        .resolve_hf_package_to_local("hf://owner/repo", 0, 1, false, false)
        .unwrap();
    assert_eq!(
        fs::read(Path::new(&local).join("layers/layer-000.gguf")).unwrap(),
        b"layer"
    );
    let events = progress.0.lock().unwrap().clone();
    assert!(events.contains(&"batch:3".to_string()), "{events:?}");
    for file in [
        "model-package.json",
        "shared/metadata.gguf",
        "layers/layer-000.gguf",
    ] {
        let position = |kind: &str| {
            events
                .iter()
                .position(|event| *event == format!("{kind}:{file}"))
                .unwrap()
        };
        assert!(position("ensuring") < position("ready"));
        assert!(position("ready") < position("drop"));
    }
    assert!(events.iter().any(|event| event.starts_with("progress:")));
    acquisition.build_client = Arc::new(|| panic!("cached resolution constructed a client"));
    let cached = acquisition
        .resolve_hf_package_to_local("hf://owner/repo", 0, 1, false, false)
        .unwrap();
    assert_eq!(cached, local);
    fs::write(Path::new(&local).join("shared/metadata.gguf"), b"metadota").unwrap();
    let error = acquisition
        .resolve_hf_package_to_local("hf://owner/repo", 0, 0, false, false)
        .unwrap_err();
    assert!(error.to_string().contains("checksum mismatch"), "{error:#}");
}

#[test]
fn failed_transfer_drops_observer_without_ready_and_releases_download_lock() {
    let mut files = fixture_files();
    files.remove("shared/metadata.gguf");
    let server = PackageServer::new(files);
    let cache = tempfile::tempdir().unwrap();
    let progress = RecordingProgress::default();
    let mut acquisition = acquisition(cache.path(), &server.endpoint);
    acquisition.progress = Arc::new(progress.clone());
    assert!(
        acquisition
            .resolve_hf_package_to_local("hf://owner/repo", 0, 1, false, false)
            .is_err()
    );
    let events = progress.0.lock().unwrap().clone();
    assert!(events.contains(&"ensuring:shared/metadata.gguf".to_string()));
    assert!(events.contains(&"drop:shared/metadata.gguf".to_string()));
    assert!(!events.contains(&"ready:shared/metadata.gguf".to_string()));
    // A later acquisition can proceed after the failure; no poisoned/stuck lock.
    acquisition.build_client = Arc::new(|| anyhow::bail!("next acquisition reached client"));
    let error = acquisition
        .resolve_hf_package_to_local("hf://owner/repo", 0, 1, false, false)
        .unwrap_err();
    assert_eq!(error.to_string(), "next acquisition reached client");
}
