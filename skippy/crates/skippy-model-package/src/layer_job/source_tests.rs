use crate::layer_job::SourceClient;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
struct Peer {
    origin: String,
    requests: Arc<Mutex<Vec<String>>>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Peer {
    fn new(rows: Vec<(u16, &'static str, Vec<u8>)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let saved = requests.clone();
        let worker = thread::spawn(move || {
            for (code, extra, body) in rows {
                let until = Instant::now() + Duration::from_secs(5);
                let mut socket = loop {
                    match listener.accept() {
                        Ok((socket, _)) => break socket,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < until);
                            thread::sleep(Duration::from_millis(2));
                        }
                        Err(e) => panic!("owned peer accept: {e}"),
                    }
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut request = Vec::new();
                let mut buffer = [0; 1024];
                while !request.windows(4).any(|b| b == b"\r\n\r\n") {
                    let n = socket.read(&mut buffer).unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&buffer[..n]);
                    assert!(request.len() < 16384);
                }
                saved
                    .lock()
                    .unwrap()
                    .push(String::from_utf8(request).unwrap());
                if extra == "__held__" {
                    thread::sleep(Duration::from_millis(500));
                    continue;
                }
                write!(socket,"HTTP/1.1 {code} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n",body.len()).unwrap();
                socket.write_all(&body).unwrap();
            }
        });
        Self {
            origin,
            requests,
            worker: Some(worker),
        }
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}
fn info() -> (u16, &'static str, Vec<u8>) {
    (
        200,
        "",
        serde_json::to_vec(
            &serde_json::json!({"sha":"b".repeat(40),"cardData":{"license":"apache-2.0"}}),
        )
        .unwrap(),
    )
}
#[test]
fn layer_job_source_resolves_once_and_binds_all_optional_metadata_to_immutable_revision() {
    let mut rows = vec![info()];
    rows.extend((0..5).map(|_| (200, "", b"{}".to_vec())));
    let peer = Peer::new(rows);
    let client = SourceClient::fixture(peer.origin.clone()).unwrap();
    let (source, files) = runtime()
        .block_on(client.admit_until(
            "fixture/model",
            "main",
            Instant::now() + Duration::from_secs(10),
        ))
        .unwrap();
    assert_eq!(files.len(), 5);
    assert_eq!(source.metadata_sha256.len(), 5);
    assert!(source.missing.is_empty());
    assert_eq!(source.license.as_deref(), Some("apache-2.0"));
    let requests = peer.requests.lock().unwrap();
    assert!(requests[0].starts_with("GET /api/models/fixture/model/revision/main "));
    for request in &requests[1..] {
        assert!(request.contains(&format!("/resolve/{}/", "b".repeat(40))));
        assert!(!request.contains("authorization:"));
    }
}
#[test]
fn layer_job_optional_entry_missing_is_distinct_from_authentication_and_ambiguous_404() {
    let mut rows = vec![info()];
    rows.extend((0..5).map(|_| (404, "X-Error-Code: EntryNotFound\r\n", Vec::new())));
    let peer = Peer::new(rows);
    let client = SourceClient::fixture(peer.origin.clone()).unwrap();
    let (source, files) = runtime()
        .block_on(client.admit_until(
            "fixture/model",
            "main",
            Instant::now() + Duration::from_secs(10),
        ))
        .unwrap();
    assert!(files.is_empty());
    assert_eq!(source.missing.len(), 5);
    drop(peer);
    for code in [401, 403, 404, 500] {
        let peer = Peer::new(vec![
            info(),
            (code, "", b"private error must not escape".to_vec()),
        ]);
        let client = SourceClient::fixture(peer.origin.clone()).unwrap();
        let error = runtime()
            .block_on(client.admit_until(
                "fixture/model",
                "main",
                Instant::now() + Duration::from_secs(10),
            ))
            .unwrap_err();
        assert!(!error.to_string().contains("private error"));
    }
}
#[test]
fn layer_job_source_rejects_expired_budget_mutable_response_pin_and_cross_origin_redirect() {
    let client = SourceClient::new(None).unwrap();
    assert!(
        runtime()
            .block_on(client.admit_until("fixture/model", "main", Instant::now()))
            .is_err()
    );
    let peer = Peer::new(vec![info()]);
    let client = SourceClient::fixture(peer.origin.clone()).unwrap();
    assert!(
        runtime()
            .block_on(client.admit_until(
                "fixture/model",
                &"a".repeat(40),
                Instant::now() + Duration::from_secs(10)
            ))
            .is_err()
    );
    drop(peer);
    let peer = Peer::new(vec![
        info(),
        (
            302,
            "Location: https://untrusted.invalid/config.json\r\n",
            Vec::new(),
        ),
    ]);
    let client = SourceClient::fixture(peer.origin.clone()).unwrap();
    assert!(
        runtime()
            .block_on(client.admit_until(
                "fixture/model",
                "main",
                Instant::now() + Duration::from_secs(10)
            ))
            .is_err()
    );
}
#[test]
fn layer_job_optional_metadata_follows_only_same_origin_pinned_cache_path() {
    let mut rows = vec![
        info(),
        (
            302,
            "Location: /api/resolve-cache/models/fixture/model/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/config.json?download=true\r\n",
            Vec::new(),
        ),
        (200, "", b"{}".to_vec()),
    ];
    rows.extend((0..4).map(|_| (200, "", b"{}".to_vec())));
    let peer = Peer::new(rows);
    let client = SourceClient::fixture(peer.origin.clone()).unwrap();
    let (source, files) = runtime()
        .block_on(client.admit_until(
            "fixture/model",
            "main",
            Instant::now() + Duration::from_secs(10),
        ))
        .unwrap();
    assert_eq!(files.len(), 5);
    assert!(source.missing.is_empty());
    assert!(peer.requests.lock().unwrap()[2].starts_with("GET /api/resolve-cache/models/fixture/model/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/config.json?download=true "));
}

#[test]
fn layer_job_source_owned_held_response_uses_shared_absolute_deadline() {
    let peer = Peer::new(vec![(200, "__held__", Vec::new())]);
    let client = SourceClient::fixture(peer.origin.clone()).unwrap();
    let result = runtime().block_on(client.admit_until(
        "fixture/model",
        "main",
        Instant::now() + Duration::from_millis(300),
    ));
    assert!(result.is_err());
    assert_eq!(peer.requests.lock().unwrap().len(), 1);
}

#[test]
fn layer_card_actual_license_lookup_prefers_pinned_source_then_first_explicit_base_model() {
    for source_license in [true, false] {
        let source = serde_json::json!({"sha":"a".repeat(40),"cardData":if source_license{serde_json::json!({"license":"apache-2.0","base_model":["base/first","base/second"]})}else{serde_json::json!({"base_model":[{"modelId":"base/first"},"base/second"]})}});
        let mut rows = vec![(200, "", serde_json::to_vec(&source).unwrap())];
        if !source_license {
            rows.push((
                200,
                "",
                serde_json::to_vec(
                    &serde_json::json!({"sha":"b".repeat(40),"cardData":{"license":"mit"}}),
                )
                .unwrap(),
            ));
        }
        let peer = Peer::new(rows);
        let client = SourceClient::fixture(peer.origin.clone()).unwrap();
        let license = runtime().block_on(client.license_until(
            "source/model",
            &"a".repeat(40),
            Instant::now() + Duration::from_secs(5),
        ));
        assert_eq!(
            license.value.as_deref(),
            Some(if source_license { "apache-2.0" } else { "mit" })
        );
        assert_eq!(
            license.repo.as_deref(),
            Some(if source_license {
                "source/model"
            } else {
                "base/first"
            })
        );
        assert!(license.warning.is_none());
        let requests = peer.requests.lock().unwrap();
        assert_eq!(requests.len(), if source_license { 1 } else { 2 });
        if !source_license {
            assert!(requests[1].starts_with("GET /api/models/base/first/revision/main "));
        }
        drop(requests);
    }
}
#[test]
fn layer_card_license_failure_preserves_warning_without_inventing_license() {
    let peer = Peer::new(vec![(401, "", b"{}".to_vec())]);
    let client = SourceClient::fixture(peer.origin.clone()).unwrap();
    let license = runtime().block_on(client.license_until(
        "source/model",
        &"a".repeat(40),
        Instant::now() + Duration::from_secs(5),
    ));
    assert!(license.value.is_none() && license.repo.is_none() && license.warning.is_some());
    assert_eq!(peer.requests.lock().unwrap().len(), 1);
}

#[test]
fn layer_projector_literal_sibling_admits_lfs_and_regular_immutable_bytes() {
    for lfs in [
        None,
        Some(serde_json::json!({"sha256":"a".repeat(64),"size":8})),
    ] {
        let row = serde_json::json!({"sha":"b".repeat(40),"siblings":[{"rfilename":"nested/mm proj.gguf","size":8,"lfs":lfs},{"rfilename":"other.gguf","size":2}]});
        let peer = Peer::new(vec![(200, "", serde_json::to_vec(&row).unwrap())]);
        let client = SourceClient::fixture(peer.origin.clone()).unwrap();
        let observed = runtime()
            .block_on(client.projector_until(
                "fixture/model",
                &"b".repeat(40),
                "nested/mm proj.gguf",
                Instant::now() + Duration::from_secs(5),
            ))
            .unwrap();
        assert_eq!(observed.path, "nested/mm proj.gguf");
        assert_eq!(observed.byte_size, 8);
        assert_eq!(observed.expected_sha256, lfs.map(|_| "a".repeat(64)));
        let requests = peer.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with(&format!(
            "GET /api/models/fixture/model/revision/{}?blobs=true ",
            "b".repeat(40)
        )));
    }
}
#[test]
fn layer_projector_missing_duplicate_malformed_pin_and_held_metadata_refuse() {
    let selected = serde_json::json!({"rfilename":"mm.gguf","size":8});
    let cases = [
        serde_json::json!({"sha":"c".repeat(40),"siblings":[selected.clone()]}),
        serde_json::json!({"sha":"b".repeat(40),"siblings":[]}),
        serde_json::json!({"sha":"b".repeat(40),"siblings":[selected.clone(),selected]}),
        serde_json::json!({"sha":"b".repeat(40),"siblings":[{"rfilename":"mm.gguf","size":8,"lfs":{"sha256":"wrong","size":8}}]}),
        serde_json::json!({"sha":"b".repeat(40),"siblings":[{"rfilename":"mm.gguf","size":8,"lfs":{"sha256":"a".repeat(64),"size":9}}]}),
    ];
    for row in cases {
        let peer = Peer::new(vec![(200, "", serde_json::to_vec(&row).unwrap())]);
        let client = SourceClient::fixture(peer.origin.clone()).unwrap();
        assert!(
            runtime()
                .block_on(client.projector_until(
                    "fixture/model",
                    &"b".repeat(40),
                    "mm.gguf",
                    Instant::now() + Duration::from_secs(5)
                ))
                .is_err()
        );
    }
    let peer = Peer::new(vec![(200, "__held__", Vec::new())]);
    let client = SourceClient::fixture(peer.origin.clone()).unwrap();
    let error = runtime()
        .block_on(client.projector_until(
            "fixture/model",
            &"b".repeat(40),
            "mm.gguf",
            Instant::now() + Duration::from_millis(200),
        ))
        .err()
        .unwrap();
    assert!(error.to_string().contains("deadline"));
    assert_eq!(peer.requests.lock().unwrap().len(), 1);
    for path in [
        "../mm.gguf",
        "a//mm.gguf",
        "mm.gguf/",
        "a\\mm.gguf",
        "mm.bin",
    ] {
        assert!(crate::layer_job::projector::selection(path).is_err());
    }
}
