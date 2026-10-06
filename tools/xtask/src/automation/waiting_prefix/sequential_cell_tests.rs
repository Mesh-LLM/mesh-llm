use super::*;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

fn input(port: u16) -> Input {
    let manifest: Manifest = serde_json::from_value(json!({"metadata":{"dataset_revision":"pinned"},
        "prompts":[{"family":"trajectory-1","bucket":"4k-8k","source_id":"session-1","prompt":"first trace"},
        {"family":"trajectory-2","prompt":"second trace"}]})).unwrap();
    Input {
        schema_version: 1,
        round: 2,
        version: Version::New,
        base_url: format!("http://127.0.0.1:{port}/v1"),
        model: "advertised-local-model".into(),
        output_tokens: 2,
        request_timeout_secs: 1.0,
        timeout_secs: 4,
        readiness_timeout_secs: 0,
        prompt_manifest_sha256: hex::encode(Sha256::digest(serde_json::to_vec(&manifest).unwrap())),
        manifest,
        provenance: json!({"binary_sha256":"caller-declared"})
            .as_object()
            .unwrap()
            .clone(),
    }
}
type PeerResult = Result<Vec<Value>, Box<dyn std::error::Error + Send + Sync>>;
struct Peer(tokio::task::JoinHandle<PeerResult>);
impl Drop for Peer {
    fn drop(&mut self) {
        self.0.abort();
    }
}
async fn peer(mode: &'static str, count: usize) -> (u16, Peer) {
    peer_controlled(mode, count, None).await
}
async fn peer_controlled(
    mode: &'static str,
    count: usize,
    cancellation: Option<Cancellation>,
) -> (u16, Peer) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut bodies = Vec::new();
            for index in 0..count {
                let (mut socket, _) = listener.accept().await?;
                let mut bytes = Vec::new();
                let boundary = loop {
                    let mut buffer = [0; 1024];
                    let n = socket.read(&mut buffer).await?;
                    if n == 0 || bytes.len() + n > 64 * 1024 { return Err("bounded fixture request absent".into()); }
                    bytes.extend_from_slice(&buffer[..n]);
                    if let Some(index) = bytes.windows(4).position(|w| w == b"\r\n\r\n") { break index+4; }
                };
                let header = std::str::from_utf8(&bytes[..boundary])?;
                let length: usize = header.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length").then(|| value.trim().parse::<usize>().ok()).flatten()
                }).ok_or("fixture content length absent")?;
                if length > 32 * 1024 { return Err("fixture body exceeds limit".into()); }
                while bytes.len()-boundary < length {
                    let mut buffer = [0;1024];
                    let n = socket.read(&mut buffer).await?;
                    if n == 0 { return Err("fixture body incomplete".into()); }
                    bytes.extend_from_slice(&buffer[..n]);
                }
                bodies.push(serde_json::from_slice(&bytes[boundary..boundary+length])?);
                if mode == "hold" || (index==2 && (mode=="measured-hold" || mode=="measured-cancel")) {
                    if mode == "measured-cancel" && let Some(cancel) = &cancellation { cancel.cancel(); }
                    std::future::pending::<()>().await;
                }
                let response = if mode == "measured-error" && index==2 {
                    let body="data: {\"error\":{\"message\":\"fixture measured refusal\"}}\n\n";
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len())
                } else if mode == "calibration-failure" {
                    "HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned()
                } else {
                    let content = if index==0 { "excluded-calibration" } else { "measured" };
                    let body = format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{content}\"}},\"finish_reason\":\"stop\"}}]}}\n\ndata: {{\"usage\":{{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{{\"cached_tokens\":0}}}}}}\n\ndata: [DONE]\n\n");
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                };
                socket.write_all(response.as_bytes()).await?;
                socket.shutdown().await?;
            }
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(bodies)
        }).await.map_err(|_| "fixture overall deadline expired")?
    });
    (port, Peer(task))
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn adaptive_serial_cell_excludes_calibration_and_preserves_trace_identity() {
    runtime().block_on(async {
        let (port, mut peer) = peer("success", 3).await;
        let input = input(port);
        let evidence = execute_with(&input, Cancellation::default(), requests::execute).await;
        assert!(evidence.error.is_none(), "{:?}", evidence.error);
        assert_eq!(evidence.round, 2);
        assert_eq!(evidence.version, Version::New);
        assert_eq!(evidence.successful_requests, 2);
        assert_eq!(evidence.requests.len(), 2);
        assert_eq!(
            evidence.prompt_manifest_metadata["dataset_revision"],
            "pinned"
        );
        assert_eq!(evidence.provenance["binary_sha256"], "caller-declared");
        assert_eq!(
            evidence.requests[0]["prompt_provenance"]["source_id"],
            "session-1"
        );
        assert_eq!(evidence.requests[0]["prompt_provenance"]["bucket"], "4k-8k");
        assert_eq!(
            evidence.requests[0]["prompt_sha256"],
            hex::encode(Sha256::digest(b"first trace"))
        );
        assert_eq!(evidence.requests[1]["request_id"], 1);
        assert_eq!(evidence.timing_origin, "measured-cell-start");
        assert!(
            evidence.requests[1]["submitted_ms"].as_f64().unwrap()
                >= evidence.requests[0]["completed_ms"].as_f64().unwrap()
        );
        assert_eq!(
            evidence.requests[0]["content_sha256"],
            hex::encode(Sha256::digest(
                serde_json::to_vec(
                    &json!({"content":"measured","reasoning_content":"","tool_calls":[]})
                )
                .unwrap()
            ))
        );
        assert_ne!(
            evidence.calibration_request.as_ref().unwrap()["requests"][0]["content_sha256"],
            evidence.requests[0]["content_sha256"]
        );
        assert!(!evidence.prefill_telemetry_available);
        let bodies = (&mut peer.0).await.unwrap().unwrap();
        assert_eq!(bodies.len(), 3);
        assert_eq!(bodies[0]["messages"], bodies[1]["messages"]);
        assert_eq!(bodies[2]["messages"][0]["content"], "second trace");
        for body in bodies {
            assert_eq!(body["model"], input.model);
            assert_eq!(body["max_tokens"], 2);
        }
    });
}
#[test]
fn adaptive_serial_cell_calibration_failure_withholds_measurement() {
    runtime().block_on(async {
        let (port, mut peer) = peer("calibration-failure", 1).await;
        let evidence = execute_with(&input(port), Cancellation::default(), requests::execute).await;
        assert!(evidence.error.unwrap().contains("calibration"));
        assert!(evidence.requests.is_empty());
        assert_eq!(evidence.successful_requests, 0);
        assert!(evidence.makespan_ms.is_none());
        assert_eq!((&mut peer.0).await.unwrap().unwrap().len(), 1);
    });
}
#[test]
fn adaptive_serial_cell_cancellation_and_http_deadline_refuse_before_measurement() {
    runtime().block_on(async {
        let cancellation = Cancellation::default();
        cancellation.cancel();
        let evidence = execute_with(&input(1), cancellation, requests::execute).await;
        assert!(evidence.error.unwrap().contains("interrupted"));
        assert!(evidence.calibration_request.is_none());
        assert!(evidence.requests.is_empty());
        let (port, mut peer) = peer("hold", 1).await;
        let mut given = input(port);
        given.request_timeout_secs = 0.2;
        let started = Instant::now();
        let evidence = execute_with(&given, Cancellation::default(), requests::execute).await;
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(evidence.error.unwrap().contains("calibration"));
        assert!(evidence.requests.is_empty());
        peer.0.abort();
        assert!((&mut peer.0).await.unwrap_err().is_cancelled());
    });
}
#[test]
fn adaptive_serial_cell_admission_binds_metadata_and_refuses_remote_or_incomplete_inputs() {
    input(12345).validate().unwrap();
    let mut given = input(12345);
    given
        .manifest
        .metadata
        .insert("dataset_revision".into(), json!("changed"));
    assert!(given.validate().is_err());
    let mut given = input(12345);
    given.base_url = "http://example.com:12345/v1".into();
    assert!(given.validate().is_err());
    let mut given = input(12345);
    given.manifest.prompts[0].family.clear();
    given.prompt_manifest_sha256 =
        hex::encode(Sha256::digest(serde_json::to_vec(&given.manifest).unwrap()));
    assert!(given.validate().is_err());
    let mut given = input(12345);
    given.manifest.prompts[0].prompt.clear();
    given.prompt_manifest_sha256 =
        hex::encode(Sha256::digest(serde_json::to_vec(&given.manifest).unwrap()));
    assert!(given.validate().is_err());
    let mut given = input(12345);
    given.output_tokens = 4097;
    assert!(given.validate().is_err());
    let mut given = input(12345);
    given.timeout_secs = 0;
    assert!(given.validate().is_err());
}

#[test]
fn adaptive_measured_sse_failure_preserves_successful_prior_row() {
    runtime().block_on(async {
        let (port, mut peer) = peer("measured-error", 3).await;
        let evidence = execute_with(&input(port), Cancellation::default(), requests::execute).await;
        assert!(evidence.calibration_request.is_some());
        assert!(evidence.error.is_some());
        assert_eq!(evidence.successful_requests, 1);
        assert_eq!(evidence.requests.len(), 2);
        assert_eq!(evidence.requests[0]["request_id"], 0);
        assert!(evidence.requests[0]["error"].is_null());
        assert!(evidence.requests[1]["error"].is_string());
        assert_eq!((&mut peer.0).await.unwrap().unwrap().len(), 3);
    });
}
#[test]
fn adaptive_measured_inflight_cancellation_and_whole_cell_budget_preserve_prior_rows() {
    runtime().block_on(async {
        for mode in ["measured-cancel", "measured-hold"] {
            let cancel = Cancellation::default();
            let (port, mut peer) = peer_controlled(mode, 3, Some(cancel.clone())).await;
            let mut given = input(port);
            given.timeout_secs = if mode == "measured-cancel" { 4 } else { 1 };
            given.request_timeout_secs = 2.0;
            let started = Instant::now();
            let evidence = execute_with(&given, cancel.clone(), requests::execute).await;
            if mode == "measured-cancel" {
                assert!(
                    cancel.is_cancelled(),
                    "peer must trigger in-flight cancellation"
                );
            }
            assert!(started.elapsed() < Duration::from_secs(3));
            assert!(evidence.calibration_request.is_some());
            assert!(evidence.error.is_some());
            assert_eq!(evidence.successful_requests, 1);
            assert!(evidence.requests[0]["error"].is_null());
            assert!(evidence.requests.len() <= 2);
            peer.0.abort();
            assert!((&mut peer.0).await.unwrap_err().is_cancelled());
        }
    });
}
