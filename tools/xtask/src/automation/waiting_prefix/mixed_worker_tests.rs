use super::*;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Notify,
};

type FixtureError = Box<dyn std::error::Error + Send + Sync>;
struct Peer(tokio::task::JoinHandle<Vec<Value>>);
impl Drop for Peer {
    fn drop(&mut self) {
        self.0.abort();
    }
}
async fn read_body(socket: &mut TcpStream) -> Result<Value, FixtureError> {
    let mut bytes = Vec::new();
    let boundary = loop {
        let mut buffer = [0; 1024];
        let size = socket.read(&mut buffer).await?;
        if size == 0 || bytes.len() + size > 65536 {
            return Err("fixture body bound".into());
        }
        bytes.extend_from_slice(&buffer[..size]);
        if let Some(index) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let length: usize = std::str::from_utf8(&bytes[..boundary])?
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())
                .flatten()
        })
        .ok_or("fixture length absent")?;
    if length > 32768 {
        return Err("fixture declared body bound".into());
    }
    while bytes.len() - boundary < length {
        let mut buffer = [0; 1024];
        let size = socket.read(&mut buffer).await?;
        if size == 0 {
            return Err("fixture incomplete body".into());
        }
        bytes.extend_from_slice(&buffer[..size]);
        if bytes.len() > 65536 {
            return Err("fixture body bound".into());
        }
    }
    Ok(serde_json::from_slice(&bytes[boundary..boundary + length])?)
}
async fn peer(mode: &'static str, cancel: Cancellation) -> (u16, Peer) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(4),async {
            let release=Arc::new(Notify::new());
            let mut tasks=tokio::task::JoinSet::new();
            let mut bodies=Vec::new();
            for _ in 0..3 {
                let (mut socket,_)=listener.accept().await.unwrap();
                let body=read_body(&mut socket).await.unwrap();
                let prompt=body["messages"][0]["content"].as_str().unwrap().to_owned();
                bodies.push(body);
                let release=release.clone();
                let cancel=cancel.clone();
                tasks.spawn(async move {
                    if prompt=="anchor" {
                        if mode=="cancel" { cancel.cancel(); std::future::pending::<()>().await; }
                        release.notified().await;
                    }
                    if prompt=="prefill" { release.notify_one(); }
                    let response=if mode=="error" && prompt=="prefill" {
                        "HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned()
                    } else {
                        let sse="data: {\"choices\":[{\"delta\":{\"content\":\"one\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"two\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n";
                        format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",sse.len())
                    };
                    socket.write_all(response.as_bytes()).await.unwrap();
                    socket.shutdown().await.unwrap();
                });
            }
            while let Some(result)=tasks.join_next().await { result.unwrap(); }
            bodies
        }).await.unwrap()
    });
    (port, Peer(task))
}
fn input(port: u16) -> Input {
    let record = |prompt: &str| PromptRecord {
        family: "trace".into(),
        prompt: prompt.into(),
        provenance: json!({"source_id":prompt}).as_object().unwrap().clone(),
    };
    let mut input = Input {
        schema_version: 1,
        round: 1,
        version: Version::New,
        base_url: format!("http://127.0.0.1:{port}/v1"),
        model: "actual-model".into(),
        request_timeout_secs: 1.0,
        timeout_secs: 2,
        readiness_timeout_secs: 0,
        warmup: record("warmup"),
        requests: vec![
            Request {
                role: Role::Anchor,
                request_index: 0,
                prompt: record("anchor"),
                output_tokens: 128,
                delay_ms: 0.0,
            },
            Request {
                role: Role::Prefill,
                request_index: 1,
                prompt: record("prefill"),
                output_tokens: 8,
                delay_ms: 50.0,
            },
        ],
        suppressed_token_ids: vec![7],
        manifest_metadata: Map::new(),
        provenance: Map::new(),
        workload_sha256: String::new(),
    };
    input.workload_sha256 = input.workload_sha().unwrap();
    input
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}
#[test]
fn mixed_concurrent_anchor_waits_for_delayed_prefill_and_preserves_partial_failures() {
    runtime().block_on(async {
        for mode in ["success", "error"] {
            let cancel = Cancellation::default();
            let (port, mut peer) = peer(mode, cancel.clone()).await;
            let input = input(port);
            let output =
                tokio::time::timeout(Duration::from_secs(3), execute(&input, cancel.clone()))
                    .await
                    .unwrap();
            let bodies = (&mut peer.0).await.unwrap();
            assert_eq!(bodies.len(), 3);
            assert_eq!(bodies[0]["max_tokens"], 4);
            assert_eq!(bodies[1]["max_tokens"], 128);
            assert_eq!(bodies[2]["max_tokens"], 8);
            for body in bodies {
                assert_eq!(body["model"], "actual-model");
                assert_eq!(body["logit_bias"]["7"], -100);
            }
            assert_eq!(output["requests"].as_array().unwrap().len(), 2);
            assert_eq!(
                output["requests"][0]["prompt_provenance"]["source_id"],
                "anchor"
            );
            assert_eq!(
                output["requests"][0]["content_gaps_ms"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
            assert!(
                output["requests"][0]["completed_ms"].as_f64().unwrap()
                    >= output["requests"][1]["submitted_ms"].as_f64().unwrap()
            );
            assert!(output["requests"][1]["submitted_ms"].as_f64().unwrap() >= 50.0);
            assert_eq!(
                output["successful_requests"],
                if mode == "success" { 2 } else { 1 }
            );
            assert_eq!(output["error"].is_null(), mode == "success");
        }
    });
}
#[test]
fn mixed_inflight_cancel_and_whole_cell_deadline_bound_scheduled_futures() {
    runtime().block_on(async {
        for mode in ["cancel", "deadline"] {
            let cancel = Cancellation::default();
            let (port, mut peer) = peer(mode, cancel.clone()).await;
            let mut input = input(port);
            if mode == "deadline" {
                input.timeout_secs = 1;
                input.requests[1].delay_ms = 5000.0;
                input.workload_sha256 = input.workload_sha().unwrap();
            }
            let started = Instant::now();
            let output =
                tokio::time::timeout(Duration::from_secs(2), execute(&input, cancel.clone()))
                    .await
                    .unwrap();
            assert_eq!(
                cancel.is_cancelled(),
                mode == "cancel",
                "peer cancellation must differ from deadline failure"
            );
            assert!(!output["error"].is_null());
            assert_eq!(output["requests"].as_array().unwrap().len(), 2);
            assert_eq!(output["successful_requests"], 0);
            assert!(output["requests"][0]["submitted_ms"].is_number());
            assert!(output["requests"][1]["submitted_ms"].is_null());
            assert!(started.elapsed() < Duration::from_secs(2));
            peer.0.abort();
            assert!((&mut peer.0).await.unwrap_err().is_cancelled());
        }
        let mut remote = input(1);
        remote.base_url = "http://example.com/v1".into();
        assert!(remote.validate().is_err());
        let mut mutated = input(1);
        mutated.requests[0].prompt.prompt.push('x');
        assert!(mutated.validate().is_err());
    });
}
