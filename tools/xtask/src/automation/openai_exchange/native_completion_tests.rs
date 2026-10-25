use super::*;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn final_event() -> Value {
    json!({"stop":true,"content":"","tokens_predicted":2,"tokens_evaluated":40,
        "tokens_cached":41,"truncated":false,"model":"fixture","stop_type":"limit",
        "timings":{"prompt_n":10,"cache_n":30,"prompt_ms":2.5,"predicted_n":2,"predicted_ms":3.5}})
}

fn stream_bytes(final_event: Value) -> Vec<u8> {
    format!(
        "data: {}\n\ndata: {}\n\ndata: {}\n\n",
        json!({"stop":false,"content":"h"}),
        json!({"stop":false,"content":"i"}),
        final_event
    )
    .into_bytes()
}

fn decode(value: Value) -> Result<Evidence, String> {
    let mut decoder = NativeDecoder::new(true, 128);
    decoder.consume(&stream_bytes(value), Duration::from_millis(5))?;
    decoder.finish(Duration::from_millis(9))
}

#[test]
fn native_sse_counts_and_hashes_survive_arbitrary_frame_partition() {
    let bytes = stream_bytes(final_event());
    for size in [1, 3, 17, bytes.len()] {
        let mut decoder = NativeDecoder::new(true, 128);
        for bytes in bytes.chunks(size) {
            decoder.consume(bytes, Duration::from_millis(5)).unwrap();
        }
        let evidence = decoder.finish(Duration::from_millis(9)).unwrap();
        assert_eq!(evidence.tokens_predicted, 2);
        assert_eq!(evidence.tokens_evaluated, 40);
        // Native tokens_cached includes post-completion occupancy, unlike cache_n.
        assert_eq!(evidence.tokens_cached, Some(41));
        assert_eq!((evidence.prompt_n, evidence.cache_n), (10, 30));
        assert_eq!(evidence.ttft_seconds, Some(0.005));
        assert_eq!(evidence.content_events, 2);
        assert_eq!(evidence.content_sha256, hex::encode(Sha256::digest(b"hi")));
        assert_eq!(
            evidence.first_generated_sha256,
            Some(hex::encode(Sha256::digest(b"h")))
        );
    }
}

#[test]
fn serial_json_has_observed_timings_without_invented_ttft() {
    let mut event = final_event();
    event["content"] = json!("hi");
    let mut decoder = NativeDecoder::new(false, 2);
    decoder
        .consume(
            &serde_json::to_vec(&event).unwrap(),
            Duration::from_millis(1),
        )
        .unwrap();
    assert!(!decoder.terminal());
    let evidence = decoder.finish(Duration::from_millis(9)).unwrap();
    assert_eq!(evidence.ttft_seconds, None);
    assert_eq!(evidence.first_generated_sha256, None);
    assert_eq!(evidence.prompt_ms, 2.5);
    assert_eq!(evidence.predicted_ms, 3.5);
    assert_eq!(evidence.content_sha256, hex::encode(Sha256::digest(b"hi")));
}

#[test]
fn native_receipt_refuses_missing_or_contradictory_usage_without_chunk_fallback() {
    for field in ["tokens_predicted", "tokens_evaluated", "timings"] {
        let mut event = final_event();
        event.as_object_mut().unwrap().remove(field);
        assert!(decode(event).is_err(), "missing {field} admitted");
    }
    for (field, value) in [
        ("tokens_predicted", json!(0)),
        ("tokens_predicted", json!(129)),
        ("tokens_evaluated", json!(0)),
        ("tokens_predicted", json!(-1)),
    ] {
        let mut event = final_event();
        event[field] = value;
        assert!(decode(event).is_err());
    }
    let mut event = final_event();
    event["timings"]["predicted_n"] = json!(3);
    assert!(
        decode(event)
            .unwrap_err()
            .contains("inconsistent token counters")
    );
    let mut event = final_event();
    event["timings"]["cache_n"] = json!(41);
    assert!(decode(event).is_err());
}

#[test]
fn native_receipt_refuses_truncation_errors_and_invalid_observed_timings() {
    for value in [json!(true), Value::Null] {
        let mut event = final_event();
        event["truncated"] = value;
        assert!(decode(event).unwrap_err().contains("prompt admission"));
    }
    for field in ["prompt_ms", "predicted_ms"] {
        let mut event = final_event();
        event["timings"][field] = json!(-1);
        assert!(decode(event).unwrap_err().contains("invalid timings"));
    }
    let mut event = final_event();
    event["error"] = json!({"message":"secret provider diagnostic"});
    assert_eq!(
        decode(event).unwrap_err(),
        "native completion returned a server error"
    );
}

#[test]
fn native_terminal_protocol_refuses_done_only_malformed_empty_and_oversized_streams() {
    for bytes in [b"data: [DONE]\n".as_slice(), b"data: {\n", b"data: \xff\n"] {
        let mut decoder = NativeDecoder::new(true, 128);
        assert!(decoder.consume(bytes, Duration::ZERO).is_err());
    }
    let mut decoder = NativeDecoder::new(true, 128);
    decoder
        .consume(
            b"data: {\"stop\":false,\"content\":\"hi\"}\n",
            Duration::ZERO,
        )
        .unwrap();
    assert!(decoder.finish(Duration::from_secs(1)).is_err());
    let mut decoder = NativeDecoder::new(true, 128);
    decoder
        .consume(
            format!("data: {}\n", final_event()).as_bytes(),
            Duration::ZERO,
        )
        .unwrap();
    assert!(
        decoder
            .finish(Duration::from_secs(1))
            .unwrap_err()
            .contains("without generated content")
    );
    let mut decoder = NativeDecoder::new(true, 128);
    assert!(
        decoder
            .consume(&vec![b'x'; LINE_LIMIT + 1], Duration::ZERO)
            .is_err()
    );
    let mut decoder = NativeDecoder::new(false, 1);
    assert!(
        decoder
            .consume(&vec![b'x'; BODY_LIMIT + 1], Duration::ZERO)
            .is_err()
    );
}

struct Peer {
    url: String,
    task: Option<tokio::task::JoinHandle<Result<String, String>>>,
}
impl Drop for Peer {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}
impl Peer {
    async fn start(status: u16, body: Vec<u8>, hold: bool) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(5), async {
                let (mut socket, _) = listener.accept().await.map_err(|error| error.to_string())?;
                let mut request = Vec::new();
                loop {
                    let mut buffer = [0; 4096];
                    let count = socket.read(&mut buffer).await.map_err(|error| error.to_string())?;
                    if count == 0 { return Err("request ended early".into()); }
                    request.extend_from_slice(&buffer[..count]);
                    if request.len() > 128 * 1024 { return Err("fixture request oversized".into()); }
                    if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                        let headers = std::str::from_utf8(&request[..end]).map_err(|error| error.to_string())?;
                        let length = headers.lines().find_map(|line| line.to_ascii_lowercase().strip_prefix("content-length:").map(str::trim).map(str::to_owned)).ok_or("missing content length")?.parse::<usize>().map_err(|error| error.to_string())?;
                        if request.len() >= end + 4 + length { break; }
                    }
                }
                let length = if hold { 100 } else { body.len() };
                socket.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n").as_bytes()).await.map_err(|error| error.to_string())?;
                if hold {
                    // A dropped request must drop its owned connection future, causing EOF.
                    let mut byte = [0; 1];
                    if socket.read(&mut byte).await.map_err(|error| error.to_string())? != 0 { return Err("expected dropped client EOF".into()); }
                } else { socket.write_all(&body).await.map_err(|error| error.to_string())?; }
                String::from_utf8(request).map_err(|error| error.to_string())
            }).await.map_err(|_| "fixture peer deadline".to_owned())?
        });
        Self {
            url,
            task: Some(task),
        }
    }
    async fn finish(&mut self) -> String {
        let result =
            tokio::time::timeout(Duration::from_secs(6), self.task.as_mut().unwrap()).await;
        if result.is_err() {
            let task = self.task.take().unwrap();
            task.abort();
            let _ = task.await;
            panic!("owned peer did not finish within deadline");
        }
        let _ = self.task.take();
        result.unwrap().unwrap().unwrap()
    }
}

#[tokio::test]
async fn native_actual_http_projects_greedy_json_and_sse_without_authorization() {
    for streaming in [false, true] {
        let mut event = final_event();
        event["content"] = json!(if streaming { "" } else { "hi" });
        let body = if streaming {
            stream_bytes(event)
        } else {
            serde_json::to_vec(&event).unwrap()
        };
        let mut peer = Peer::start(200, body, false).await;
        let result = tokio::time::timeout(
            Duration::from_secs(4),
            request(&peer.url, "fixed prompt", 2, streaming),
        )
        .await;
        let captured = peer.finish().await;
        let evidence = result.unwrap().unwrap();
        assert_eq!(evidence.tokens_predicted, 2);
        let (headers, body) = captured.split_once("\r\n\r\n").unwrap();
        assert!(headers.starts_with("POST /completion HTTP/1.1"));
        assert!(!headers.to_ascii_lowercase().contains("authorization:"));
        let body: Value = serde_json::from_str(body).unwrap();
        assert_eq!(
            body,
            json!({"prompt":"fixed prompt","n_predict":2,"temperature":0,"top_k":1,"cache_prompt":true,"stream":streaming})
        );
    }
}

#[tokio::test]
async fn native_actual_http_refuses_status_and_dropping_held_request_closes_owned_connection() {
    let mut peer = Peer::start(201, Vec::new(), false).await;
    let result = tokio::time::timeout(
        Duration::from_secs(4),
        request(&peer.url, "fixed prompt", 1, false),
    )
    .await;
    peer.finish().await;
    assert!(
        result
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("HTTP 201")
    );
    let mut peer = Peer::start(200, Vec::new(), true).await;
    let result = tokio::time::timeout(
        Duration::from_millis(100),
        request(&peer.url, "fixed prompt", 1, false),
    )
    .await;
    peer.finish().await;
    assert!(result.is_err());
}

#[tokio::test]
async fn cache_openai_exact_status_is_distinct_from_existing_replay_status_policy() {
    let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":30}}}\n\ndata: [DONE]\n\n";
    let mut peer = Peer::start(201, body.to_vec(), false).await;
    let result = tokio::time::timeout(
        Duration::from_secs(4),
        super::super::cache_request(&peer.url, &json!({})),
    )
    .await;
    // A rejected status may drop the client before body delivery. Own and cancel
    // this peer before asserting; the status-only observation is sufficient.
    if let Some(task) = peer.task.take() {
        task.abort();
        let _ = task.await;
    }
    assert!(
        result
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("HTTP 201")
    );
    let mut peer = Peer::start(201, body.to_vec(), false).await;
    let result = tokio::time::timeout(
        Duration::from_secs(4),
        super::super::request(&peer.url, &json!({}), false),
    )
    .await;
    peer.finish().await;
    assert_eq!(result.unwrap().unwrap().completion_tokens, 1);
}
