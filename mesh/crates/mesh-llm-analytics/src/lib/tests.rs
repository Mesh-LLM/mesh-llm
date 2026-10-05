//! End-to-end coverage for the flush loop.
//!
//! These drive [`flush_loop`] directly against a throwaway HTTP server rather
//! than through [`init`], which keeps the tests off the real `~/.mesh-llm`
//! and free of the process-global reporter.

use super::*;
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// A one-shot HTTP server that returns the first request body it receives.
async fn capture_one_request() -> (SocketAddr, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        let Ok((mut stream, _)) = listener.accept().await else {
            return;
        };
        let mut raw = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let Ok(read) = stream.read(&mut buffer).await else {
                return;
            };
            if read == 0 {
                break;
            }
            raw.extend_from_slice(&buffer[..read]);

            let text = String::from_utf8_lossy(&raw).into_owned();
            let Some((headers, body)) = text.split_once("\r\n\r\n") else {
                continue;
            };
            let declared = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.trim()
                        .eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            if body.len() >= declared {
                let _ = stream
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                    .await;
                let _ = stream.flush().await;
                let _ = tx.send(body.to_owned());
                return;
            }
        }
    });

    (addr, rx)
}

#[tokio::test]
async fn queued_events_are_delivered_as_a_batch_on_shutdown() {
    let (addr, body) = capture_one_request().await;
    let endpoint = format!("http://{addr}/batch/");
    let (sender, receiver) = mpsc::channel(QUEUE_CAPACITY);

    let worker = tokio::spawn(flush_loop(
        receiver,
        "phc_test_key".to_owned(),
        "install-under-test".to_owned(),
        endpoint,
    ));

    for event in [Event::ServeStarted, Event::ServeStopped] {
        sender
            .send(Envelope {
                event,
                properties: Properties::new(),
                captured_at: Utc::now(),
            })
            .await
            .expect("queue event");
    }
    drop(sender);
    worker.await.expect("flush loop");

    let raw = tokio::time::timeout(std::time::Duration::from_secs(5), body)
        .await
        .expect("server responded")
        .expect("body");
    let payload: serde_json::Value = serde_json::from_str(&raw).expect("json body");

    assert_eq!(payload["api_key"], "phc_test_key");
    let batch = payload["batch"].as_array().expect("batch array");
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0]["event"], "serve_started");
    assert_eq!(batch[1]["event"], "serve_stopped");
}

#[tokio::test]
async fn delivered_events_carry_the_privacy_properties_and_no_free_text() {
    let (addr, body) = capture_one_request().await;
    let endpoint = format!("http://{addr}/batch/");
    let (sender, receiver) = mpsc::channel(QUEUE_CAPACITY);

    let worker = tokio::spawn(flush_loop(
        receiver,
        "phc_test_key".to_owned(),
        "install-under-test".to_owned(),
        endpoint,
    ));
    sender
        .send(Envelope {
            event: Event::ModelLoaded,
            properties: Properties::new()
                .with("model", Label::sanitize_or_redact("/home/dan/secret.gguf")),
            captured_at: Utc::now(),
        })
        .await
        .expect("queue event");
    drop(sender);
    worker.await.expect("flush loop");

    let raw = tokio::time::timeout(std::time::Duration::from_secs(5), body)
        .await
        .expect("server responded")
        .expect("body");

    // The path never reaches the wire, in any form.
    assert!(!raw.contains("secret.gguf"), "model path leaked: {raw}");
    assert!(!raw.contains("/home/dan"), "home directory leaked: {raw}");

    let payload: serde_json::Value = serde_json::from_str(&raw).expect("json body");
    let properties = &payload["batch"][0]["properties"];
    assert_eq!(properties["model"], "redacted");
    assert_eq!(properties["$geoip_disable"], true);
    assert!(properties["$ip"].is_null());
    assert_eq!(properties["distinct_id"], "install-under-test");
    assert_eq!(properties["$lib"], LIB_NAME);
}

#[tokio::test]
async fn an_unreachable_endpoint_is_not_an_error() {
    // Port 1 on loopback refuses instantly. The loop must still finish.
    let (sender, receiver) = mpsc::channel(QUEUE_CAPACITY);
    let worker = tokio::spawn(flush_loop(
        receiver,
        "phc_test_key".to_owned(),
        "install-under-test".to_owned(),
        "http://127.0.0.1:1/batch/".to_owned(),
    ));
    sender
        .send(Envelope {
            event: Event::CliCommand,
            properties: Properties::new(),
            captured_at: Utc::now(),
        })
        .await
        .expect("queue event");
    drop(sender);

    tokio::time::timeout(std::time::Duration::from_secs(10), worker)
        .await
        .expect("flush loop terminated")
        .expect("flush loop did not panic");
}

#[tokio::test]
async fn capture_is_inert_before_init() {
    // No reporter has been installed in this process, so capture must be a
    // no-op rather than a panic, and shutdown must return immediately.
    capture(Event::CliCommand, Properties::new());
    tokio::time::timeout(std::time::Duration::from_secs(1), shutdown())
        .await
        .expect("shutdown returned promptly");
}
