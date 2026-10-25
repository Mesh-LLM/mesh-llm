//! A backend tail gate proves live delivery before the response completes.
use super::{LiveHost, MODEL, SSE, read_entity_request, send_request};
use crate::inference::election::InferenceTarget;
use serde_json::Value;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
};

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_stream_delivers_client_frame_and_observer_bytes_before_backend_tail() {
    let mut host = Box::pin(LiveHost::start(false, true)).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    host.targets
        .targets
        .insert(MODEL.into(), vec![InferenceTarget::Local(port)]);
    let (release_tx, release_rx) = oneshot::channel::<()>();
    let completed = Arc::new(AtomicBool::new(false));
    let backend_completed = completed.clone();
    let backend = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_entity_request(&mut stream).await;
        assert_eq!(
            serde_json::from_slice::<Value>(&request).unwrap()["stream"],
            true
        );
        let first_end = SSE.windows(2).position(|bytes| bytes == b"\n\n").unwrap() + 2;
        stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",SSE.len()).as_bytes()).await.unwrap();
        stream.write_all(&SSE[..first_end]).await.unwrap();
        release_rx.await.unwrap();
        stream.write_all(&SSE[first_end..]).await.unwrap();
        stream.shutdown().await.unwrap();
        backend_completed.store(true, Ordering::Release);
    });
    let (mut client, handler) = host.connection().await;
    send_request(&mut client,"/v1/chat/completions",br#"{"model":"allowed-model","stream":true,"messages":[{"role":"user","content":"hello"}]}"#).await;
    let mut wire = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let mut chunk = [0u8; 4096];
            let len = client.read(&mut chunk).await.unwrap();
            assert!(len > 0);
            wire.extend_from_slice(&chunk[..len]);
            if wire.windows(7).any(|bytes| bytes == b"data: {")
                && wire.windows(2).any(|bytes| bytes == b"\n\n")
            {
                break;
            }
        }
    })
    .await
    .expect("client received no complete SSE frame while backend tail was gated");
    assert!(wire.starts_with(b"HTTP/1.1 200"));
    assert!(String::from_utf8_lossy(&wire).contains("\"content\":\"hi\""));
    assert!(!completed.load(Ordering::Acquire));
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let log = std::fs::read_to_string(host.root.path().join("receipts.jsonl"))
                .unwrap_or_default();
            if log
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .any(|entry| {
                    entry["metadata"]["kind"] == "openai_exchange_response"
                        && entry["complete"] == false
                        && entry["receipt"]["byte_count"]
                            .as_u64()
                            .is_some_and(|count| count > 0)
                })
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("installed observer received no body bytes before the backend tail");
    assert!(!completed.load(Ordering::Acquire));
    assert!(
        !host
            .events()
            .iter()
            .any(|event| event["phase"] == "exchange_finished")
    );
    release_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(5), client.read_to_end(&mut wire))
        .await
        .unwrap()
        .unwrap();
    handler.await.unwrap();
    backend.await.unwrap();
    assert!(completed.load(Ordering::Acquire));
    let events = host.events();
    super::assert_terminal(&events, "completed", &wire);
    super::assert_receiver_receipt(
        &host,
        events.last().unwrap(),
        "openai_exchange_response",
        &super::response_entity(&wire),
    );
    host.stop().await;
}
