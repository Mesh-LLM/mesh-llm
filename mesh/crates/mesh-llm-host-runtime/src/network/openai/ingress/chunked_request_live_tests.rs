//! Installed observer commitments must ignore HTTP transfer framing.
use super::{BUFFERED, LiveHost, MODEL, assert_receiver_receipt, assert_terminal};
use crate::inference::election::InferenceTarget;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::time::Duration;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_chunked_request_commitments_and_body_copies_exclude_framing() {
    let mut host = Box::pin(LiveHost::start(false, true)).await;
    let body = br#"{ "model": "allowed-model", "messages": [{"role":"user","content":"hello"}] }"#;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    host.targets.targets.insert(
        MODEL.into(),
        vec![InferenceTarget::Local(
            listener.local_addr().unwrap().port(),
        )],
    );
    let backend = tokio::spawn(async move {
        for _ in 0..3 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let received = read_chunked_entity(&mut stream).await;
            assert_eq!(
                received, body,
                "backend entity changed with transfer framing"
            );
            stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", BUFFERED.len()).as_bytes()).await.unwrap();
            stream.write_all(BUFFERED).await.unwrap();
            stream.shutdown().await.unwrap();
        }
    });
    for chunk_size in [1, 11, body.len()] {
        let prior_events = host.events().len();
        let (mut client, handler) = host.connection().await;
        client
            .write_all(&chunked_request(body, chunk_size))
            .await
            .unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(10), client.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        handler.await.unwrap();
        assert!(response.starts_with(b"HTTP/1.1 200"));
        let events = host.events();
        let exchange = &events[prior_events..];
        let received = exchange
            .iter()
            .find(|event| event["phase"] == "request_received")
            .unwrap();
        let selected = exchange
            .iter()
            .find(|event| event["phase"] == "backend_selected")
            .unwrap();
        let digest = hex::encode(Sha256::digest(body));
        for commitment in [
            &received["request_wire_digest"],
            &selected["effective_request_wire_digest"],
        ] {
            assert_eq!(commitment["sha256"], digest);
            assert_eq!(commitment["byte_count"], body.len());
        }
        assert_eq!(
            selected["body"],
            serde_json::from_slice::<Value>(body).unwrap()
        );
        assert_receiver_receipt(&host, received, "openai_exchange_original", body);
        assert_receiver_receipt(&host, selected, "openai_exchange_effective", body);
        assert_terminal(exchange, "completed", &response);
        // Decode an actual installed recipient event through the public SDK.
        let terminal = exchange
            .iter()
            .find(|event| event["phase"] == "exchange_finished")
            .unwrap();
        let typed = mesh_llm_plugin::openai_exchange::OpenAiExchangeEvent::parse(terminal).unwrap();
        assert!(typed.observation_id.is_some());
        assert_eq!(
            typed.evidence_complete,
            terminal["evidence_complete"].as_bool()
        );
        assert_eq!(
            typed.observer_evidence_complete,
            terminal["observer_evidence_complete"].as_bool()
        );
        assert_eq!(typed.evidence_complete, Some(true));
        assert_eq!(typed.observer_evidence_complete, Some(true));
        assert_eq!(typed.admission_denied, Some(false));
        assert_eq!(typed.required_admission_failure, Some(false));
    }
    backend.await.unwrap();
    host.stop().await;
}

fn chunked_request(body: &[u8], chunk_size: usize) -> Vec<u8> {
    let mut wire = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nTrailer: X-Fixture-Trailer\r\nConnection: close\r\n\r\n".to_vec();
    for chunk in body.chunks(chunk_size) {
        wire.extend_from_slice(format!("{:x};fixture=segmented\r\n", chunk.len()).as_bytes());
        wire.extend_from_slice(chunk);
        wire.extend_from_slice(b"\r\n");
    }
    wire.extend_from_slice(b"0\r\nX-Fixture-Trailer: excluded\r\n\r\n");
    wire
}

async fn read_chunked_entity(stream: &mut TcpStream) -> Vec<u8> {
    let mut wire = Vec::new();
    loop {
        let mut chunk = [0u8; 8192];
        let count = stream.read(&mut chunk).await.unwrap();
        assert!(count > 0, "backend request ended before its trailer");
        wire.extend_from_slice(&chunk[..count]);
        if wire.ends_with(b"X-Fixture-Trailer: excluded\r\n\r\n") {
            break;
        }
    }
    let header_end = wire
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .unwrap()
        + 4;
    assert!(
        String::from_utf8_lossy(&wire[..header_end])
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
    );
    let mut cursor = header_end;
    let mut entity = Vec::new();
    loop {
        let end = cursor
            + wire[cursor..]
                .windows(2)
                .position(|bytes| bytes == b"\r\n")
                .unwrap();
        let line = std::str::from_utf8(&wire[cursor..end]).unwrap();
        let count = usize::from_str_radix(line.split(';').next().unwrap(), 16).unwrap();
        if count == 0 {
            return entity;
        }
        cursor = end + 2;
        entity.extend_from_slice(&wire[cursor..cursor + count]);
        cursor += count + 2;
    }
}
