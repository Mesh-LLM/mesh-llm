//! concurrency qualification: the peer waits for the entire cohort before replying.
use super::*;
use crate::process::Cancellation;
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    thread,
    time::{Duration, Instant},
};

fn request(socket: &mut TcpStream) -> Value {
    socket.set_nonblocking(false).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    socket
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let count = socket.read(&mut buffer).unwrap();
        assert!(count > 0, "overlap peer closed before complete request");
        bytes.extend_from_slice(&buffer[..count]);
        assert!(bytes.len() <= 262144, "overlap fixture request too large");
        let Some(end) = bytes.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
            continue;
        };
        let head = std::str::from_utf8(&bytes[..end]).unwrap();
        assert!(head.starts_with("POST /v1/chat/completions "));
        let length = head
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                key.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap();
        if bytes.len() >= end + 4 + length {
            return serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
        }
    }
}

#[test]
fn kv_overlap_all_requests_reach_peer_before_any_reply_and_partial_failures_survive() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let count = 4;
    let peer = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut sockets = Vec::new();
        let mut requests = Vec::new();
        while sockets.len() < count {
            assert!(
                Instant::now() < deadline,
                "peer did not observe the complete concurrent cohort"
            );
            match listener.accept() {
                Ok((mut socket, _)) => {
                    requests.push(request(&mut socket));
                    sockets.push(socket);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(2))
                }
                Err(error) => panic!("overlap accept failed: {error}"),
            }
        }
        // No reply is sent until all four complete request bodies are observed.
        for (index, socket) in sockets.iter_mut().enumerate() {
            let status = if index == 1 { 503 } else { 200 };
            let body = if status == 503 {
                json!({"error":"finite cohort failure"})
            } else {
                json!({"choices":[{"message":{"content":"fixture response"}}]})
            }
            .to_string();
            socket.write_all(format!("HTTP/1.1 {status} fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
        }
        requests
    });
    let http = Arc::new(
        Http::new(
            url::Url::parse(&endpoint).unwrap(),
            Duration::from_secs(2),
            Cancellation::default(),
            "kv-fixture",
        )
        .unwrap(),
    );
    let contexts = super::super::kv_requests::overlap("fixture", 1, count);
    let expected = contexts
        .iter()
        .map(|context| context.label.clone())
        .collect::<Vec<_>>();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let results = runtime.block_on(dispatch(http, contexts)).unwrap();
    let requests = peer.join().unwrap();
    assert_eq!(requests.len(), count);
    assert_eq!(
        results
            .iter()
            .map(|result| result.context.label.clone())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        results.iter().filter(|result| result.reply.is_ok()).count(),
        3
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| result
                .reply
                .as_ref()
                .is_err_and(|failure| failure.status == Some(503)))
            .count(),
        1
    );
}

#[test]
fn kv_overlap_cancelled_participants_reach_barrier_without_stranding_siblings() {
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let http = Arc::new(
        Http::new(
            url::Url::parse("http://127.0.0.1:1/v1").unwrap(),
            Duration::from_secs(1),
            cancellation,
            "kv-fixture",
        )
        .unwrap(),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let results = runtime
        .block_on(dispatch(
            http.clone(),
            super::super::kv_requests::overlap("fixture", 1, 4),
        ))
        .unwrap();
    assert_eq!(results.len(), 4);
    assert!(results.iter().all(|result| {
        result
            .reply
            .as_ref()
            .is_err_and(|failure| failure.detail.contains("cancelled"))
    }));
    assert!(
        runtime
            .block_on(dispatch(
                http,
                super::super::kv_requests::overlap("fixture", 1, 1)
            ))
            .is_err()
    );
}
