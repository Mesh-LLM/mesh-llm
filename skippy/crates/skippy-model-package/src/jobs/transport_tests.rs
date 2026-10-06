use super::*;
use futures::StreamExt as _;
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::mpsc,
    thread,
    time::Duration,
};
const CREDENTIAL: &str = "inert-private-credential";
fn spec() -> JobSpec {
    JobSpec {
        docker_image: "image@sha256:declared".into(),
        command: vec!["command".into()],
        arguments: vec!["argument".into()],
        environment: HashMap::new(),
        secrets: HashMap::from([("HF_TOKEN".into(), CREDENTIAL.into())]),
        flavor: "cpu-basic".into(),
        timeout_seconds: 60,
        volumes: vec![JobVolume {
            volume_type: "bucket".into(),
            source: "readonly-source".into(),
            mount_path: "/input".into(),
            read_only: Some(true),
            revision: Some("immutable-revision".into()),
        }],
    }
}
pub(super) fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}
pub(super) struct Peer {
    endpoint: String,
    pub(super) requests: mpsc::Receiver<Vec<u8>>,
    release: Option<mpsc::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Peer {
    fn new(response: Vec<u8>, hold: bool) -> Self {
        Self::many(vec![(response, hold)])
    }
    pub(super) fn many(responses: Vec<(Vec<u8>, bool)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let (sent, requests) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let thread = thread::spawn(move || {
            for (response, hold) in responses {
                let deadline = Instant::now() + Duration::from_secs(3);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            if matches!(
                                gate.try_recv(),
                                Ok(()) | Err(mpsc::TryRecvError::Disconnected)
                            ) || Instant::now() >= deadline
                            {
                                return;
                            }
                            thread::park_timeout(Duration::from_millis(2));
                        }
                        Err(_) => return,
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let request = read_request(&mut stream);
                let _ = sent.send(request);
                let _ = stream.write_all(&response);
                if hold
                    && matches!(
                        gate.recv_timeout(Duration::from_secs(2)),
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected)
                    )
                {
                    return;
                }
            }
        });
        Self {
            endpoint,
            requests,
            release: Some(release),
            thread: Some(thread),
        }
    }
    pub(super) fn client(&self, limits: TransportLimits) -> HfJobsClient {
        // Private fixture-only HTTP endpoint; production constructor never admits it.
        HfJobsClient {
            http: transport::client().unwrap(),
            endpoint: self.endpoint.clone(),
            token: CREDENTIAL.into(),
            limits: limits.validate().unwrap(),
        }
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}
fn read_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0_u8; 4096];
        let Ok(n) = stream.read(&mut chunk) else {
            break;
        };
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..n]);
        assert!(bytes.len() <= 65536);
        if let Some(end) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
            let headers = std::str::from_utf8(&bytes[..end]).unwrap();
            let length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(str::trim)
                        .map(str::parse::<usize>)
                        .map(Result::unwrap)
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                break;
            }
        }
    }
    bytes
}
pub(super) fn response(status: u16, body: &[u8]) -> Vec<u8> {
    let mut bytes = format!(
        "HTTP/1.1 {status} fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    bytes.extend_from_slice(body);
    bytes
}
#[test]
fn jobs_trusted_origin_and_credential_grammar_refuse_without_echo() {
    for origin in [
        "http://huggingface.co",
        "https://evil.invalid",
        "https://huggingface.co.evil.invalid",
        "https://user:secret@huggingface.co",
        "https://huggingface.co/api",
        "https://huggingface.co/?token=secret",
        "https://huggingface.co/#secret",
    ] {
        let error =
            HfJobsClient::new_admitted(origin, CREDENTIAL.into(), TransportLimits::default())
                .err()
                .unwrap()
                .to_string();
        assert!(!error.contains("secret") && !error.contains(CREDENTIAL));
    }
    assert_eq!(
        transport::origin("https://huggingface.co/").unwrap(),
        "https://huggingface.co"
    );
    for token in ["", "line\nsecret", "line\rsecret", "nul\0secret"] {
        assert!(transport::token(token).is_err());
    }
    for part in ["", "..", "a/b", "a?secret", "a\nsecret"] {
        assert!(transport::url("https://huggingface.co", &[part]).is_err());
    }
    assert!(transport::deadline(Instant::now(), Duration::from_secs(1)).is_err());
}
#[test]
fn jobs_debug_is_redacted_but_bounded_explicit_submission_preserves_spec() {
    let spec = spec();
    assert!(!format!("{spec:?}").contains(CREDENTIAL));
    assert!(transport::encode(&spec, 32).is_err());
    let peer = Peer::new(
        response(200, br#"{"id":"job-1","status":{"stage":"PENDING"}}"#),
        false,
    );
    let client = peer.client(TransportLimits::default());
    let info = runtime().block_on(client.submit("owner", &spec)).unwrap();
    assert_eq!(info.id, "job-1");
    let request = peer.requests.recv_timeout(Duration::from_secs(2)).unwrap();
    let text = String::from_utf8(request).unwrap();
    assert!(text.starts_with("POST /api/jobs/owner HTTP/1.1"));
    assert!(
        text.to_ascii_lowercase()
            .contains(&format!("authorization: bearer {CREDENTIAL}"))
    );
    let body: serde_json::Value =
        serde_json::from_str(text.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["secrets"]["HF_TOKEN"], CREDENTIAL);
    assert_eq!(body["volumes"][0]["readOnly"], true);
    assert_eq!(body["volumes"][0]["revision"], "immutable-revision");
}
#[test]
fn jobs_error_json_and_redirect_refuse_without_server_or_transport_diagnostics() {
    for status in [302, 401, 500] {
        let peer = Peer::new(response(status, CREDENTIAL.as_bytes()), false);
        let error = runtime()
            .block_on(peer.client(TransportLimits::default()).list("owner"))
            .unwrap_err()
            .to_string();
        assert!(error.contains(&status.to_string()));
        assert!(!error.contains(CREDENTIAL) && !error.contains(&peer.endpoint));
    }
    let peer = Peer::new(response(200, CREDENTIAL.as_bytes()), false);
    let error = runtime()
        .block_on(peer.client(TransportLimits::default()).list("owner"))
        .unwrap_err()
        .to_string();
    assert_eq!(error, "HF Jobs JSON response invalid");
    let peer = Peer::new(response(200, &[b'x'; 129]), false);
    let limits = TransportLimits {
        json_bytes: 128,
        ..TransportLimits::default()
    };
    assert!(
        runtime()
            .block_on(peer.client(limits).list("owner"))
            .unwrap_err()
            .to_string()
            .contains("byte bound")
    );
}
#[test]
fn jobs_absolute_response_deadline_does_not_reset_after_headers() {
    let peer = Peer::new(
        b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\nConnection: close\r\n\r\n[".to_vec(),
        true,
    );
    let client = peer.client(TransportLimits::default());
    let error = runtime()
        .block_on(client.list_until("owner", Instant::now() + Duration::from_millis(200)))
        .unwrap_err();
    assert!(error.to_string().contains("deadline"));
    peer.requests.recv_timeout(Duration::from_secs(2)).unwrap();
}
#[test]
fn jobs_log_stream_preserves_utf8_plain_data_and_refuses_incomplete_tail() {
    let body = "data: {\"data\":\"café\",\"timestamp\":null}\n\ndata: plain\n";
    let peer = Peer::new(response(200, body.as_bytes()), false);
    runtime().block_on(async {
        let client = peer.client(TransportLimits::default());
        let stream = client.stream_logs("owner", "job-1").await.unwrap();
        let rows: Vec<_> = stream.collect().await;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].as_ref().unwrap(), "café");
        assert_eq!(rows[1].as_ref().unwrap(), "plain");
    });
    let peer = Peer::new(response(200, b"data: incomplete"), false);
    runtime().block_on(async {
        let stream = peer
            .client(TransportLimits::default())
            .stream_logs("owner", "job-1")
            .await
            .unwrap();
        let rows: Vec<_> = stream.collect().await;
        assert_eq!(rows.len(), 1);
        assert!(rows[0].is_err());
    });
}
#[test]
fn jobs_logs_bound_lines_total_bytes_and_absolute_pending_deadline() {
    for (body, cap, line) in [
        (b"data: too-long\n".as_slice(), 64, 8),
        (b"data: one\ndata: two\n".as_slice(), 12, 12),
    ] {
        let peer = Peer::new(response(200, body), false);
        let limits = TransportLimits {
            log_bytes: cap,
            log_line_bytes: line,
            ..TransportLimits::default()
        };
        runtime().block_on(async {
            let rows: Vec<_> = peer
                .client(limits)
                .stream_logs("owner", "job-1")
                .await
                .unwrap()
                .collect()
                .await;
            assert!(rows.iter().any(Result::is_err));
        });
    }
    let peer = Peer::new(
        b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\nConnection: close\r\n\r\ndata: first\n"
            .to_vec(),
        true,
    );
    runtime().block_on(async {
        let client = peer.client(TransportLimits::default());
        let stream = client
            .stream_logs_until(
                "owner",
                "job-1",
                Instant::now() + Duration::from_millis(300),
            )
            .await
            .unwrap();
        let mut stream = std::pin::pin!(stream);
        assert_eq!(stream.next().await.unwrap().unwrap(), "first");
        assert!(
            stream
                .next()
                .await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("deadline")
        );
        assert!(stream.next().await.is_none());
    });
}
#[test]
fn jobs_redirect_never_contacts_the_supplied_second_origin() {
    let target = Peer::new(response(200, b"[]"), false);
    let bytes = format!("HTTP/1.1 302 Found\r\nLocation: {}/api/jobs/owner\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", target.endpoint).into_bytes();
    let redirect = Peer::new(bytes, false);
    let error = runtime()
        .block_on(redirect.client(TransportLimits::default()).list("owner"))
        .unwrap_err();
    assert!(error.to_string().contains("302"));
    assert!(matches!(
        target.requests.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
}
#[test]
fn jobs_expired_deadline_and_dropped_pending_log_future_leave_no_owned_task() {
    let peer = Peer::new(
        b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\nConnection: close\r\n\r\ndata: observed\n"
            .to_vec(),
        true,
    );
    let client = peer.client(TransportLimits::default());
    runtime().block_on(async {
        assert!(client.list_until("owner", Instant::now()).await.is_err());
        let stream = client.stream_logs("owner", "job-1").await.unwrap();
        let mut stream = Box::pin(stream);
        assert_eq!(stream.next().await.unwrap().unwrap(), "observed");
        // Cancellation owner drops this live stream; no detached polling task exists.
        drop(stream);
    });
    peer.requests.recv_timeout(Duration::from_secs(2)).unwrap();
    drop(client);
    drop(peer); // Releases and deterministically joins the bounded fixture even after cancellation.
}
