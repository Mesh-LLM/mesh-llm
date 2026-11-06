//! Cross-package executable/collector test; the binary is an explicit input.
use super::execution;
use crate::process::{self, Cancellation, ProcessSpec, RawCaptureOptions, Value};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

struct Endpoint {
    address: String,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<usize>>,
}

impl Endpoint {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = format!("http://{}/v1", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut count = 0;
            while !stopped.load(Ordering::Acquire) && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream
                            .set_read_timeout(Some(Duration::from_secs(5)))
                            .unwrap();
                        stream
                            .set_write_timeout(Some(Duration::from_secs(5)))
                            .unwrap();
                        consume_request(&mut stream);
                        let content = if count < 2 { "round" } else { "sphere" };
                        let cached = if count == 0 { 0 } else { 20 };
                        let body = serde_json::json!({
                            "choices":[{"message":{"content":content}}],
                            "usage":{"prompt_tokens":28,"prompt_tokens_details":{"cached_tokens":cached}}
                        }).to_string();
                        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
                        count += 1;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                }
            }
            count
        });
        Self {
            address,
            stop,
            worker: Some(worker),
        }
    }

    fn finish(mut self) -> usize {
        self.stop.store(true, Ordering::Release);
        self.worker.take().unwrap().join().unwrap()
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn consume_request(stream: &mut TcpStream) {
    let mut bytes = Vec::new();
    let mut chunk = [0; 1024];
    let header_end = loop {
        let count = stream.read(&mut chunk).unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&chunk[..count]);
        assert!(bytes.len() < 32768);
        if let Some(index) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let length = std::str::from_utf8(&bytes[..header_end])
        .unwrap()
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap();
    assert!(header_end + length < 32768);
    while bytes.len() < header_end + length {
        let count = stream.read(&mut chunk).unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&chunk[..count]);
    }
}

#[test]
#[ignore = "requires a prebuilt skippy-correctness binary in SKIPPY_KV_CORRECTNESS_TEST_BINARY"]
fn actual_correctness_stdout_is_admitted_by_the_cache_collector() {
    let binary = std::env::var_os("SKIPPY_KV_CORRECTNESS_TEST_BINARY")
        .expect("explicit prebuilt correctness binary");
    let binary = PathBuf::from(binary).canonicalize().unwrap();
    let root = tempfile::tempdir().unwrap();
    let prompt = root.path().join("prompt.txt");
    std::fs::write(&prompt, "A finite cache prefix.").unwrap();
    let peer = Endpoint::start();
    let arguments = [
        "open-ai-cache-reuse".into(),
        "--base-url".into(),
        peer.address.clone().into(),
        "--model".into(),
        "fixture".into(),
        "--prompt-file".into(),
        prompt.into_os_string(),
        "--max-tokens".into(),
        "8".into(),
    ]
    .into_iter()
    .map(Value::Public)
    .collect();
    // The production observer sees raw lines before log sanitization. Capture
    // that same bounded stdout, rather than re-reading redacted diagnostics.
    let captured = process::supervise_raw(
        &ProcessSpec {
            executable: binary,
            arguments,
            cwd: root.path().into(),
            environment: execution::environment(&[(
                "MESH_LLM_DATA_DIR",
                root.path().join("data").into_os_string(),
            )]),
        },
        &execution::limits(Duration::from_secs(20)),
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(64 * 1024),
            stderr: None,
        },
    )
    .unwrap();
    assert!(captured.process.success());
    assert!(captured.process.cleanup.complete);
    assert!(!captured.process.cleanup.forced);
    let stdout = String::from_utf8(captured.stdout.unwrap().as_bytes().to_vec()).unwrap();
    assert_eq!(peer.finish(), 4);
    execution::server::verify_captured_client_stdout(&stdout).unwrap();
    assert!(execution::server::verify_captured_client_stdout(&(stdout.clone() + &stdout)).is_err());
    let mut receipt: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    receipt["growing_chat_repeat"]["content"] = serde_json::json!("changed");
    assert!(execution::server::verify_captured_client_stdout(&receipt.to_string()).is_err());
}
