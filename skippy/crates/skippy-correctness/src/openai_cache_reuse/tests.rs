use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use serde_json::{Value, json};

use super::{OpenAiCacheReuseArgs, run};

static NEXT_PROMPT: AtomicU64 = AtomicU64::new(0);

struct Prompt(PathBuf);

impl Prompt {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "skippy-cache-api-{}-{}.txt",
            std::process::id(),
            NEXT_PROMPT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, "A finite cache prefix.\n").unwrap();
        Self(path)
    }
}

impl Drop for Prompt {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

struct Peer {
    base_url: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Vec<Value>>>,
}

impl Peer {
    fn new(cached: [u64; 4]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut requests = Vec::new();
            // Keep accepting after the expected sequence so unwanted extra calls are captured.
            while !stopped.load(Ordering::Acquire) && Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        stream
                            .set_write_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        let request = request_body(&mut stream);
                        let index = requests.len();
                        requests.push(request);
                        let response = json!({
                            "choices": [{"message": {"content": format!("answer-{index}")}}],
                            "usage": {"prompt_tokens": 100,
                                "prompt_tokens_details": {"cached_tokens": cached.get(index).copied().unwrap_or(0)}}
                        }).to_string();
                        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).unwrap();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                }
            }
            requests
        });
        Self {
            base_url,
            stop,
            thread: Some(thread),
        }
    }

    fn finish(mut self) -> Vec<Value> {
        self.stop.store(true, Ordering::Release);
        self.thread.take().unwrap().join().unwrap()
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn request_body(stream: &mut TcpStream) -> Value {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 1024];
    let header_end = loop {
        let count = stream
            .read(&mut chunk)
            .expect("bounded fixture header read");
        assert!(count > 0, "request ended before headers");
        bytes.extend_from_slice(&chunk[..count]);
        assert!(bytes.len() <= 32 * 1024, "fixture request cap");
        if let Some(position) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers = std::str::from_utf8(&bytes[..header_end]).unwrap();
    assert!(headers.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
    let length = headers
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .expect("fixed JSON request content length");
    assert!(header_end + length <= 32 * 1024);
    while bytes.len() < header_end + length {
        let count = stream.read(&mut chunk).expect("bounded fixture body read");
        assert!(count > 0, "request ended before body");
        bytes.extend_from_slice(&chunk[..count]);
        assert!(bytes.len() <= 32 * 1024);
    }
    serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap()
}

fn args(peer: &Peer, prompt: &Prompt) -> OpenAiCacheReuseArgs {
    OpenAiCacheReuseArgs {
        base_url: peer.base_url.clone(),
        model: "fixture-model".into(),
        prompt_file: prompt.0.clone(),
        max_tokens: 8,
        request_timeout_secs: 1,
    }
}

fn assert_sequence(requests: &[Value]) {
    for request in requests {
        assert_eq!(request["model"], "fixture-model");
        assert_eq!(request["max_tokens"], 8);
        assert_eq!(request["reasoning_effort"], "none");
        assert_eq!(
            request["messages"][0],
            json!({"role": "user", "content": "A finite cache prefix.\n"})
        );
        assert_eq!(request.as_object().unwrap().len(), 4);
    }
    if requests.len() > 1 {
        assert_eq!(
            requests[0], requests[1],
            "repeat must preserve the entire request"
        );
        assert_eq!(requests[0]["messages"].as_array().unwrap().len(), 1);
    }
    if requests.len() > 2 {
        assert_eq!(
            requests[2]["messages"],
            json!([
                {"role": "user", "content": "A finite cache prefix.\n"},
                {"role": "assistant", "content": "answer-0"},
                {"role": "user", "content": "Summarize your answer in one word."}
            ]),
            "growth must use seed content, not repeat content"
        );
    }
    if requests.len() > 3 {
        assert_eq!(
            requests[2], requests[3],
            "grown repeat must preserve the entire request"
        );
    }
}

#[test]
fn actual_cache_api_preserves_repeat_and_growing_four_request_sequence() {
    let prompt = Prompt::new();
    let peer = Peer::new([0, 100, 35, 100]);
    run(args(&peer, &prompt)).unwrap();
    let requests = peer.finish();
    assert_eq!(requests.len(), 4);
    assert_sequence(&requests);
}

#[test]
fn actual_cache_api_refuses_each_invalid_hit_before_any_later_request() {
    let prompt = Prompt::new();
    for (index, label) in [
        (1, "repeat"),
        (2, "growing chat"),
        (3, "growing chat repeat"),
    ] {
        for invalid in [0, 101] {
            let mut cached = [0, 50, 50, 50];
            cached[index] = invalid;
            let peer = Peer::new(cached);
            let error = run(args(&peer, &prompt)).unwrap_err().to_string();
            let requests = peer.finish();
            assert_eq!(requests.len(), index + 1, "{label}: {error}");
            assert!(
                error.contains(&format!("{label} did not reuse a prompt prefix")),
                "{error}"
            );
            assert!(
                error.contains(&format!("cached={invalid} prompt=100")),
                "{error}"
            );
            assert_sequence(&requests);
        }
    }
}

#[test]
fn actual_cache_api_rejects_zero_tokens_and_empty_prompt_before_http() {
    let prompt = Prompt::new();
    let peer = Peer::new([0, 50, 50, 50]);
    let mut input = args(&peer, &prompt);
    input.max_tokens = 0;
    assert!(
        run(input)
            .unwrap_err()
            .to_string()
            .contains("must be positive")
    );
    std::fs::write(&prompt.0, " \n\t").unwrap();
    assert!(
        run(args(&peer, &prompt))
            .unwrap_err()
            .to_string()
            .contains("prompt file is empty")
    );
    assert!(peer.finish().is_empty());
}
