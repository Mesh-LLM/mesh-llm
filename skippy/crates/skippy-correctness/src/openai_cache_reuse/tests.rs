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
        Self::with_content(cached, ["answer-0", "answer-0", "answer-2", "answer-2"])
    }

    fn with_content(cached: [u64; 4], content: [&str; 4]) -> Self {
        Self::with_telemetry(cached.map(|value| Some(json!(value))), content)
    }

    fn with_telemetry(cached: [Option<Value>; 4], content: [&str; 4]) -> Self {
        let content = content.map(str::to_owned);
        Self::answering(move |index, _| {
            (
                content.get(index).cloned().unwrap_or_default(),
                cached.get(index).cloned().flatten(),
            )
        })
    }

    /// A peer that answers request `index` with `respond(index, request)`:
    /// the content and the cached-token count it reports, if any.
    fn answering(
        respond: impl Fn(usize, &Value) -> (String, Option<Value>) + Send + 'static,
    ) -> Self {
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
                        let (content, cached) = respond(index, &request);
                        requests.push(request);
                        let mut response = json!({
                            "choices": [{"message": {"content": content}}],
                            "usage": {"prompt_tokens": 100,
                                "prompt_tokens_details": {}}
                        });
                        if let Some(value) = cached {
                            response["usage"]["prompt_tokens_details"]["cached_tokens"] = value;
                        }
                        let response = response.to_string();
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
        wave: 1,
    }
}

/// A peer that answers every request from its last message alone, as a
/// deterministic server does whichever way a request reaches it. Only the
/// first request is cold.
fn consistent_peer() -> Peer {
    Peer::answering(|index, request| {
        let last = request["messages"]
            .as_array()
            .and_then(|messages| messages.last())
            .and_then(|message| message["content"].as_str())
            .unwrap_or_default();
        (
            format!("re: {last}"),
            Some(json!(if index == 0 { 0 } else { 60 })),
        )
    })
}

fn last_user_message(request: &Value) -> &str {
    request["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_str())
        .unwrap()
}

#[test]
fn a_wave_repeats_the_grown_chat_then_answers_each_branch_together_and_alone() {
    let prompt = Prompt::new();
    let peer = consistent_peer();
    run(OpenAiCacheReuseArgs {
        wave: 3,
        ..args(&peer, &prompt)
    })
    .unwrap();
    let requests = peer.finish();
    // four sequential, three copies, three branches together, three alone
    assert_eq!(requests.len(), 4 + 3 + 3 + 3);
    assert_eq!(requests[0], requests[1]);
    assert_eq!(requests[2], requests[3]);
    for copy in &requests[4..7] {
        assert_eq!(copy, &requests[2], "a wave copy is the grown chat again");
    }
    let mut together: Vec<&str> = requests[7..10].iter().map(last_user_message).collect();
    together.sort_unstable();
    assert_eq!(
        together,
        [
            "Reply with the number 0 and nothing else.",
            "Reply with the number 1 and nothing else.",
            "Reply with the number 2 and nothing else."
        ]
    );
    for request in &requests[7..] {
        assert_eq!(request["messages"].as_array().unwrap().len(), 5);
    }
    let alone: Vec<&str> = requests[10..].iter().map(last_user_message).collect();
    assert_eq!(alone, {
        let mut sorted = alone.clone();
        sorted.sort_unstable();
        sorted
    });
}

#[test]
fn a_branch_that_answers_differently_alone_fails_by_name() {
    let prompt = Prompt::new();
    // the last request is branch 1 on its own
    let peer = Peer::answering(|index, request| {
        let last = last_user_message(request).to_owned();
        let content = if index == 4 + 2 + 2 + 1 {
            "something else".to_owned()
        } else {
            format!("re: {last}")
        };
        (content, Some(json!(if index == 0 { 0 } else { 60 })))
    });
    let error = run(OpenAiCacheReuseArgs {
        wave: 2,
        ..args(&peer, &prompt)
    })
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("wave branch 1 alone generated-content mismatch"),
        "{error}"
    );
}

#[test]
fn a_wave_copy_that_misses_the_cache_fails() {
    let prompt = Prompt::new();
    let peer = Peer::answering(|index, request| {
        let last = last_user_message(request).to_owned();
        let cached = if index == 0 || index >= 4 { 0 } else { 60 };
        (format!("re: {last}"), Some(json!(cached)))
    });
    let error = run(OpenAiCacheReuseArgs {
        wave: 2,
        ..args(&peer, &prompt)
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("wave copy"), "{error}");
    assert!(error.contains("did not reuse a prompt prefix"), "{error}");
}

fn assert_sequence(requests: &[Value]) {
    for request in requests {
        assert_eq!(request["model"], "fixture-model");
        assert_eq!(request["max_tokens"], 8);
        assert_eq!(request["reasoning_effort"], "none");
        assert_eq!(request["temperature"], 0);
        assert_eq!(request["seed"], 42);
        assert_eq!(
            request["messages"][0],
            json!({"role": "user", "content": "A finite cache prefix.\n"})
        );
        assert_eq!(request.as_object().unwrap().len(), 6);
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

#[test]
fn generated_content_mismatch_fails_before_any_later_request() {
    let prompt = Prompt::new();
    for (contents, requests_expected) in [
        (["seed", "changed", "grown", "grown"], 2),
        (["seed", "seed", "grown", "changed"], 4),
    ] {
        let peer = Peer::with_content([0, 50, 50, 50], contents);
        let error = run(args(&peer, &prompt)).unwrap_err().to_string();
        assert!(error.contains("generated-content mismatch"), "{error}");
        assert_eq!(peer.finish().len(), requests_expected);
    }
}

#[test]
fn cold_seed_must_be_uncached_and_nonempty() {
    let prompt = Prompt::new();
    for (cached, contents, expected) in [
        (
            [1, 50, 50, 50],
            ["seed", "seed", "grown", "grown"],
            "cold seed unexpectedly",
        ),
        (
            [0, 50, 50, 50],
            ["", "", "grown", "grown"],
            "cold seed returned empty",
        ),
    ] {
        let peer = Peer::with_content(cached, contents);
        assert!(
            run(args(&peer, &prompt))
                .unwrap_err()
                .to_string()
                .contains(expected)
        );
        assert_eq!(peer.finish().len(), 1);
    }
    let peer = Peer::with_content([0, 50, 50, 50], ["seed", "seed", "", ""]);
    assert!(
        run(args(&peer, &prompt))
            .unwrap_err()
            .to_string()
            .contains("growing chat returned empty content")
    );
    assert_eq!(peer.finish().len(), 3);
}

#[test]
fn unavailable_or_invalid_cache_telemetry_fails_at_each_request() {
    let prompt = Prompt::new();
    for index in 0..4 {
        for invalid in [
            None,
            Some(json!(null)),
            Some(json!("0")),
            Some(json!(-1)),
            Some(json!(1.5)),
        ] {
            let mut cached = [0, 50, 50, 50].map(|value| Some(json!(value)));
            cached[index] = invalid;
            let peer =
                Peer::with_telemetry(cached, ["answer-0", "answer-0", "answer-2", "answer-2"]);
            let error = run(args(&peer, &prompt)).unwrap_err().to_string();
            assert!(error.contains("unsigned cached token count"), "{error}");
            assert_eq!(peer.finish().len(), index + 1);
        }
    }
}
