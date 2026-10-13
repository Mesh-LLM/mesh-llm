//! CLI fixture: every overlap body reaches the peer before any reply.
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
pub const PIN: &str = "KV-PIN-8842";
pub const PRIMARY: &str = "KV-STABILITY-PRIMARY-7429";
pub const SECONDARY: &str = "KV-STABILITY-SECONDARY-319";
#[derive(Clone, Copy)]
pub enum Behavior {
    Healthy,
    WrongPressure,
    WrongCache,
    FailedOverlap,
    HeldOverlap,
    WrongToolKey,
    MalformedTool,
    CacheShortfall,
    SuffixShortfall,
}
pub struct Server {
    pub base: String,
    pub requests: Arc<Mutex<Vec<Value>>>,
    pub cohorts: Arc<AtomicUsize>,
    pub closed: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl Server {
    pub fn new(count: usize, behavior: Behavior) -> Self {
        Self::with_log(count, behavior, None)
    }
    pub fn with_log(count: usize, behavior: Behavior, native_log: Option<PathBuf>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/tenant/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let cohorts = Arc::new(AtomicUsize::new(0));
        let observed = cohorts.clone();
        let closed = Arc::new(AtomicUsize::new(0));
        let disconnected = closed.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let handle = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(20);
            let mut held = Vec::<(TcpStream, Value, usize)>::new();
            let mut failure_sent = false;
            let mut appended = false;
            while !stopping.load(Ordering::SeqCst) && Instant::now() < deadline {
                let mut socket = match listener.accept() {
                    Ok((socket, _)) => socket,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("KV peer accept: {error}"),
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let Some(payload) = request(&mut socket) else {
                    continue;
                };
                if !appended {
                    if let Some(path) = &native_log {
                        let mut file = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(path)
                            .unwrap();
                        writeln!(file, "failed to find a memory slot after request").unwrap();
                    }
                    appended = true;
                }
                let index = {
                    let mut requests = captured.lock().unwrap();
                    requests.push(payload.clone());
                    requests.len()
                };
                if initial_overlap(&payload) {
                    held.push((socket, payload, index));
                    if held.len() == count {
                        observed.fetch_add(1, Ordering::SeqCst);
                        if matches!(behavior, Behavior::HeldOverlap) {
                            await_closed(&mut held, &stopping, &disconnected);
                            continue;
                        }
                        // The entire cohort has supplied complete bodies before the first reply.
                        for (mut socket, payload, index) in held.drain(..) {
                            let fail = matches!(behavior, Behavior::FailedOverlap)
                                && !failure_sent
                                && payload.get("tools").is_some();
                            failure_sent |= fail;
                            if fail {
                                send(&mut socket, 503, json!({"error":"overlap fixture failure"}));
                            } else {
                                send(&mut socket, 200, response(&payload, index, behavior));
                            }
                        }
                    }
                } else {
                    send(&mut socket, 200, response(&payload, index, behavior));
                }
            }
        });
        Self {
            base,
            requests,
            cohorts,
            closed,
            stop,
            thread: Some(handle),
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread.take() {
            handle.join().unwrap();
        }
    }
}
fn initial_overlap(payload: &Value) -> bool {
    let Some(messages) = payload["messages"].as_array() else {
        return false;
    };
    messages.len() == 2
        && messages[1]["content"]
            .as_str()
            .is_some_and(|text| text.starts_with("Concurrent "))
}
fn response(payload: &Value, index: usize, behavior: Behavior) -> Value {
    let prompt = payload["messages"].as_array().unwrap().last().unwrap()["content"]
        .as_str()
        .unwrap_or("");
    let message = if payload.get("tool_choice").is_some() {
        tool_message(prompt, index, behavior)
    } else {
        let wrong = (matches!(behavior, Behavior::WrongPressure)
            && prompt.starts_with("Pressure turn "))
            || (matches!(behavior, Behavior::WrongCache) && prompt.contains("measured tail beta"));
        json!({"role":"assistant","content":if wrong {"wrong visible answer".to_owned()} else {format!("{PIN} {PRIMARY} {SECONDARY}")},
            "reasoning_content":"private fixture reasoning"})
    };
    let cached = if matches!(behavior, Behavior::CacheShortfall) {
        2047
    } else {
        2048
    };
    let tokens = if matches!(behavior, Behavior::SuffixShortfall) {
        2305
    } else {
        2304
    };
    json!({"model":"resolved-kv-fixture","choices":[{"finish_reason":if payload.get("tool_choice").is_some(){"tool_calls"}else{"stop"},"message":message}],
        "usage":{"prompt_tokens":tokens,"prompt_tokens_details":{"cached_tokens":cached},"completion_tokens":10}})
}
fn send(socket: &mut TcpStream, status: u16, body: Value) {
    let body = body.to_string();
    let bytes = format!(
        "HTTP/1.1 {status} fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = socket.write_all(bytes.as_bytes());
}
fn request(socket: &mut TcpStream) -> Option<Value> {
    let mut bytes = Vec::new();
    let mut chunk = [0; 8192];
    loop {
        let count = socket.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
        assert!(bytes.len() <= 262144, "KV fixture request exceeds bound");
        let Some(end) = bytes.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
            continue;
        };
        let head = std::str::from_utf8(&bytes[..end]).unwrap();
        assert!(head.starts_with("POST /tenant/v1/chat/completions "));
        let length = head
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                key.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap();
        assert!(length <= 262144);
        if bytes.len() >= end + 4 + length {
            return Some(serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap());
        }
    }
}

fn await_closed(
    held: &mut Vec<(TcpStream, Value, usize)>,
    stop: &AtomicBool,
    closed: &AtomicUsize,
) {
    for (socket, _, _) in held.iter() {
        socket.set_nonblocking(true).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    while !held.is_empty() && !stop.load(Ordering::SeqCst) && Instant::now() < deadline {
        held.retain(|(socket, _, _)| {
            let mut byte = [0];
            match socket.peek(&mut byte) {
                Ok(0) => {
                    closed.fetch_add(1, Ordering::SeqCst);
                    false
                }
                Err(error) if error.kind() != std::io::ErrorKind::WouldBlock => {
                    closed.fetch_add(1, Ordering::SeqCst);
                    false
                }
                _ => true,
            }
        });
        thread::sleep(Duration::from_millis(2));
    }
    held.clear();
}

fn tool_message(prompt: &str, index: usize, behavior: Behavior) -> Value {
    let key = if prompt.contains("key=secondary")
        || (matches!(behavior, Behavior::WrongToolKey) && prompt.starts_with("Attempt "))
    {
        "secondary"
    } else {
        "primary"
    };
    let arguments = if matches!(behavior, Behavior::MalformedTool) && prompt.starts_with("Attempt ")
    {
        "{invalid".to_owned()
    } else {
        json!({"key":key}).to_string()
    };
    json!({"role":"assistant","content":null,"reasoning_content":"private fixture reasoning",
        "tool_calls":[{"id":format!("call-{index}"),"type":"function","private_marker":"omit",
            "function":{"name":"lookup_probe_fact","arguments":arguments}}]})
}
