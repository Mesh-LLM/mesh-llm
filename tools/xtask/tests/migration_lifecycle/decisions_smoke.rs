//! Actual registered Decisions CLI with a bounded native HTTP peer; no model/runtime proof.
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
struct Peer {
    base: String,
    calls: Arc<Mutex<Vec<(String, Value)>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
fn request(socket: &mut TcpStream) -> Option<(String, Value)> {
    socket.set_nonblocking(false).ok()?;
    socket
        .set_read_timeout(Some(Duration::from_millis(200)))
        .ok()?;
    socket
        .set_write_timeout(Some(Duration::from_millis(200)))
        .ok()?;
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    let end = loop {
        let count = socket.read(&mut chunk).ok()?;
        if count == 0 || bytes.len() + count > 65536 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let header = std::str::from_utf8(&bytes[..end]).ok()?;
    let path = header.lines().next()?.split_whitespace().nth(1)?.to_owned();
    let length = header
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    if end + length > 65536 {
        return None;
    }
    while bytes.len() < end + length {
        let count = socket.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    let body = if length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&bytes[end..end + length]).ok()?
    };
    Some((path, body))
}
fn models() -> Value {
    json!({"data":[{"id":"mesh","capabilities":["system_one"]},{"id":"auto","capabilities":["system_one"]},
        {"id":"ordinary","capabilities":[]},{"id":"actual","capabilities":["system_one"]},
        {"id":"second","capabilities":["system_one"]}]})
}
fn answers(model: &str) -> Value {
    json!({"model":model,"answers":[
        {"type":"predicate","name":"urgent","probability":0.6},
        {"type":"choice","name":"team","choice":"billing","confidence":0.8,
            "probabilities":[{"value":"billing","probability":0.8},{"value":"support","probability":0.2}]},
        {"type":"score","name":"frustration","score":1.25,"confidence":0.7,
            "probabilities":[{"value":0,"label":"0","probability":0.3},{"value":1,"label":"1","probability":0.7}]}],
        "usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}})
}
impl Peer {
    fn start(mode: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let observed = calls.clone();
        let stopped = stop.clone();
        let thread = thread::spawn(move || serve(listener, mode, &observed, &stopped));
        Self {
            base,
            calls,
            stop,
            thread: Some(thread),
        }
    }
}
fn serve(
    listener: TcpListener,
    mode: &str,
    calls: &Mutex<Vec<(String, Value)>>,
    stop: &AtomicBool,
) {
    let deadline = Instant::now() + Duration::from_secs(8);
    while !stop.load(Ordering::SeqCst) && Instant::now() < deadline {
        let Ok((mut socket, _)) = listener.accept() else {
            thread::sleep(Duration::from_millis(5));
            continue;
        };
        let Some((path, body)) = request(&mut socket) else {
            continue;
        };
        calls.lock().unwrap().push((path.clone(), body.clone()));
        if mode == "stall" {
            while !stop.load(Ordering::SeqCst) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            continue;
        }
        if mode == "cumulative" {
            thread::sleep(Duration::from_millis(700));
        }
        let (status, bytes) = reply(mode, &path, &body);
        let header = format!(
            "HTTP/1.1 {status} fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        );
        let _ = socket.write_all(header.as_bytes());
        let _ = socket.write_all(&bytes);
    }
}
fn reply(mode: &str, path: &str, input: &Value) -> (u16, Vec<u8>) {
    if mode == "status" {
        return (503, b"secret untrusted body".to_vec());
    }
    if mode == "redirect" {
        return (302, Vec::new());
    }
    if mode == "malformed" {
        return (200, b"not JSON".to_vec());
    }
    if mode == "oversize" {
        return (200, vec![b' '; 1024 * 1024 + 1]);
    }
    let mut body = if path == "/v1/models" {
        models()
    } else {
        answers(input["model"].as_str().unwrap_or("absent"))
    };
    if mode == "no-model" {
        body = json!({"data":[]});
    }
    if path == "/v1/decisions" {
        match mode {
            "boolean" => body["answers"][0]["probability"] = json!(true),
            "wrong-model" => body["model"] = json!("other"),
            "usage" => body["usage"]["total_tokens"] = json!(-1),
            "order" => body["answers"].as_array_mut().unwrap().swap(0, 1),
            _ => {}
        }
    }
    (200, serde_json::to_vec(&body).unwrap())
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.thread.take().unwrap().join().unwrap();
    }
}
fn run(peer: &Peer, extra: &[&str]) -> std::process::Output {
    run_with_base(&peer.base, extra)
}
fn run_with_base(base: &str, extra: &[&str]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "decisions-smoke",
            "--base-url",
            base,
            "--timeout",
            "1",
        ])
        .args(extra)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "Decisions CLI exceeded parent bound: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(5));
    }
}
#[test]
fn decisions_cli_discovers_real_capability_and_sends_all_three_questions() {
    for extra in [vec![], vec!["--model", "second"]] {
        let peer = Peer::start("valid");
        let output = run(&peer, &extra);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let calls = peer.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], ("/v1/models".into(), Value::Null));
        assert_eq!(calls[1].0, "/v1/decisions");
        let selected = if extra.is_empty() { "actual" } else { "second" };
        assert_eq!(calls[1].1["model"], selected);
        assert_eq!(
            calls[1].1["input"],
            "I was charged twice. Please refund me today."
        );
        let questions = calls[1].1["questions"].as_array().unwrap();
        assert_eq!(questions.len(), 3);
        assert_eq!(questions[0]["type"], "predicate");
        assert_eq!(questions[1]["choices"][1]["value"], "support");
        assert_eq!(questions[2]["levels"][1]["label"], "1");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains(&format!("passed: model={selected}"))
        );
    }
}
#[test]
fn decisions_cli_refuses_unadvertised_model_before_post() {
    for (mode, extra) in [
        ("no-model", vec![]),
        ("valid", vec!["--model", "ordinary"]),
        ("valid", vec!["--model", "mesh"]),
    ] {
        let peer = Peer::start(mode);
        let output = run(&peer, &extra);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert_eq!(peer.calls.lock().unwrap().len(), 1);
    }
}
#[test]
fn decisions_cli_refuses_corrupt_responses_status_redirect_and_body_overflow() {
    for mode in [
        "boolean",
        "wrong-model",
        "usage",
        "order",
        "status",
        "redirect",
        "malformed",
        "oversize",
    ] {
        let peer = Peer::start(mode);
        let output = run(&peer, &[]);
        assert!(!output.status.success(), "accepted {mode}");
        assert!(
            output.stdout.is_empty(),
            "{mode}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(!String::from_utf8_lossy(&output.stderr).contains("secret untrusted body"));
    }
}
#[test]
fn decisions_cli_global_deadline_refuses_stalled_peer() {
    let peer = Peer::start("stall");
    let started = Instant::now();
    let output = run(&peer, &[]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(started.elapsed() < Duration::from_secs(4));
}
#[test]
fn decisions_cli_deadline_is_shared_across_discovery_and_post() {
    let peer = Peer::start("cumulative");
    let output = run(&peer, &[]);
    assert!(
        !output.status.success(),
        "per-request deadlines incorrectly admitted a 1.4s exchange"
    );
    assert!(output.stdout.is_empty());
    assert_eq!(peer.calls.lock().unwrap().len(), 2);
}
#[test]
fn decisions_cli_refuses_sensitive_endpoint_before_network() {
    let peer = Peer::start("valid");
    let output = run_with_base(
        &peer.base.replacen("http://", "http://user:secret@", 1),
        &[],
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(peer.calls.lock().unwrap().is_empty());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("secret"));
}
#[test]
fn decisions_cli_interrupt_refuses_stalled_peer_without_success_output() {
    let peer = Peer::start("stall");
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "decisions-smoke",
            "--base-url",
            &peer.base,
            "--timeout",
            "10",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while peer.calls.lock().unwrap().is_empty() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("Decisions interrupt fixture did not reach the actual HTTP request");
        }
        thread::sleep(Duration::from_millis(5));
    }
    // The actual registered CLI installed its owned interrupt before issuing this request.
    assert_eq!(
        unsafe { libc::kill(i32::try_from(child.id()).unwrap(), libc::SIGINT) },
        0
    );
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("Decisions CLI failed to finish interrupt cleanup");
        }
        thread::sleep(Duration::from_millis(5));
    }
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}
