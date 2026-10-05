//! HTTP collector contract fixture. Does not claim real OTLP ingestion.
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Good,
    Delayed,
    WrongRun,
    MissingDecode,
    Loss,
    FailFinalize,
    StallCollection,
}

struct Run {
    config: Value,
    finalized: bool,
    polls: usize,
}

pub(super) struct Collector {
    pub http: String,
    pub calls: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Collector {
    pub fn start(mode: Mode) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let http = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let stopped = stop.clone();
        let recorded = calls.clone();
        let thread = thread::spawn(move || {
            let mut runs = BTreeMap::<String, Run>::new();
            while !stopped.load(Ordering::SeqCst) {
                let mut socket = match listener.accept() {
                    Ok((socket, _)) => socket,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("collector fixture accept: {error}"),
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                let Some((method, path, body)) = request(&mut socket) else {
                    continue;
                };
                recorded.lock().unwrap().push(format!("{method} {path}"));
                if matches!(mode, Mode::StallCollection) && path.ends_with("/report.json") {
                    while !stopped.load(Ordering::SeqCst) {
                        thread::sleep(Duration::from_millis(5));
                    }
                    break;
                }
                let (status, body) = route(&mut runs, mode, &method, &path, body);
                let body = serde_json::to_vec(&body).unwrap();
                let _ = write!(
                    socket,
                    "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(&body);
            }
        });
        Self {
            http,
            calls,
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for Collector {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn request(socket: &mut TcpStream) -> Option<(String, String, Value)> {
    let mut bytes = Vec::new();
    loop {
        let mut buffer = [0_u8; 4096];
        let count = socket.read(&mut buffer).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > 1024 * 1024 {
            return None;
        }
        let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let header = std::str::from_utf8(&bytes[..end]).ok()?;
        let length = header
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if bytes.len() < end + 4 + length {
            continue;
        }
        let mut line = header.lines().next()?.split_whitespace();
        let method = line.next()?.to_owned();
        let path = line.next()?.to_owned();
        let body = if length == 0 {
            Value::Null
        } else {
            serde_json::from_slice(&bytes[end + 4..end + 4 + length]).ok()?
        };
        return Some((method, path, body));
    }
}

fn route(
    runs: &mut BTreeMap<String, Run>,
    mode: Mode,
    method: &str,
    path: &str,
    body: Value,
) -> (u16, Value) {
    if method == "POST" && path == "/v1/runs" {
        let Some(id) = body["run_id"].as_str().map(str::to_owned) else {
            return (400, json!({}));
        };
        if runs.contains_key(&id) {
            return (409, json!({"error":"duplicate fixture run"}));
        }
        runs.insert(
            id.clone(),
            Run {
                config: body,
                finalized: false,
                polls: 0,
            },
        );
        return (200, json!({"run_id":id,"status":"running"}));
    }
    let Some(rest) = path.strip_prefix("/v1/runs/") else {
        return (404, json!({}));
    };
    let Some((id, action)) = rest.split_once('/') else {
        return (404, json!({}));
    };
    let Some(run) = runs.get_mut(id) else {
        return (404, json!({}));
    };
    if method == "POST" && action == "finalize" {
        if matches!(mode, Mode::FailFinalize) {
            return (500, json!({"error":"fixture finalization failure"}));
        }
        run.finalized = true;
        return (200, json!({"run_id":id,"status":"completed"}));
    }
    if method == "GET" && action == "report.json" {
        run.polls += 1;
        return (200, report(id, run, mode));
    }
    (404, json!({}))
}

fn report(id: &str, run: &Run, mode: Mode) -> Value {
    let mut spans = Vec::new();
    let seed = run.config["seed_request_count"].as_u64().unwrap_or(0);
    let count = run.config["measurement_request_count"].as_u64().unwrap();
    if !matches!(mode, Mode::Delayed) || run.polls > 1 {
        for request in 1..=seed + count {
            for (name, start, end) in [
                ("stage.openai_decode_token", 2000000, 2100000),
                ("stage.openai_generation_summary", 1000000, 3000000),
            ] {
                if matches!(mode, Mode::MissingDecode) && name == "stage.openai_decode_token" {
                    continue;
                }
                spans.push(
                    json!({"run_id":id,"request_id":request.to_string(),"stage_id":"stage-0",
                    "trace_id":request.to_string(),"span_id":name,"name":name,
                    "start_time_unix_nanos":start,"end_time_unix_nanos":end}),
                );
            }
        }
    }
    json!({"run":{"run_id":if matches!(mode,Mode::WrongRun){"unrelated"}else{id},
        "status":if run.finalized{"completed"}else{"running"},
        "finished_at_unix_nanos":if run.finalized{json!(4000000)}else{Value::Null}},
        "counts":{"spans":spans.len()},"telemetry_loss":{"dropped_events":if matches!(mode,Mode::Loss){1}else{0},"export_errors":0},"spans":spans})
}
