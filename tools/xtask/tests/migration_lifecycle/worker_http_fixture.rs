//! Bounded loopback-only HTTP peer; no model, runtime, or remote endpoint.
use std::{
    io::{self, Read, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Healthy,
    MeasurementTimeout,
    MeasurementError,
    WarmupError,
}

#[derive(Debug)]
pub(super) struct Request {
    pub method: String,
    pub path: String,
    pub body: serde_json::Value,
}

pub(super) struct Fixture {
    pub port: u16,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<io::Result<Vec<Request>>>>,
}

impl Fixture {
    pub fn start(mode: Mode) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let cancel = stop.clone();
        let thread = thread::spawn(move || serve(listener, mode, &cancel));
        Self {
            port,
            stop,
            thread: Some(thread),
        }
    }

    pub fn finish(mut self) -> Vec<Request> {
        self.stop.store(true, Ordering::SeqCst);
        self.thread
            .take()
            .unwrap()
            .join()
            .expect("fixture thread panicked")
            .expect("fixture HTTP failure")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            // Every socket operation is nonblocking with a 200 ms write deadline;
            // all loops check this flag and a single absolute fixture deadline.
            let _ = thread.join();
        }
    }
}

fn active(stop: &AtomicBool, deadline: Instant) -> bool {
    !stop.load(Ordering::SeqCst) && Instant::now() < deadline
}

fn serve(listener: TcpListener, mode: Mode, stop: &AtomicBool) -> io::Result<Vec<Request>> {
    let deadline = Instant::now() + Duration::from_secs(16);
    let mut requests = Vec::new();
    let mut posts = 0;
    while active(stop, deadline) {
        match listener.accept() {
            Ok((mut socket, address)) => {
                if requests.len() >= 8 {
                    return Err(io::Error::other("HTTP fixture exceeds eight requests"));
                }
                if !address.ip().is_loopback() {
                    return Err(io::Error::other("nonlocal fixture peer"));
                }
                let request = request(&mut socket, stop, deadline)?;
                if request.method == "POST" {
                    posts += 1;
                }
                let models = request.method == "GET" && request.path == "/v1/models";
                requests.push(request);
                if models {
                    response(
                        &mut socket,
                        200,
                        "application/json",
                        br#"{"data":[{"id":"served-fixture"}]}"#,
                    )?;
                } else if posts == 1 && matches!(mode, Mode::WarmupError)
                    || posts == 2 && matches!(mode, Mode::MeasurementError)
                {
                    response(&mut socket, 500, "application/json", b"{}")?;
                } else if posts == 2 && matches!(mode, Mode::MeasurementTimeout) {
                    while active(stop, deadline) {
                        thread::sleep(Duration::from_millis(2));
                    }
                } else {
                    streaming(&mut socket, if posts == 1 { 5 } else { 3 }, stop, deadline)?;
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(error) => return Err(error),
        }
    }
    if !stop.load(Ordering::SeqCst) {
        return Err(io::Error::new(io::ErrorKind::TimedOut, "fixture deadline"));
    }
    Ok(requests)
}

fn request(socket: &mut TcpStream, stop: &AtomicBool, deadline: Instant) -> io::Result<Request> {
    socket.set_nonblocking(true)?;
    let mut bytes = Vec::new();
    let mut buffer = [0; 4096];
    while active(stop, deadline) {
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            let header = std::str::from_utf8(&bytes[..end]).map_err(io::Error::other)?;
            let size = header
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>())
                })
                .transpose()
                .map_err(io::Error::other)?
                .unwrap_or(0);
            if size > 65536 || end > 16384 {
                return Err(io::Error::other("HTTP fixture input exceeds bound"));
            }
            if bytes.len() >= end + 4 + size {
                let mut words = header.lines().next().unwrap_or_default().split_whitespace();
                return Ok(Request {
                    method: words.next().unwrap_or_default().into(),
                    path: words.next().unwrap_or_default().into(),
                    body: if size == 0 {
                        serde_json::Value::Null
                    } else {
                        serde_json::from_slice(&bytes[end + 4..end + 4 + size])
                            .map_err(io::Error::other)?
                    },
                });
            }
        }
        match socket.read(&mut buffer) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "HTTP fixture request EOF",
                ));
            }
            Ok(count) => bytes.extend_from_slice(&buffer[..count]),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(error) => return Err(error),
        }
        if bytes.len() > 81920 {
            return Err(io::Error::other("HTTP fixture request exceeds bound"));
        }
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        "HTTP fixture request interrupted/deadline",
    ))
}

fn header(socket: &mut TcpStream, status: u16, content_type: &str, bytes: usize) -> io::Result<()> {
    bounded_write(socket, format!("HTTP/1.1 {status} Fixture\r\nContent-Type: {content_type}\r\nContent-Length: {bytes}\r\nConnection: close\r\n\r\n").as_bytes())
}

fn bounded_write(socket: &mut TcpStream, mut bytes: &[u8]) -> io::Result<()> {
    let deadline = Instant::now() + Duration::from_millis(200);
    while !bytes.is_empty() && Instant::now() < deadline {
        match socket.write(bytes) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "HTTP fixture response EOF",
                ));
            }
            Ok(count) => bytes = &bytes[count..],
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(error) => return Err(error),
        }
    }
    if bytes.is_empty() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "HTTP fixture response deadline",
        ))
    }
}

fn response(
    socket: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    header(socket, status, content_type, body.len())?;
    bounded_write(socket, body)
}

fn streaming(
    socket: &mut TcpStream,
    tokens: u64,
    stop: &AtomicBool,
    deadline: Instant,
) -> io::Result<()> {
    let first = b"data: {\"choices\":[{\"delta\":{\"content\":\"fixture\"}}]}\n\n";
    let final_part =
        format!("data: {{\"usage\":{{\"completion_tokens\":{tokens}}}}}\n\ndata: [DONE]\n\n");
    header(
        socket,
        200,
        "text/event-stream",
        first.len() + final_part.len(),
    )?;
    bounded_write(socket, first)?;
    let pause = Instant::now() + Duration::from_millis(20);
    while active(stop, deadline) && Instant::now() < pause {
        thread::sleep(Duration::from_millis(2));
    }
    bounded_write(socket, final_part.as_bytes())
}
