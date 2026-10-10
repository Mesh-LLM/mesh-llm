//! Finite loopback HTTP observations used by the actual split snapshot adapter.
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub(super) struct Server {
    pub address: SocketAddr,
    stop: Arc<AtomicBool>,
    calls: Arc<Mutex<Vec<String>>>,
    worker: Option<JoinHandle<Vec<String>>>,
}
fn request(stream: &mut TcpStream) -> std::io::Result<String> {
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut bytes = Vec::new();
    let mut chunk = [0; 1024];
    let mut complete = false;
    while bytes.len() < 8192 {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(std::io::Error::other(
                "fixture request header deadline expired",
            ));
        }
        stream.set_read_timeout(Some(remaining))?;
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if bytes.windows(4).any(|part| part == b"\r\n\r\n") {
            complete = true;
            break;
        }
    }
    if !complete {
        return Err(std::io::Error::other(
            "fixture request header is incomplete",
        ));
    }
    String::from_utf8_lossy(&bytes)
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .map(str::to_owned)
        .ok_or_else(|| std::io::Error::other("fixture request lacks a path"))
}
fn respond(
    mut stream: TcpStream,
    payloads: &BTreeMap<String, Vec<u8>>,
    delay: Duration,
    stop: &AtomicBool,
    calls: &Mutex<Vec<String>>,
) -> Result<(), String> {
    let path = request(&mut stream).map_err(|error| error.to_string())?;
    calls.lock().unwrap().push(path.clone());
    let deadline = Instant::now() + delay;
    while Instant::now() < deadline {
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(10));
    }
    let body = payloads
        .get(&path)
        .ok_or_else(|| format!("unexpected snapshot endpoint: {path}"))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(1)))
        .map_err(|error| error.to_string())?;
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    // A delayed fixture may finish after curl has closed its deadline-bounded request.
    if stream.write_all(response.as_bytes()).is_ok() {
        let _ = stream.write_all(body);
    }
    Ok(())
}
impl Server {
    pub fn start(payloads: BTreeMap<String, Vec<u8>>, delay: Duration) -> Self {
        assert!(delay <= Duration::from_secs(5));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed = calls.clone();
        let cancelled = stop.clone();
        let payloads = Arc::new(payloads);
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(12);
            let mut requests = Vec::new();
            let mut errors = Vec::new();
            while !cancelled.load(Ordering::Acquire) && Instant::now() < deadline {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if requests.len() >= 64 {
                            errors.push("fixture request budget exceeded".into());
                            break;
                        }
                        let payloads = payloads.clone();
                        let stop = cancelled.clone();
                        let calls = observed.clone();
                        requests.push(thread::spawn(move || {
                            respond(stream, &payloads, delay, &stop, &calls)
                        }));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10))
                    }
                    Err(error) => {
                        errors.push(format!("fixture accept failed: {error}"));
                        break;
                    }
                }
            }
            cancelled.store(true, Ordering::Release);
            for request in requests {
                match request.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => errors.push(error),
                    Err(_) => errors.push("HTTP fixture request worker panicked".into()),
                }
            }
            errors
        });
        Self {
            address,
            stop,
            calls,
            worker: Some(worker),
        }
    }
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let result = worker.join();
            if !thread::panicking() {
                let errors = result.expect("HTTP fixture worker panicked");
                assert!(errors.is_empty(), "HTTP fixture errors: {errors:?}");
            }
        }
    }
}
