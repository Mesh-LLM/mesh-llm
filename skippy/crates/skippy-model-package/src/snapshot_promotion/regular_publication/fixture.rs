use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
pub(super) struct Reply {
    pub status: u16,
    pub body: Vec<u8>,
    pub hold: bool,
    pub changed_file: Option<std::path::PathBuf>,
}
pub(super) struct Server {
    pub endpoint: String,
    pub requests: Arc<Mutex<Vec<Vec<u8>>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Server {
    pub fn start(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let worker = thread::spawn(move || {
            for reply in replies {
                let deadline = Instant::now() + Duration::from_secs(10);
                let mut socket = loop {
                    if stopped.load(Ordering::Acquire) {
                        return;
                    }
                    match listener.accept() {
                        Ok((socket, _)) => break socket,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            if Instant::now() >= deadline {
                                return;
                            }
                            thread::sleep(Duration::from_millis(2));
                        }
                        Err(_) => return,
                    }
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut request = Vec::new();
                let mut chunk = [0; 4096];
                loop {
                    let read = match socket.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(count) => count,
                    };
                    request.extend_from_slice(&chunk[..read]);
                    if request.len() > 1024 * 1024 {
                        return;
                    }
                    if let Some(end) = request.windows(4).position(|b| b == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                        let count = headers
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length: "))
                            .map_or(0, |v| v.parse::<usize>().unwrap());
                        if request.len() >= end + 4 + count {
                            break;
                        }
                    }
                }
                captured.lock().unwrap().push(request);
                if let Some(path) = reply.changed_file {
                    std::fs::write(path, b"{\"v\":2}").unwrap();
                }
                if reply.hold {
                    while !stopped.load(Ordering::Acquire) && Instant::now() < deadline {
                        thread::sleep(Duration::from_millis(2));
                    }
                    return;
                }
                let header = format!(
                    "HTTP/1.1 {} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    reply.status,
                    reply.body.len()
                );
                if socket.write_all(header.as_bytes()).is_err()
                    || socket.write_all(&reply.body).is_err()
                {
                    return;
                }
            }
        });
        Self {
            endpoint,
            requests,
            stop,
            worker: Some(worker),
        }
    }
    pub fn finish(mut self) -> Vec<Vec<u8>> {
        self.stop.store(true, Ordering::Release);
        self.worker.take().unwrap().join().unwrap();
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
