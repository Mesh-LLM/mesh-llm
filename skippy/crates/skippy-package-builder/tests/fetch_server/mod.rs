//! Finite owned local HTTP peer for package acquisition tests.
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};
pub(super) struct Server {
    pub endpoint: String,
    pub requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Server {
    pub fn start(root: &Path, mode: &str) -> Self {
        let mut files = BTreeMap::new();
        for name in [
            "model-package.json",
            "metadata.gguf",
            "embeddings.gguf",
            "output.gguf",
            "projector.gguf",
            "layers/0.gguf",
            "layers/1.gguf",
        ] {
            files.insert(name.to_owned(), fs::read(root.join(name)).unwrap());
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(vec![]));
        let stop = Arc::new(AtomicBool::new(false));
        let seen = requests.clone();
        let stopping = stop.clone();
        let mode = mode.to_owned();
        let thread = thread::spawn(move || {
            let mut infos = 0;
            while !stopping.load(Ordering::SeqCst) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::park_timeout(Duration::from_millis(5));
                    continue;
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                let mut request = Vec::new();
                let mut buffer = [0; 4096];
                while request.len() < 16384 && !request.windows(4).any(|bytes| bytes == b"\r\n\r\n")
                {
                    let Ok(count) = stream.read(&mut buffer) else {
                        break;
                    };
                    if count == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..count]);
                }
                let request = String::from_utf8_lossy(&request);
                let line = request.lines().next().unwrap_or("").to_owned();
                seen.lock().unwrap().push(line.clone());
                if mode == "hard-timeout" {
                    while !stopping.load(Ordering::SeqCst) {
                        thread::park_timeout(Duration::from_millis(5));
                    }
                    continue;
                }
                if mode == "metadata-stall" {
                    thread::park_timeout(Duration::from_millis(600));
                    continue;
                }
                reply(&mut stream, &line, &mode, &files, &mut infos);
            }
        });
        Self {
            endpoint,
            requests,
            stop,
            thread: Some(thread),
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.thread.take().unwrap().join().unwrap();
    }
}
fn reply(
    stream: &mut std::net::TcpStream,
    line: &str,
    mode: &str,
    files: &BTreeMap<String, Vec<u8>>,
    infos: &mut usize,
) {
    use sha2::{Digest as _, Sha256};
    if line.contains("/api/models/") {
        *infos += 1;
        let commit = if *infos > 1 {
            "b".repeat(40)
        } else {
            "a".repeat(40)
        };
        let body = serde_json::to_vec(&serde_json::json!({"id":"org/repo","sha":commit})).unwrap();
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(&body);
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("");
    let name = path
        .split("/resolve/")
        .nth(1)
        .and_then(|tail| tail.split_once('/'))
        .map(|(_, name)| name)
        .unwrap_or("");
    let Some(original) = files.get(name) else {
        let _ = stream
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    };
    if mode == "missing-projector" && name == "projector.gguf" {
        let _ = stream
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    }
    let commit = match mode {
        "wrong-commit" => "b".repeat(40),
        "malformed-commit" => "../outside".into(),
        _ => "a".repeat(40),
    };
    let mut body = original.clone();
    if name == "projector.gguf" {
        if mode == "bad-size" {
            body.push(0);
        }
        if mode == "bad-digest" {
            body[0] ^= 1;
        }
    }
    let etag: String = Sha256::digest(&body)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nX-Repo-Commit: {commit}\r\nETag: \"{etag}\"\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    if line.starts_with("GET ") {
        let _ = stream.write_all(&body);
    }
}
