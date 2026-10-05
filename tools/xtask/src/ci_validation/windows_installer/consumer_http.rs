//! Owned, bounded loopback archive peer for the complete installer entrypoint.
use sha2::{Digest, Sha256};
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
pub(super) const ASSET: &str = "mesh-llm-x86_64-pc-windows-msvc.zip";
pub(super) struct Server {
    pub base: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<io::Result<Vec<String>>>>,
}
impl Server {
    pub fn start(archive: Vec<u8>) -> Self {
        assert!(!archive.is_empty() && archive.len() <= 64 * 1024 * 1024);
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let stop = Arc::new(AtomicBool::new(false));
        let cancel = stop.clone();
        let thread = thread::spawn(move || serve(listener, &archive, &cancel));
        Self {
            base,
            stop,
            thread: Some(thread),
        }
    }
    pub fn finish(mut self) -> Vec<String> {
        self.stop.store(true, Ordering::SeqCst);
        self.thread
            .take()
            .unwrap()
            .join()
            .expect("installer HTTP thread panicked")
            .expect("installer HTTP fixture failed")
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
fn active(stop: &AtomicBool, deadline: Instant) -> bool {
    !stop.load(Ordering::SeqCst) && Instant::now() < deadline
}
fn serve(listener: TcpListener, archive: &[u8], stop: &AtomicBool) -> io::Result<Vec<String>> {
    let deadline = Instant::now() + Duration::from_secs(70);
    let sidecar = format!("{}  {ASSET}\n", hex::encode(Sha256::digest(archive)));
    let asset = format!("/{ASSET}");
    let checksum = format!("/{ASSET}.sha256");
    let mut requests = Vec::new();
    while active(stop, deadline) {
        match listener.accept() {
            Ok((mut socket, peer)) => {
                if !peer.ip().is_loopback() || requests.len() >= 4 {
                    return Err(io::Error::other(
                        "unexpected installer HTTP peer/request count",
                    ));
                }
                socket.set_nonblocking(true)?;
                let path = request(&mut socket, stop, deadline)?;
                let bytes = if path == asset {
                    archive
                } else if path == checksum {
                    sidecar.as_bytes()
                } else {
                    return Err(io::Error::other("fixture refuses unrelated URL"));
                };
                requests.push(path);
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bytes.len()
                );
                let write_deadline = (Instant::now() + Duration::from_secs(10)).min(deadline);
                write(&mut socket, header.as_bytes(), stop, write_deadline)?;
                write(&mut socket, bytes, stop, write_deadline)?;
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(e) => return Err(e),
        }
    }
    if !stop.load(Ordering::SeqCst) {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "installer HTTP absolute deadline",
        ));
    }
    Ok(requests)
}
fn request(socket: &mut TcpStream, stop: &AtomicBool, deadline: Instant) -> io::Result<String> {
    let deadline = (Instant::now() + Duration::from_secs(2)).min(deadline);
    let mut bytes = Vec::new();
    let mut buffer = [0; 1024];
    while active(stop, deadline) {
        if bytes.windows(4).any(|p| p == b"\r\n\r\n") {
            let header = std::str::from_utf8(&bytes).map_err(io::Error::other)?;
            let words = header
                .lines()
                .next()
                .unwrap_or_default()
                .split_whitespace()
                .collect::<Vec<_>>();
            if words.len() != 3 || words[0] != "GET" {
                return Err(io::Error::other("installer fixture permits GET only"));
            }
            return Ok(words[1].to_owned());
        }
        match socket.read(&mut buffer) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "installer request EOF",
                ));
            }
            Ok(n) => bytes.extend_from_slice(&buffer[..n]),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(e) => return Err(e),
        }
        if bytes.len() > 8192 {
            return Err(io::Error::other("installer request header too large"));
        }
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        "installer request interrupted/deadline",
    ))
}
fn write(
    socket: &mut TcpStream,
    mut bytes: &[u8],
    stop: &AtomicBool,
    deadline: Instant,
) -> io::Result<()> {
    while !bytes.is_empty() && active(stop, deadline) {
        match socket.write(&bytes[..bytes.len().min(65536)]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "installer response EOF",
                ));
            }
            Ok(n) => bytes = &bytes[n..],
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(e) => return Err(e),
        }
    }
    if bytes.is_empty() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "installer response interrupted/deadline",
        ))
    }
}
