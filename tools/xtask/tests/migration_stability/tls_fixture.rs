use serde_json::Value;
use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub struct Reply {
    status: u16,
    body: Vec<u8>,
    chunked: bool,
    hold: bool,
    declared: Option<usize>,
}
impl Reply {
    pub fn json(status: u16, value: &Value) -> Self {
        Self::bytes(status, value.to_string().into_bytes())
    }
    pub fn bytes(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            body,
            chunked: false,
            hold: false,
            declared: None,
        }
    }
    pub fn stream(body: String, hold: bool) -> Self {
        Self {
            status: 200,
            body: body.into_bytes(),
            chunked: true,
            hold,
            declared: None,
        }
    }
    pub fn incomplete() -> Self {
        Self {
            status: 200,
            body: vec![b'{'],
            chunked: false,
            hold: true,
            declared: Some(99),
        }
    }
}

pub struct Server {
    pub base: String,
    pub ca: PathBuf,
    pub requests: Arc<Mutex<Vec<(String, Value)>>>,
    pub held_connections_closed: Arc<AtomicBool>,
    pub response_started: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    _directory: tempfile::TempDir,
}

impl Server {
    pub fn new(replies: Vec<Reply>) -> Self {
        let rcgen::CertifiedKey { cert, signing_key } =
            rcgen::generate_simple_self_signed(vec!["localhost".into(), "127.0.0.1".into()])
                .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let ca = directory.path().join("fixture-ca.pem");
        fs::write(&ca, cert.pem()).unwrap();
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![cert.der().clone()],
            rustls::pki_types::PrivatePkcs8KeyDer::from(signing_key.serialize_der()).into(),
        )
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("https://{}/tenant/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let output = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let expected_holds = replies.iter().filter(|reply| reply.hold).count();
        let held_connections_closed = Arc::new(AtomicBool::new(expected_holds == 0));
        let closed = held_connections_closed.clone();
        let response_started = Arc::new(AtomicBool::new(false));
        let started = response_started.clone();
        let worker = thread::spawn(move || {
            let config = Arc::new(config);
            let deadline = Instant::now() + Duration::from_secs(8);
            let mut closed_holds = 0usize;
            for reply in replies {
                let socket = loop {
                    if stopping.load(Ordering::SeqCst) || Instant::now() >= deadline {
                        return;
                    }
                    match listener.accept() {
                        Ok((socket, _)) => break socket,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(5))
                        }
                        Err(error) => panic!("TLS fixture accept failed: {error}"),
                    }
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let connection = rustls::ServerConnection::new(config.clone()).unwrap();
                let mut stream = rustls::StreamOwned::new(connection, socket);
                let Ok(request) = request(&mut stream) else {
                    return;
                };
                output.lock().unwrap().push(request);
                if send(&mut stream, &reply).is_err() {
                    return;
                }
                started.store(true, Ordering::SeqCst);
                if reply.hold {
                    let mut byte = [0u8; 1];
                    let close = match stream.read(&mut byte) {
                        Ok(0) => true,
                        Err(error) => matches!(
                            error.kind(),
                            std::io::ErrorKind::ConnectionReset
                                | std::io::ErrorKind::UnexpectedEof
                                | std::io::ErrorKind::BrokenPipe
                        ),
                        Ok(_) => false,
                    };
                    if !close {
                        return;
                    }
                    closed_holds += 1;
                    closed.store(closed_holds == expected_holds, Ordering::SeqCst);
                }
            }
        });
        Self {
            base,
            ca,
            requests,
            held_connections_closed,
            response_started,
            stop,
            worker: Some(worker),
            _directory: directory,
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let joined = self.worker.take().unwrap().join();
        if !thread::panicking() {
            assert!(joined.is_ok(), "TLS fixture worker failed");
        }
    }
}

fn request(
    stream: &mut rustls::StreamOwned<rustls::ServerConnection, TcpStream>,
) -> std::io::Result<(String, Value)> {
    let mut bytes = Vec::new();
    let mut chunk = [0; 8192];
    let boundary = loop {
        let count = stream.read(&mut chunk)?;
        if count == 0 || bytes.len() > 2 * 1024 * 1024 + 8192 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let head = String::from_utf8(bytes[..boundary].to_vec()).unwrap();
    let length = head
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    assert!(length <= 2 * 1024 * 1024);
    while bytes.len() < boundary + length {
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    Ok((
        head,
        if length == 0 {
            Value::Null
        } else {
            serde_json::from_slice(&bytes[boundary..boundary + length]).unwrap()
        },
    ))
}

fn send(
    stream: &mut rustls::StreamOwned<rustls::ServerConnection, TcpStream>,
    reply: &Reply,
) -> std::io::Result<()> {
    let framing = if reply.chunked {
        "Transfer-Encoding: chunked\r\n".into()
    } else {
        format!(
            "Content-Length: {}\r\n",
            reply.declared.unwrap_or(reply.body.len())
        )
    };
    write!(
        stream,
        "HTTP/1.1 {} fixture\r\n{framing}Content-Type: application/json\r\nConnection: close\r\n\r\n",
        reply.status
    )?;
    if reply.chunked {
        write!(stream, "{:x}\r\n", reply.body.len())?;
    }
    for bytes in reply.body.chunks(8192) {
        stream.write_all(bytes)?;
    }
    if reply.chunked {
        stream.write_all(b"\r\n")?;
        if !reply.hold {
            stream.write_all(b"0\r\n\r\n")?;
        }
    }
    stream.flush()
}
