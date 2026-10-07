//! Explicit local SDK qualification; never substitutes for ordinary native boundary tests.
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

// The production child owner accepts this exact outcome carrier through super.
// No launcher, cleanup or orchestration behavior is copied into the fixture.
struct CommandOutcome {
    exit_status: Option<i32>,
    success: bool,
    timed_out: bool,
}
#[path = "../../src/evals/process_cleanup.rs"]
mod process_cleanup;

// Retained SDK interface test only: no server, source transformation, oracle or policy helper.
const SDK_CLIENT: &str = r#"
import asyncio, json, sys
import aiohttp
import swerex.runtime.remote as remote
from swerex.runtime.abstract import ReadFileRequest

async def main():
    client = remote.RemoteRuntime(auth_token="fixture", host=sys.argv[1], timeout=0.5)
    result = {"module": remote.__file__}
    try:
        response = await asyncio.wait_for(client.read_file(ReadFileRequest(path="fixture")), 5)
        result.update(content=response.content)
    except (TimeoutError, aiohttp.ClientError) as error:
        result.update(exception=type(error).__module__ + "." + type(error).__name__,
                      message=str(error), status=getattr(error, "status", None))
    print("SDK_RESULT=" + json.dumps(result))

asyncio.run(main())
"#;

#[derive(Clone, Copy, Debug)]
enum Scenario {
    TransferredTimeout,
    TransferredClient,
    Transient,
    Permanent,
}
impl Scenario {
    fn response(self, count: usize) -> (u16, Value) {
        match self {
            Self::TransferredTimeout => (
                511,
                transfer("builtins.TimeoutError", "retained-runtime-timeout"),
            ),
            Self::TransferredClient => (
                511,
                transfer("aiohttp.ClientError", "retained-runtime-client-error"),
            ),
            Self::Transient if count == 1 => (503, Value::Null),
            Self::Transient => (200, json!({"content":"fixture-ok"})),
            Self::Permanent => (404, json!({"detail":"fixture-not-found"})),
        }
    }
    fn check(self, result: &Value, requests: &[String]) -> Result<()> {
        let expected = if matches!(self, Self::Transient) {
            2
        } else {
            1
        };
        if requests.len() != expected || requests.iter().any(String::is_empty) {
            bail!("{self:?}: expected {expected} nonempty request IDs, observed {requests:?}");
        }
        match self {
            Self::TransferredTimeout
                if result["exception"] == "builtins.TimeoutError"
                    && result["message"] == "retained-runtime-timeout" =>
            {
                Ok(())
            }
            Self::TransferredClient
                if result["exception"] == "aiohttp.client_exceptions.ClientError"
                    && result["message"] == "retained-runtime-client-error" =>
            {
                Ok(())
            }
            Self::Transient if result["content"] == "fixture-ok" && requests[0] == requests[1] => {
                Ok(())
            }
            Self::Permanent
                if result["exception"] == "aiohttp.client_exceptions.ClientResponseError"
                    && result["status"] == 404 =>
            {
                Ok(())
            }
            _ => bail!("{self:?}: unexpected actual SDK outcome {result}"),
        }
    }
}
fn transfer(class: &str, message: &str) -> Value {
    json!({"swerexception":{"class_path":class,"message":message,"traceback":"","extra_info":{}}})
}

struct Peer {
    url: String,
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<Result<Vec<String>>>>,
}
impl Peer {
    fn start(scenario: Scenario) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://{}", listener.local_addr()?);
        listener.set_nonblocking(true)?;
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let join = thread::spawn(move || serve(listener, worker_stop, scenario));
        Ok(Self {
            url,
            stop,
            join: Some(join),
        })
    }
    fn finish(&mut self) -> Result<Vec<String>> {
        self.stop.store(true, Ordering::SeqCst);
        self.join
            .take()
            .context("peer already joined")?
            .join()
            .map_err(|_| anyhow::anyhow!("native peer panicked"))?
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
fn serve(listener: TcpListener, stop: Arc<AtomicBool>, scenario: Scenario) -> Result<Vec<String>> {
    let deadline = Instant::now() + Duration::from_secs(12);
    let mut requests = Vec::new();
    while !stop.load(Ordering::SeqCst) && Instant::now() < deadline {
        match listener.accept() {
            Ok((stream, _)) => {
                let mut stream = blocking_connection(stream)?;
                if requests.len() >= 6 {
                    bail!("unexpected retry amplification");
                }
                requests.push(request(&mut stream)?);
                let (status, body) = scenario.response(requests.len());
                let body = if status == 503 {
                    "not-json".into()
                } else {
                    serde_json::to_string(&body)?
                };
                write!(
                    stream,
                    "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )?;
                stream.flush()?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5))
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(requests)
}
fn blocking_connection(stream: TcpStream) -> Result<TcpStream> {
    // macOS may inherit the listener's nonblocking mode. Socket deadlines
    // below require blocking accepted connections.
    stream.set_nonblocking(false)?;
    Ok(stream)
}

#[test]
fn accepted_connection_clears_nonblocking_mode() {
    use std::os::fd::AsRawFd;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (stream, _) = listener.accept().unwrap();
    // Force inheritance independently of the host OS accept implementation.
    stream.set_nonblocking(true).unwrap();
    let stream = blocking_connection(stream).unwrap();
    // SAFETY: F_GETFL queries a live owned socket descriptor, without mutation.
    let flags = unsafe { libc::fcntl(stream.as_raw_fd(), libc::F_GETFL) };
    assert!(flags >= 0);
    assert_eq!(flags & libc::O_NONBLOCK, 0);
}

fn request(stream: &mut TcpStream) -> Result<String> {
    let deadline = Instant::now() + Duration::from_secs(2);
    stream.set_read_timeout(Some(Duration::from_millis(750)))?;
    stream.set_write_timeout(Some(Duration::from_millis(750)))?;
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 1024];
    let header_end = loop {
        if Instant::now() >= deadline {
            bail!("native peer header deadline");
        }
        if bytes.len() > 16384 {
            bail!("native peer request headers exceed bound");
        }
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            bail!("incomplete HTTP request");
        }
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(end) = bytes.windows(4).position(|x| x == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let headers = std::str::from_utf8(&bytes[..header_end])?;
    if headers.lines().next() != Some("POST /read_file HTTP/1.1") {
        bail!("unexpected SDK endpoint");
    }
    let mut length = None;
    let mut request_id = None;
    for line in headers.lines().skip(1) {
        if let Some((key, value)) = line.split_once(':') {
            if key.eq_ignore_ascii_case("content-length")
                && length.replace(value.trim().parse::<usize>()?).is_some()
            {
                bail!("duplicate length");
            }
            if key.eq_ignore_ascii_case("x-request-id")
                && request_id.replace(value.trim().to_owned()).is_some()
            {
                bail!("duplicate request ID");
            }
            if key.eq_ignore_ascii_case("transfer-encoding") {
                bail!("unexpected chunked request");
            }
        }
    }
    let length = length.context("missing bounded request length")?;
    if length > 4096 {
        bail!("SDK request body exceeds bound");
    }
    while bytes.len() < header_end + length {
        if Instant::now() >= deadline {
            bail!("native peer body deadline");
        }
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            bail!("incomplete request body");
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    let body: Value = serde_json::from_slice(&bytes[header_end..header_end + length])?;
    if body["path"] != "fixture" {
        bail!("unexpected SDK request payload");
    }
    request_id.context("missing actual SDK request ID")
}

#[test]
fn native_peer_accepts_a_split_request_under_socket_deadlines() {
    let mut peer = Peer::start(Scenario::Permanent).unwrap();
    let mut stream = TcpStream::connect(peer.url.strip_prefix("http://").unwrap()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let body = br#"{"path":"fixture"}"#;
    write!(
        stream,
        "POST /read_file HTTP/1.1\r\nContent-Length: {}\r\nX-Request-ID: split-request\r\n",
        body.len()
    )
    .unwrap();
    stream.flush().unwrap();
    // The nonblocking listener accepts before the complete header arrives.
    // An inherited nonblocking stream would refuse the second read on macOS.
    thread::sleep(Duration::from_millis(50));
    stream.write_all(b"\r\n").unwrap();
    stream.write_all(body).unwrap();
    stream.flush().unwrap();
    let mut response = String::new();
    stream.take(4096).read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 404 Fixture\r\n"));
    assert!(response.contains("fixture-not-found"));
    assert_eq!(peer.finish().unwrap(), ["split-request"]);
}

fn hash(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in Sha256::digest(bytes).iter() {
        text.push(char::from(HEX[usize::from(*byte >> 4)]));
        text.push(char::from(HEX[usize::from(*byte & 15)]));
    }
    text
}
fn bounded_read(path: &Path, cap: u64) -> Result<Vec<u8>> {
    use std::os::unix::fs::OpenOptionsExt as _;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    if !file.metadata()?.is_file() {
        bail!("SDK qualification needs regular file: {}", path.display());
    }
    let mut bytes = Vec::new();
    file.take(cap.checked_add(1).context("SDK fixture cap overflow")?)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > cap {
        bail!("SDK qualification file exceeds bound: {}", path.display());
    }
    Ok(bytes)
}
fn admitted_environment() -> Result<PathBuf> {
    let path = PathBuf::from(
        std::env::var_os("SKIPPY_SWEREX_TEST_ENVIRONMENT")
            .context("explicit prepared Modal environment path is required")?,
    );
    if !path.is_absolute() {
        bail!("test environment must be absolute");
    }
    let path = path.canonicalize()?;
    let receipt: Value = serde_json::from_slice(&bounded_read(
        path.parent()
            .context("environment parent")?
            .join("receipt.json")
            .as_path(),
        8 * 1048576,
    )?)?;
    if receipt["patch_profile"] != "swerex-1.4.0-modal-modern-v3"
        || receipt["configuration"]["deployment"] != "modal"
    {
        bail!("not prepared ModalV3");
    }
    for (relative, expected) in [
        (
            "deployment/modal.py",
            "a827f2b4c90cc56aeffb152b40a24a941af75360735613ee7a8abb457daddc41",
        ),
        (
            "deployment/config.py",
            "a19643cd219c3fa15de001cb520cf087165c64b47c264688dfa4c26385b02e45",
        ),
        (
            "runtime/remote.py",
            "7c0dd7a42dfcc99b2dc51f961cb0fe2aee81c09caae08a256694b4e37b4da532",
        ),
    ] {
        if hash(&bounded_read(
            &path
                .join("lib/python3.11/site-packages/swerex")
                .join(relative),
            1048576,
        )?) != expected
        {
            bail!("actual SDK profile hash differs: {relative}");
        }
    }
    Ok(path)
}
fn client(environment: &Path, url: &str, capture: &Path) -> Result<Value> {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(capture)?;
    let mut command = Command::new(environment.join("bin/python"));
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .args(["-I", "-B", "-c", SDK_CLIENT, url])
        .stdin(Stdio::null())
        .stdout(file.try_clone()?)
        .stderr(file);
    process_cleanup::configure_child_group(&mut command);
    let mut child = command.spawn()?;
    let outcome = process_cleanup::wait_with_timeout_observed(
        &mut child,
        Some(Duration::from_secs(4)),
        || {
            if fs::metadata(capture)?.len() > 65536 {
                bail!("SDK capture exceeds bound");
            }
            Ok(())
        },
    )?;
    // Normal exit can burst past the polling cap; descriptor cap+1 remains mandatory.
    // This read happens only after the production owner finishes owned cleanup.
    let bytes = bounded_read(capture, 65536)?;
    if !outcome.success || outcome.timed_out || outcome.exit_status != Some(0) {
        bail!("SDK child failed: {}", String::from_utf8_lossy(&bytes));
    }
    let text = std::str::from_utf8(&bytes)?;
    let line = text
        .lines()
        .filter_map(|line| line.strip_prefix("SDK_RESULT="))
        .collect::<Vec<_>>();
    if line.len() != 1 {
        bail!("missing or duplicate actual SDK result: {text}");
    }
    let result: Value = serde_json::from_str(line[0])?;
    let module = PathBuf::from(
        result["module"]
            .as_str()
            .context("SDK loaded module path")?,
    )
    .canonicalize()?;
    if module
        != environment
            .join("lib/python3.11/site-packages/swerex/runtime/remote.py")
            .canonicalize()?
    {
        bail!("actual SDK imported outside prepared profile");
    }
    Ok(result)
}

#[test]
#[ignore = "explicit prepared SDK local qualification, no installs/services; normal native boundaries remain required"]
fn actual_prepared_modal_sdk_transferred_errors_do_not_retry_and_transient_status_does() {
    let environment = admitted_environment().unwrap();
    let fixture = super::Fixture::new();
    for (index, scenario) in [
        Scenario::TransferredTimeout,
        Scenario::TransferredClient,
        Scenario::Transient,
        Scenario::Permanent,
    ]
    .into_iter()
    .enumerate()
    {
        let mut peer = Peer::start(scenario).unwrap();
        let result = client(
            &environment,
            &peer.url,
            &fixture.root.join(format!("sdk-{index}.log")),
        );
        let requests = peer.finish().unwrap();
        let result = result.unwrap();
        scenario.check(&result, &requests).unwrap();
        println!(
            "actual SDK {scenario:?}: {} requests; {result}",
            requests.len()
        );
    }
}
