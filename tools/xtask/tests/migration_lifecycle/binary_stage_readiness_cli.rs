//! Real loopback sockets and finite caller-owned processes, without HTTP or native serving.
#![cfg(unix)]
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    net::{Ipv4Addr, SocketAddr, TcpListener},
    process::{Child, Command},
    time::{Duration, Instant},
};
struct Server(Child);
impl Server {
    fn sleeping(seconds: &str) -> Self {
        Self(Command::new("/bin/sleep").arg(seconds).spawn().unwrap())
    }
    fn pid(&self) -> u32 {
        self.0.id()
    }
    fn alive(&mut self) -> bool {
        self.0.try_wait().unwrap().is_none()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn address() -> SocketAddr {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .unwrap()
        .local_addr()
        .unwrap()
}
fn invoke(
    address: SocketAddr,
    pid: u32,
    seconds: u64,
    cancellation: &Cancellation,
) -> process::ProcessReport {
    let arguments = [
        "automation".to_owned(),
        "binary-stage-readiness".into(),
        "--host".into(),
        address.ip().to_string(),
        "--port".into(),
        address.port().to_string(),
        "--server-pid".into(),
        pid.to_string(),
        "--timeout-secs".into(),
        seconds.to_string(),
    ];
    process::supervise(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: arguments
                .into_iter()
                .map(|a| Value::Public(a.into()))
                .collect(),
            cwd: std::env::current_dir().unwrap(),
            environment: BTreeMap::new(),
        },
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 16384,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        OutputFiles::default(),
    )
    .unwrap()
}
#[test]
fn delayed_tcp_listener_becomes_ready_without_http_and_server_remains_caller_owned() {
    let address = address();
    let mut server = Server::sleeping("10");
    let listener = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(250));
        let socket = TcpListener::bind(address).unwrap();
        socket.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(4);
        loop {
            match socket.accept() {
                Ok(_) => return,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => (),
                Err(error) => panic!("{error}"),
            }
            assert!(Instant::now() < deadline, "no actual readiness connection");
            std::thread::sleep(Duration::from_millis(10));
        }
    });
    let output = invoke(address, server.pid(), 3, &Cancellation::default());
    listener.join().unwrap();
    assert!(output.success(), "{output:?}");
    assert!(output.cleanup.complete);
    assert!(server.alive());
}
#[test]
fn closed_port_reaches_deadline_without_killing_live_server() {
    let mut server = Server::sleeping("10");
    let before = Instant::now();
    let output = invoke(address(), server.pid(), 1, &Cancellation::default());
    assert!(!output.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr.bytes_retained).contains("deadline exceeded"),
        "{output:?}"
    );
    assert!(before.elapsed() >= Duration::from_secs(1));
    assert!(before.elapsed() < Duration::from_secs(4));
    assert!(output.cleanup.complete);
    assert!(server.alive());
}
#[test]
fn caller_reaped_early_server_exit_fails_before_socket_deadline() {
    let mut server = Server::sleeping("0.2");
    let pid = server.pid();
    let before = Instant::now();
    let output = std::thread::scope(|scope| {
        scope.spawn(|| {
            server.0.wait().unwrap();
        });
        invoke(address(), pid, 3, &Cancellation::default())
    });
    assert!(!output.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr.bytes_retained).contains("server exited"),
        "{output:?}"
    );
    assert!(before.elapsed() < Duration::from_secs(2));
    assert!(output.cleanup.complete);
}
#[test]
fn cancellation_stops_wait_only_and_preserves_unrelated_caller_server() {
    let mut server = Server::sleeping("10");
    let cancellation = Cancellation::default();
    let trigger = cancellation.clone();
    let before = Instant::now();
    let output = std::thread::scope(|scope| {
        scope.spawn(move || {
            std::thread::sleep(Duration::from_millis(350));
            trigger.cancel();
        });
        invoke(address(), server.pid(), 3, &cancellation)
    });
    assert!(!output.success(), "{output:?}");
    assert!(output.cleanup.complete, "{output:?}");
    assert!(before.elapsed() < Duration::from_secs(3));
    assert!(server.alive());
}
