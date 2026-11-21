use super::HardwareReport;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

const CHILD: &str = "usage_reporting::hardware_probe::tests::blocking_probe_child";

#[test]
fn blocking_probe_child() {
    let Ok(address) = std::env::var("MESH_TEST_HARDWARE_PROBE_ADDRESS") else {
        return;
    };
    let mut connection = TcpStream::connect(address).unwrap();
    connection
        .write_all(&std::process::id().to_le_bytes())
        .unwrap();
    let mut release = [0];
    connection.read_exact(&mut release).unwrap();
}

/// #3067: the same blocking task and child runner used by the startup survey
/// must finish before Tokio is dropped. The socket is both a readiness handshake
/// and a failure-path release, so a broken test leaves no hanging child behind.
#[test]
fn shutdown_reaps_inflight_probe_and_allows_runtime_teardown() {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let accept = std::thread::spawn(move || {
        let (mut connection, _) = listener.accept().unwrap();
        connection
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        let mut pid = [0; 4];
        connection.read_exact(&mut pid).unwrap();
        ready_tx
            .send((connection, u32::from_le_bytes(pid)))
            .unwrap();
    });
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let result: anyhow::Result<()> = runtime.block_on(async {
            let mut report = HardwareReport::start(move || {
                let result = skippy_hardware_profile::probe::output(
                    Command::new(std::env::current_exe().unwrap())
                        .args(["--exact", CHILD, "--nocapture"])
                        .env("MESH_TEST_HARDWARE_PROBE_ADDRESS", address.to_string()),
                );
                assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::Interrupted);
            });
            stop_rx.await?;
            report.shutdown().await
        });
        drop(runtime);
        done_tx.send(result).unwrap();
    });
    let (mut connection, pid) = ready_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    accept.join().unwrap();
    stop_tx.send(()).unwrap();
    let completed = done_rx.recv_timeout(Duration::from_secs(10));
    // Release the fixture on regression before asserting, so the test is bounded.
    if completed.is_err() {
        let _ = connection.write_all(&[1]);
    }
    completed
        .expect("shutdown and runtime teardown must finish with a live probe")
        .unwrap();
    worker.join().unwrap();
    assert_eq!(
        mesh_llm_system::process::process_liveness(pid),
        mesh_llm_system::process::Liveness::Dead
    );
    let mut byte = [0];
    assert_eq!(
        connection.read(&mut byte).unwrap(),
        0,
        "probe must close its connection"
    );
}
