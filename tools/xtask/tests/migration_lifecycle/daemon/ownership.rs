use super::{
    cli::{case, command},
    protocol::{Behavior, Plan},
};
use crate::{pty::Terminal, support::Sentinel};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpListener};

fn theft(index: usize) {
    let case = case(&Plan {
        behavior: Behavior::HoldBind,
        ..Plan::default()
    });
    let mut sentinel = Sentinel::new(&case);
    let mut terminal = Terminal::start_command(&case, "daemon-readiness");
    terminal.wait_file("bind.armed");
    let port = case.audit().arguments[index].parse::<u16>().unwrap();
    let thief = TcpListener::bind((Ipv4Addr::LOCALHOST, port)).unwrap();
    std::fs::write(case.native.join("bind.release"), b"release").unwrap();
    let status = terminal.finish();
    assert!(!status.success());
    assert!(
        std::fs::read(case.native.join("cli.stdout"))
            .unwrap()
            .is_empty()
    );
    assert!(!case.native.join("models.count").exists());
    assert_eq!(thief.local_addr().unwrap().port(), port);
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
    terminal.disarm();
}

#[test]
fn d04_api_stolen() {
    theft(3);
}
#[test]
fn d04_console_stolen() {
    theft(5);
}

#[test]
fn d02_unrelated_models_listener_survives() {
    let case = case(&Plan {
        behavior: Behavior::ExternalModels,
        ..Plan::default()
    });
    let mut sentinel = Sentinel::new(&case);
    let stdout = std::fs::File::create(case.native.join("cli.stdout")).unwrap();
    let stderr = std::fs::File::create(case.native.join("cli.stderr")).unwrap();
    let mut cli = command(&case)
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
        .unwrap();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !case.native.join("bind.armed").exists() {
        assert!(std::time::Instant::now() < until);
        std::thread::yield_now();
    }
    let port = case.audit().arguments[3].parse::<u16>().unwrap();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port)).unwrap();
    listener.set_nonblocking(true).unwrap();
    std::fs::write(case.native.join("bind.release"), b"release").unwrap();
    let mut count = 0;
    let status = loop {
        if let Some(status) = cli.try_wait().unwrap() {
            break status;
        }
        if let Ok((mut stream, _)) = listener.accept() {
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(1)))
                .unwrap();
            let mut request = [0; 4096];
            assert!(stream.read(&mut request).unwrap() > 0);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                .unwrap();
            count += 1;
        }
        assert!(std::time::Instant::now() < until);
        std::thread::yield_now();
    };
    assert!(!status.success());
    assert_eq!(count, 1);
    let diagnostics = std::fs::read_to_string(case.native.join("cli.stderr")).unwrap();
    assert!(
        diagnostics.contains("attribution_unavailable"),
        "{diagnostics}"
    );
    assert_eq!(listener.local_addr().unwrap().port(), port);
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}

#[test]
fn d08_redirect_is_not_followed() {
    let elsewhere = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    elsewhere.set_nonblocking(true).unwrap();
    let wire = format!(
        "HTTP/1.1 302 Found\r\nLocation: http://{}/elsewhere\r\nContent-Length: 0\r\n\r\n",
        elsewhere.local_addr().unwrap()
    )
    .into_bytes();
    let case = case(&Plan {
        models_code: 302,
        wire: Some(wire),
        ..Plan::default()
    });
    let mut sentinel = Sentinel::new(&case);
    let output = command(&case).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        elsewhere.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}
