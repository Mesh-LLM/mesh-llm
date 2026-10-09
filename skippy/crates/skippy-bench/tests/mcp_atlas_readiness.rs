//! Finite loopback proof for the local probe; upstream services are not launched.
#![cfg(unix)]
use std::{net::TcpListener, process::Command};

fn probe(port: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_skippy-bench"))
        .args(["eval", "port-ready", port])
        .output()
        .unwrap()
}

#[test]
fn native_probe_admits_ipv4_loopback_and_refuses_closed_or_invalid_ports() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    let ready = probe(&port);
    assert!(ready.status.success(), "{ready:?}");
    assert!(ready.stdout.is_empty());
    drop(listener);
    let closed = probe(&port);
    assert!(!closed.status.success(), "{closed:?}");
    assert!(String::from_utf8_lossy(&closed.stderr).contains("127.0.0.1:"));
    for invalid in ["0", "65536", "-1", "127.0.0.1:1984", "1984 extra"] {
        let result = probe(invalid);
        assert!(!result.status.success(), "{invalid}: {result:?}");
        assert!(result.stdout.is_empty());
    }
}

#[test]
fn actual_template_port_function_uses_the_native_probe_and_preserves_failure() {
    if !zsh_available() {
        return;
    }
    let template = include_str!("../src/evals/adapters/templates/mcp_atlas_run.sh");
    let function = template
        .split("port_ready() {{\n")
        .nth(1)
        .unwrap()
        .split("\n}}\n")
        .next()
        .unwrap();
    assert!(!function.contains("python"));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    let invoke = || {
        Command::new("zsh")
            .args([
                "-f",
                "-c",
                &format!("set -euo pipefail\nport_ready() {{\n{function}\n}}\nport_ready \"$1\""),
                "finite-readiness",
                &port,
            ])
            .env("READINESS_HELPER", env!("CARGO_BIN_EXE_skippy-bench"))
            .output()
            .unwrap()
    };
    let ready = invoke();
    assert!(ready.status.success(), "{ready:?}");
    assert!(ready.stdout.is_empty() && ready.stderr.is_empty());
    drop(listener);
    let closed = invoke();
    assert_eq!(closed.status.code(), Some(1), "{closed:?}");
    assert!(closed.stdout.is_empty() && closed.stderr.is_empty());
    assert!(template.contains("if ! port_ready 1984; then"));
    assert!(template.contains("if ! port_ready 3000; then"));
    assert!(template.contains("for _ in {{1..90}}; do"));
    assert!(template.contains("curl -fsS --max-time 5"));
}

fn zsh_available() -> bool {
    Command::new("zsh")
        .args(["-f", "-c", "true"])
        .output()
        .is_ok_and(|output| output.status.success())
}
