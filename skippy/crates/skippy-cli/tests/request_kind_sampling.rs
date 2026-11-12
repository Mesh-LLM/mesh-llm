//! A lane reused across request kinds samples each request with its own
//! sampler. Opt in with a matching locally built runtime bundle
//! (`SKIPPY_TEST_RUNTIME_BUNDLE`) and a small generation GGUF
//! (`SKIPPY_TEST_GENERATION_MODEL`); without both the test returns early.
#![cfg(feature = "dynamic-native-runtime")]

use std::{
    ffi::OsString,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

/// A greedy chat leaves its backend sampler on the lane after reset. A
/// sampled completion that reuses the lane must still produce tokens.
#[test]
fn sampled_completion_after_greedy_chat_on_the_same_lane_succeeds() {
    let (Some(bundle), Some(model)) = (
        non_empty_env("SKIPPY_TEST_RUNTIME_BUNDLE").map(PathBuf::from),
        non_empty_env("SKIPPY_TEST_GENERATION_MODEL"),
    ) else {
        return;
    };
    let home = tempfile::tempdir().unwrap();
    let address = unused_local_address();
    let mut command = Command::new(env!("CARGO_BIN_EXE_skippy"));
    command
        .env("HOME", home.path())
        .env("XDG_CACHE_HOME", home.path().join("cache"))
        .env("XDG_CONFIG_HOME", home.path().join("config"))
        .env("XDG_DATA_HOME", home.path().join("data"))
        // Detaching on reset would hide the kept sampler under test.
        .env_remove("SKIPPY_DETACH_SAMPLER_ON_RESET")
        .arg("--runtime-bundle")
        .arg(&bundle)
        .args(["serve", "--model-path"])
        .arg(&model)
        .args(["--model-id", "test-model", "--bind-addr"])
        .arg(address.to_string())
        // One lane, so every request after the chat reuses its lane.
        .args([
            "--generation-concurrency",
            "1",
            "--ctx-size",
            "1024",
            "--n-gpu-layers",
            "0",
            "--prefix-cache",
            "off",
            "--native-mtp=false",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut server = ChildGuard(command.spawn().unwrap());
    wait_until_ready(address, &mut server);

    let chat = r#"{"model":"test-model","messages":[{"role":"user","content":"Reply with a short greeting."}],"max_tokens":4,"temperature":0,"reasoning_budget":-1}"#;
    let completion =
        r#"{"model":"test-model","prompt":"Hello","max_tokens":4,"temperature":0.8,"seed":7}"#;
    for round in 0..2 {
        let (status, body) = post(address, "/v1/chat/completions", chat);
        assert_eq!(status, 200, "round {round} chat: {body}");
        for index in 0..2 {
            let (status, body) = post(address, "/v1/completions", completion);
            assert_eq!(status, 200, "round {round} completion {index}: {body}");
        }
    }
}

fn non_empty_env(name: &str) -> Option<OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

fn unused_local_address() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

fn wait_until_ready(address: SocketAddr, server: &mut ChildGuard) {
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        if let Some((200, _)) = request(address, "GET", "/v1/models", None) {
            return;
        }
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "server exited before readiness"
        );
        assert!(Instant::now() < deadline, "server did not become ready");
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn post(address: SocketAddr, path: &str, body: &str) -> (u16, String) {
    request(address, "POST", path, Some(body)).expect("request failed")
}

/// Sends one HTTP/1.1 request on a fresh connection and returns the status
/// code and the response body.
fn request(
    address: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(5)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(300)))
        .ok()?;
    let body = body.unwrap_or("");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).ok()?;
    let response = String::from_utf8_lossy(&response).into_owned();
    let (status_line, rest) = response.split_once("\r\n")?;
    let status = status_line.split_whitespace().nth(1)?.parse().ok()?;
    let body = rest.split_once("\r\n\r\n").map_or(rest, |(_, body)| body);
    Some((status, body.to_owned()))
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
