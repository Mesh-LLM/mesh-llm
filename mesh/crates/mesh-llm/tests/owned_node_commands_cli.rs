use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Output};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const SCAN_RESPONSE: &str = r#"{
  "target_node_id": "abcd",
  "disposition": "executed",
  "inventory": [
    {"canonical_model_ref": "z/model", "total_size_bytes": 100},
    {"canonical_model_ref": "a/model", "total_size_bytes": 42}
  ]
}"#;

#[test]
fn scan_refresh_is_visible_while_compatibility_spelling_stays_hidden() {
    let help = run_mesh_llm(["runtime", "--help"]);
    assert!(help.status.success());
    let stdout = stdout(&help);
    assert!(stdout.contains("scan-refresh"), "runtime help: {stdout}");
    assert!(
        !stdout.contains("refresh-inventory"),
        "hidden compatibility spelling leaked into help: {stdout}"
    );

    let compatibility_help = run_mesh_llm(["runtime", "refresh-inventory", "--help"]);
    assert!(
        compatibility_help.status.success(),
        "compatibility spelling should still parse: {}",
        stderr(&compatibility_help)
    );
}

#[test]
fn scan_refresh_requires_exactly_one_explicit_endpoint() {
    let missing = run_mesh_llm(["runtime", "scan-refresh"]);
    assert!(!missing.status.success());
    assert!(stderr(&missing).contains("--endpoint"));

    let repeated = run_mesh_llm([
        "runtime",
        "scan-refresh",
        "--endpoint",
        "first",
        "--endpoint",
        "second",
    ]);
    assert!(!repeated.status.success());
}

#[test]
fn scan_refresh_human_output_is_deterministic() {
    let (port, server) = serve_once(SCAN_RESPONSE);
    let output = run_mesh_llm([
        "runtime",
        "scan-refresh",
        "--endpoint",
        "endpoint-token",
        "--port",
        &port.to_string(),
    ]);
    assert!(
        output.status.success(),
        "command failed: {}",
        stderr(&output)
    );
    let request = server.join().expect("test server should not panic");
    assert_scan_request(&request, "/api/runtime/control/scan-refresh");
    assert_eq!(
        stdout(&output),
        concat!(
            "🔐 Owner-control scan refresh\n",
            "\n",
            "Disposition: executed\n",
            "Target: abcd\n",
            "Models: 2\n",
            "Total bytes: 142\n",
            "Model refs:\n",
            "  a/model\n",
            "  z/model\n",
        )
    );
}

#[test]
fn scan_refresh_json_output_preserves_the_api_object() {
    let (port, server) = serve_once(SCAN_RESPONSE);
    let output = run_mesh_llm([
        "runtime",
        "scan-refresh",
        "--endpoint",
        "endpoint-token",
        "--port",
        &port.to_string(),
        "--json",
    ]);
    assert!(
        output.status.success(),
        "command failed: {}",
        stderr(&output)
    );
    let request = server.join().expect("test server should not panic");
    assert_scan_request(&request, "/api/runtime/control/scan-refresh");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&stdout(&output)).unwrap(),
        serde_json::from_str::<serde_json::Value>(SCAN_RESPONSE).unwrap()
    );
}

fn run_mesh_llm<const N: usize>(args: [&str; N]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mesh-llm"))
        .args(args)
        .output()
        .expect("mesh-llm command should run")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout should be utf-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr should be utf-8")
}

fn serve_once(response_body: &'static str) -> (u16, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("test server should bind");
    let port = listener.local_addr().unwrap().port();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("request should connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let request = read_http_request(&mut stream);
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream.write_all(response.as_bytes()).unwrap();
        String::from_utf8(request).expect("request should be utf-8")
    });
    (port, server)
}

fn read_http_request(stream: &mut impl Read) -> Vec<u8> {
    let mut request = Vec::new();
    loop {
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).expect("request should be readable");
        assert!(read > 0, "connection closed before request completed");
        request.extend_from_slice(&chunk[..read]);
        if request_is_complete(&request) {
            return request;
        }
    }
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
        return false;
    };
    let body_start = header_end + 4;
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    request.len() >= body_start + content_length
}

fn assert_scan_request(request: &str, path: &str) {
    assert!(request.starts_with(&format!("POST {path} HTTP/1.1\r\n")));
    assert!(request.contains(r#"{"endpoint":"endpoint-token"}"#));
}
