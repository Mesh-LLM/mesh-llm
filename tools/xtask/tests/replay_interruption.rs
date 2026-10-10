#[cfg(unix)]
#[test]
fn interrupted_request_retains_partial_evidence_and_fails_command() {
    use std::{
        io::Read,
        net::TcpListener,
        process::{Command, Stdio},
    };
    let state = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let input = state.path().join("cell.json");
    let requests = state.path().join("requests.jsonl");
    let summary = state.path().join("summary.json");
    std::fs::write(&input, serde_json::to_vec(&serde_json::json!({
        "trajectories":[{"session_id":"interrupted","source_dataset":"fixture","agent_framework":"fixture",
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]}],
        "model":"fixture","base_url":format!("http://{address}/v1"),"concurrency":1,
        "max_output_tokens":2048,"request_timeout_seconds":2
    })).unwrap()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "execute-cell", "--input"])
        .arg(input)
        .arg("--requests-output")
        .arg(&requests)
        .arg("--summary-output")
        .arg(&summary)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        let mut request = [0; 4096];
        assert!(stream.read(&mut request).unwrap() > 0);
        assert!(
            Command::new("/bin/kill")
                .args(["-TERM", &pid.to_string()])
                .status()
                .unwrap()
                .success()
        );
        let mut remainder = Vec::new();
        stream.read_to_end(&mut remainder).unwrap();
    });
    let result = child.wait_with_output().unwrap();
    server.join().unwrap();
    assert!(!result.status.success());
    let rows = std::fs::read_to_string(requests).unwrap();
    assert_eq!(rows.lines().count(), 1);
    let record: serde_json::Value = serde_json::from_str(rows.trim()).unwrap();
    assert_eq!(record["request_id"], "interrupted:0");
    assert!(record["error"].is_string());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(summary).unwrap()).unwrap();
    assert_eq!(report["acceptance"]["passed"], false);
}
