use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::TcpListener,
    path::Path,
    process::Command,
    time::{Duration, Instant},
};
fn trajectory(id: &str, turns: usize) -> Value {
    let mut messages = vec![json!({"role":"user","content":"captured"})];
    for index in 0..turns {
        messages.push(json!({"role":"assistant","content":format!("recorded-{index}")}));
        messages.push(json!({"role":"tool","tool_call_id":format!("call-{index}"),"content":format!("observation-{index}")}));
    }
    json!({"session_id":id,"source_dataset":"capture","agent_framework":"goose","recorded_model":null,"messages":messages})
}
fn serve(listener: TcpListener, count: usize) -> Vec<Value> {
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut bodies = Vec::new();
    for _ in 0..count {
        let (mut socket, _) = loop {
            match listener.accept() {
                Ok(pair) => break pair,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        Instant::now() < deadline,
                        "CLI did not send expected requests"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("{error}"),
            }
        };
        socket.set_nonblocking(false).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut chunk = [0; 4096];
        loop {
            let n = socket.read(&mut chunk).unwrap();
            assert!(n > 0);
            bytes.extend_from_slice(&chunk[..n]);
            if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&bytes[..end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                if bytes.len() >= end + 4 + length {
                    break;
                }
            }
        }
        let end = bytes.windows(4).position(|v| v == b"\r\n\r\n").unwrap();
        bodies.push(serde_json::from_slice(&bytes[end + 4..]).unwrap());
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":19000,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":100}}}\n\ndata: [DONE]\n\n";
        socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",sse.len()).as_bytes()).unwrap();
    }
    bodies
}
fn cell(
    mode: Option<&str>,
    sessions: usize,
    turns: usize,
    expected: usize,
) -> (Value, Vec<Value>, Vec<Value>) {
    let temporary = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let mut input = json!({"trajectories":(0..sessions).map(|index|trajectory(&format!("s-{index}"),turns)).collect::<Vec<_>>(),"model":"fixture","base_url":base,"concurrency":1,"max_output_tokens":2048,"request_timeout_seconds":1});
    if let Some(mode) = mode {
        input["replay_mode"] = mode.into();
    }
    let path = temporary.path().join("input.json");
    std::fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let raw = temporary.path().join("raw.jsonl");
    let summary = temporary.path().join("summary.json");
    let server = std::thread::spawn(move || serve(listener, expected));
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "execute-cell", "--input"])
        .arg(path)
        .arg("--requests-output")
        .arg(&raw)
        .arg("--summary-output")
        .arg(&summary)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bodies = server.join().unwrap();
    let raw = std::fs::read_to_string(raw).unwrap();
    let rows = raw
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    for row in &rows {
        assert!(row.get("event").is_none());
        assert!(row.get("body").is_none());
    }
    let summary = serde_json::from_slice(&std::fs::read(summary).unwrap()).unwrap();
    (summary, rows, bodies)
}
#[test]
fn final_cell_cli_measures_original_last_id_and_complete_recorded_prefix() {
    let (summary, raw, bodies) = cell(Some("final"), 1, 4, 1);
    assert_eq!(summary["replay_mode"], "final");
    assert_eq!(summary["acceptance"]["passed"], true);
    assert_eq!(raw[0]["request_id"], "s-0:3");
    assert_eq!(raw[0]["assistant_turn"], 3);
    assert_eq!(bodies[0]["messages"].as_array().unwrap().len(), 7);
    assert_eq!(bodies[0]["messages"][6]["content"], "observation-2");
}
#[test]
fn checkpoint_cell_cli_balances_framework_stages_and_preserves_all_prefixes() {
    let (summary, raw, bodies) = cell(Some("checkpoint"), 4, 10, 4);
    assert_eq!(summary["replay_mode"], "checkpoint");
    assert_eq!(summary["acceptance"]["passed"], true);
    let mut indices = raw
        .iter()
        .map(|row| row["assistant_turn"].as_u64().unwrap())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    assert_eq!(indices, [0, 3, 6, 9]);
    for (row, body) in raw.iter().zip(bodies) {
        let index = row["assistant_turn"].as_u64().unwrap();
        assert_eq!(
            body["messages"].as_array().unwrap().len(),
            usize::try_from(index * 2 + 1).unwrap()
        );
    }
}
#[test]
fn omitted_cell_mode_still_executes_all_assistant_turns() {
    let (summary, raw, _) = cell(None, 1, 4, 4);
    assert_eq!(summary["replay_mode"], "all");
    assert_eq!(summary["acceptance"]["passed"], true);
    assert_eq!(
        raw.iter()
            .map(|row| row["request_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["s-0:0", "s-0:1", "s-0:2", "s-0:3"]
    );
}
#[test]
fn selected_cell_cannot_claim_full_session_qualification_before_io() {
    let temporary = tempfile::tempdir().unwrap();
    let input = json!({"trajectories":[trajectory("s",4)],"model":"fixture","base_url":"http://127.0.0.1:1/v1","concurrency":1,"max_output_tokens":1,"request_timeout_seconds":1,"replay_mode":"final","eligibility":{"context_tokens":20000,"maximum_output_tokens":1,"minimum_session_prompt_tokens":18000}});
    let path = temporary.path().join("input.json");
    std::fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let raw = temporary.path().join("raw.jsonl");
    let summary = temporary.path().join("summary.json");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "execute-cell", "--input"])
        .arg(path)
        .arg("--requests-output")
        .arg(&raw)
        .arg("--summary-output")
        .arg(&summary)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!raw.exists());
    assert!(!summary.exists());
}
#[cfg(unix)]
#[test]
fn documented_plan_cli_is_read_only_with_default_checkpoint_and_abba() {
    use std::os::unix::fs::PermissionsExt;
    let temporary = tempfile::tempdir().unwrap();
    let git = temporary.path().join("git");
    std::fs::write(&git,"#!/bin/sh\nset -eu\ntest \"$GIT_MASTER\" = 1\ntest \"$1\" = rev-parse\ncase \"$4\" in stable*) printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\n';; *) printf 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\\n';; esac\n").unwrap();
    std::fs::set_permissions(git, std::fs::Permissions::from_mode(0o700)).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args([
            "automation",
            "replay-matrix",
            "plan",
            "--repo",
            env!("CARGO_MANIFEST_DIR"),
        ])
        .args([
            "--ref",
            "stable=stable",
            "--ref",
            "main=HEAD",
            "--model",
            "hf://owner/model",
            "--passes",
            "2",
            "--trajectories-per-framework",
            "4",
        ])
        .arg("--repo")
        .arg(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap(),
        )
        .env("PATH", temporary.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let plan: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(plan["options"]["mode"], "checkpoint");
    assert_eq!(
        plan["order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["stable", "main", "main", "stable"]
    );
    assert_eq!(std::fs::read_dir(temporary.path()).unwrap().count(), 1);
}
#[test]
fn execute_run_captured_mesh_uri_admission_is_separate_from_local_model_pin_verification() {
    let temporary = tempfile::tempdir().unwrap();
    let manifest = temporary.path().join("manifest.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(
            &json!({"cohorts":{"warmup":[trajectory("w",4)],"1":[trajectory("s",4)]}}),
        )
        .unwrap(),
    )
    .unwrap();
    let host = temporary.path().join("host");
    std::fs::write(&host, b"fixture host identity, must never execute").unwrap();
    let runtime = temporary.path().join("runtime");
    std::fs::create_dir(&runtime).unwrap();
    let local_model = temporary.path().join("model.gguf");
    std::fs::write(&local_model, b"fixture wrong bytes").unwrap();
    for (model, sha, expected) in [
        (
            json!("hf://owner/model"),
            "",
            "replay arm build digest changed",
        ),
        (
            json!(local_model),
            &"0".repeat(64),
            "model SHA-256 mismatch",
        ),
        (
            json!("hf://owner/model"),
            &"0".repeat(64),
            "run requires absolute paths",
        ),
    ] {
        let output = temporary.path().join(format!(
            "artifact-{}",
            if sha.is_empty() {
                "uri"
            } else if model == json!(local_model) {
                "local-pin"
            } else {
                "uri-pin"
            }
        ));
        let input = json!({"manifest":manifest,"requirements":{"concurrency":[1],"minimum_worker_waves":1,"warmup_turns":4,"required_frameworks":[]},"builds":[{"label":"main","ref":"HEAD","commit":"a".repeat(40),"binary":host,"binary_sha256":"wrong","runtime_root":runtime,"runtime":runtime,"runtime_sha256":"wrong","worktree":temporary.path()}],"context_qualification":"captured","model":model,"model_reference":"hf://owner/model","model_sha256":sha,"passes":1,"max_output_tokens":1,"request_timeout_seconds":1,"startup_timeout_seconds":2,"timeout_seconds":3,"output":output,"replay_mode":"final"});
        let path = temporary.path().join("input.json");
        std::fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "execute-run", "--input"])
            .arg(path)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!output.exists());
    }
}
