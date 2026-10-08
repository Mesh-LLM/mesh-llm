//! Actual xtask worker failure publishes correlated partial evidence, with an owned loopback peer.
use crate::process;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, num::NonZeroUsize, path::Path, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
struct Peer(tokio::task::JoinHandle<Vec<Value>>);
impl Drop for Peer {
    fn drop(&mut self) {
        self.0.abort();
    }
}
async fn peer() -> (u16, Peer) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(5),async move {
            let mut requests=Vec::new();
            for index in 0..3 {
                let (mut socket,_)=listener.accept().await.unwrap();
                let mut bytes=Vec::new();
                let boundary=loop {
                    let mut buffer=[0;1024];let n=socket.read(&mut buffer).await.unwrap();
                    assert!(n>0&&bytes.len()+n<=65536);bytes.extend_from_slice(&buffer[..n]);
                    if let Some(at)=bytes.windows(4).position(|w|w==b"\r\n\r\n"){break at+4;}
                };
                let header=std::str::from_utf8(&bytes[..boundary]).unwrap();
                let length=header.lines().find_map(|line|{let(k,v)=line.split_once(':')?;k.eq_ignore_ascii_case("content-length").then(||v.trim().parse::<usize>().unwrap())}).unwrap();
                assert!(length<=32768);
                while bytes.len()-boundary<length {let mut buffer=[0;1024];let n=socket.read(&mut buffer).await.unwrap();assert!(n>0);bytes.extend_from_slice(&buffer[..n]);}
                requests.push(serde_json::from_slice(&bytes[boundary..boundary+length]).unwrap());
                let body=if index==2 {"data: {\"error\":{\"message\":\"fixture refusal\"}}\n\n".to_owned()} else {
                    "data: {\"choices\":[{\"delta\":{\"content\":\"fixture\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n".to_owned()
                };
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();socket.shutdown().await.unwrap();
            }
            requests
        }).await.expect("bounded fixture peer")
    });
    (port, Peer(task))
}
fn invoke(root: &Path, input: &Path, output: &Path) -> process::RawProcessReport {
    process::supervise_raw(
        &process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            arguments: ["automation", "waiting-prefix", "sequential-cell", "--input"]
                .into_iter()
                .map(|s| process::Value::Public(s.into()))
                .chain([
                    process::Value::Public(input.as_os_str().to_owned()),
                    process::Value::Public("--output".into()),
                    process::Value::Public(output.as_os_str().to_owned()),
                ])
                .collect(),
            environment: BTreeMap::new(),
        },
        &process::Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}
fn clean_failure(raw: &process::RawProcessReport) {
    let p = &raw.process;
    assert_eq!(p.outcome, process::Outcome::Exited);
    assert_eq!(p.status.as_ref().unwrap().code(), Some(1));
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none()
    );
    assert!(p.stdout.line_capture_complete && p.stderr.line_capture_complete);
    assert_eq!(
        raw.stdout.as_ref().unwrap().as_bytes().len() as u64,
        p.stdout.bytes_seen
    );
    assert_eq!(
        raw.stderr.as_ref().unwrap().as_bytes().len() as u64,
        p.stderr.bytes_seen
    );
}
#[test]
fn adaptive_cli_measured_failure_atomically_publishes_correlated_partial_receipt() {
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
        let directory=tempfile::tempdir().unwrap();let root=directory.path();let (port,mut peer)=peer().await;
        let manifest=json!({"metadata":{"revision":"pinned"},"prompts":[{"family":"trace-1","prompt":"first","source_id":"source-1"},{"family":"trace-2","prompt":"second"}]});
        let input=json!({"schema_version":1,"round":2,"version":"new","base_url":format!("http://127.0.0.1:{port}/v1"),"model":"fixture-model","output_tokens":2,"request_timeout_secs":1.0,"timeout_secs":4,"readiness_timeout_secs":0,"prompt_manifest_sha256":hex::encode(Sha256::digest(serde_json::to_vec(&manifest).unwrap())),"manifest":manifest,"provenance":{}});
        // Public typed Input field order; compact identity differs from map-sorted wire JSON.
        let fields=["schema_version","round","version","base_url","model","output_tokens","request_timeout_secs","timeout_secs","readiness_timeout_secs","prompt_manifest_sha256","manifest","provenance"];
        let compact=format!("{{{}}}",fields.iter().map(|key|format!("{}:{}",serde_json::to_string(key).unwrap(),serde_json::to_string(&input[*key]).unwrap())).collect::<Vec<_>>().join(","));
        let expected_input_sha=hex::encode(Sha256::digest(compact.as_bytes()));
        let input_path=root.join("input.json");let output=root.join("receipt.json");std::fs::write(&input_path,serde_json::to_vec(&input).unwrap()).unwrap();
        let worker_root=root.to_owned();let worker_output=output.clone();
        let raw=tokio::task::spawn_blocking(move||invoke(&worker_root,&input_path,&worker_output)).await.unwrap();clean_failure(&raw);
        assert!(String::from_utf8_lossy(raw.stderr.as_ref().unwrap().as_bytes()).contains("evidence retained"));
        let receipt:Value=serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(receipt["schema_version"],1);assert_eq!(receipt["round"],2);assert_eq!(receipt["version"],"new");assert_eq!(receipt["model"],"fixture-model");
        assert_eq!(receipt["input_sha256"],expected_input_sha);
        assert!(receipt["error"].is_string());assert_eq!(receipt["successful_requests"],1);assert_eq!(receipt["requests"].as_array().unwrap().len(),2);
        assert!(receipt["requests"][0]["error"].is_null());assert!(receipt["requests"][1]["error"].is_string());assert_eq!(receipt["requests"][0]["prompt_provenance"]["source_id"],"source-1");
        assert!(receipt["calibration_request"].is_object());assert_eq!(receipt["timing_origin"],"measured-cell-start");
        let bodies=(&mut peer.0).await.unwrap();assert_eq!(bodies.len(),3);assert_eq!(bodies[0]["messages"],bodies[1]["messages"]);
        assert_eq!(std::fs::read_dir(root).unwrap().count(),2);directory.close().unwrap();
    });
}
#[test]
fn adaptive_cli_fifo_without_writer_refuses_before_open_or_publication() {
    use std::os::unix::ffi::OsStrExt as _;
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.json");
    let output = directory.path().join("receipt.json");
    let path = std::ffi::CString::new(input.as_os_str().as_bytes()).unwrap();
    // Live NUL-terminated owned fixture path.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let raw = invoke(directory.path(), &input, &output);
    clean_failure(&raw);
    assert!(
        String::from_utf8_lossy(raw.stderr.as_ref().unwrap().as_bytes())
            .contains("must be a regular file")
    );
    assert!(!output.exists());
    directory.close().unwrap();
}
