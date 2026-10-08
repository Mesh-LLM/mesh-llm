#[path = "../src/cli_output.rs"]
mod cli_output;
#[path = "../src/automation/replay_matrix/progress.rs"]
mod progress;

use progress::{Event, Metrics, Outcome, Phase, Record, Request};
use std::{
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn request(session: &str) -> Record {
    let mut record = Record::boundary(Phase::Preflight, Event::Started, "cohort-2".into());
    record.request = Some(Request {
        request_id: format!("{session}:0"),
        session_id: session.into(),
        assistant_turn: 0,
    });
    record.concurrency = Some(2);
    record
}
fn wire(record: &Record) -> Vec<u8> {
    format!(
        "{}{}",
        progress::PREFIX,
        serde_json::to_string(record).unwrap()
    )
    .into_bytes()
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn completed_probe_preserves_identity_prompt_elapsed_and_outcome() {
    let mut record = request("s");
    record.event = Event::Completed;
    record.elapsed_seconds = 4.25;
    record.outcome = Some(Outcome::Success);
    record.metrics = Some(Metrics {
        prompt_tokens: Some(32768),
        completion_tokens: Some(1),
        ..Default::default()
    });
    let decoded = progress::decode(&wire(&record)).unwrap();
    assert_eq!(decoded, record);
    assert_eq!(decoded.request.unwrap().request_id, "s:0");
    assert_eq!(decoded.metrics.unwrap().prompt_tokens, Some(32768));
}

#[test]
fn rejects_telemetry_malformed_unknown_and_oversized_records() {
    assert!(progress::decode(br#"{"phase":"server","message":"private"}"#).is_none());
    assert!(progress::decode(b"replay-progress-v1 not-json").is_none());
    let mut value = serde_json::to_value(request("s")).unwrap();
    value["secret"] = "do not forward".into();
    assert!(progress::decode(format!("{}{}", progress::PREFIX, value).as_bytes()).is_none());
    assert!(progress::decode(&vec![b'x'; 8193]).is_none());
    let mut record = request("s");
    record.elapsed_seconds = -1.0;
    assert!(progress::decode(&wire(&record)).is_none());
}

#[test]
fn stalled_local_http_request_beats_before_completion_with_injected_clock() {
    runtime().block_on(async {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let (release, wait) = tokio::sync::oneshot::channel();
        let server = async {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = [0; 64];
            let count = stream.read(&mut bytes).await.unwrap();
            assert!(bytes[..count].starts_with(b"GET /"));
            wait.await.unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
        };
        let operation = async {
            let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
            stream
                .write_all(b"GET / HTTP/1.1\r\nHost: fixture\r\n\r\n")
                .await
                .unwrap();
            let mut bytes = [0; 64];
            stream.read(&mut bytes).await.unwrap()
        };
        let mut release = Some(release);
        let mut observed = Vec::new();
        let measured = progress::monitor(
            operation,
            request("s"),
            Duration::from_millis(1),
            || Duration::from_secs(77),
            |record| {
                observed.push(record.clone());
                if let Some(release) = release.take() {
                    release.send(()).unwrap();
                }
            },
        );
        let (count, ()) = tokio::time::timeout(Duration::from_secs(2), async {
            tokio::join!(measured, server)
        })
        .await
        .unwrap();
        assert!(count > 0);
        assert!(!observed.is_empty());
        assert!(
            observed
                .iter()
                .all(|record| record.event == Event::Heartbeat && record.elapsed_seconds == 77.0)
        );
    });
}

#[test]
fn success_error_timeout_and_cancelled_operations_stop_the_timer() {
    runtime().block_on(async {
        for outcome in [
            Outcome::Success,
            Outcome::Error,
            Outcome::Timeout,
            Outcome::Cancelled,
        ] {
            let (release, wait) = tokio::sync::oneshot::channel();
            let records = Arc::new(Mutex::new(Vec::new()));
            let output = Arc::clone(&records);
            let mut release = Some(release);
            let operation = async {
                wait.await.unwrap();
                outcome
            };
            let completed = tokio::time::timeout(
                Duration::from_secs(2),
                progress::monitor(
                    operation,
                    request("s"),
                    Duration::from_millis(1),
                    || Duration::from_secs(1),
                    move |record| {
                        output.lock().unwrap().push(record.clone());
                        if let Some(release) = release.take() {
                            release.send(()).unwrap();
                        }
                    },
                ),
            )
            .await
            .unwrap();
            assert_eq!(completed, outcome);
            let count = records.lock().unwrap().len();
            tokio::time::sleep(Duration::from_millis(5)).await;
            assert_eq!(records.lock().unwrap().len(), count);
        }
        assert_eq!(progress::HEARTBEAT, Duration::from_secs(30));
        assert_eq!(
            progress::in_flight(async { 42 }, request("instant")).await,
            42
        );
    });
}

#[test]
fn dropping_an_in_flight_future_cancels_its_heartbeat() {
    runtime().block_on(async {
        let (cancel, cancelled) = tokio::sync::oneshot::channel();
        let count = Arc::new(Mutex::new(0));
        let seen = Arc::clone(&count);
        let mut cancel = Some(cancel);
        let monitored = progress::monitor(std::future::pending::<()>(), request("cancelled"),
            Duration::from_millis(1), || Duration::ZERO, move |_| {
                *seen.lock().unwrap() += 1;
                if let Some(cancel) = cancel.take() { cancel.send(()).unwrap(); }
            });
        tokio::time::timeout(Duration::from_secs(2), async {
            tokio::select! { () = monitored => panic!("pending request completed"), result = cancelled => result.unwrap() }
        }).await.unwrap();
        let before = *count.lock().unwrap();
        assert!(before > 0);
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert_eq!(*count.lock().unwrap(), before);
    });
}

#[test]
fn concurrent_sessions_keep_distinct_request_identities() {
    runtime().block_on(async {
        let output = Arc::new(Mutex::new(Vec::new()));
        let one = Arc::clone(&output);
        let two = Arc::clone(&output);
        let (release_one, wait_one) = tokio::sync::oneshot::channel();
        let (release_two, wait_two) = tokio::sync::oneshot::channel();
        let mut release_one = Some(release_one);
        let mut release_two = Some(release_two);
        let left = progress::monitor(
            async { wait_one.await.unwrap() },
            request("left"),
            Duration::from_millis(1),
            || Duration::ZERO,
            move |record| {
                one.lock().unwrap().push(record.clone());
                if let Some(release) = release_one.take() {
                    release.send(()).unwrap();
                }
            },
        );
        let right = progress::monitor(
            async { wait_two.await.unwrap() },
            request("right"),
            Duration::from_millis(1),
            || Duration::ZERO,
            move |record| {
                two.lock().unwrap().push(record.clone());
                if let Some(release) = release_two.take() {
                    release.send(()).unwrap();
                }
            },
        );
        tokio::time::timeout(Duration::from_secs(2), async { tokio::join!(left, right) })
            .await
            .unwrap();
        let identities = output
            .lock()
            .unwrap()
            .iter()
            .map(|record| record.request.as_ref().unwrap().request_id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(identities, ["left:0".into(), "right:0".into()].into());
    });
}

#[test]
fn forwarding_retains_only_decoded_records_and_propagates_writer_failure() {
    let (send, receive) = mpsc::channel();
    std::thread::scope(|scope| {
        let forwarder = progress::Forwarder::with_sink(scope, move |record| {
            send.send(record.clone()).unwrap();
            Ok(())
        });
        assert!(!forwarder.line(b"native server telemetry").unwrap());
        assert!(forwarder.line(&wire(&request("worker"))).unwrap());
        forwarder.finish().unwrap();
    });
    assert_eq!(
        receive.try_iter().collect::<Vec<_>>(),
        vec![request("worker")]
    );
    std::thread::scope(|scope| {
        let forwarder =
            progress::Forwarder::with_sink(scope, |_| Err(std::io::Error::other("closed output")));
        assert!(forwarder.line(&wire(&request("worker"))).unwrap());
        assert!(forwarder.finish().is_err());
    });
    // Production constructor uses the same scoped drain and is finite when empty.
    std::thread::scope(|scope| progress::Forwarder::new(scope).finish().unwrap());
    let _stdout_factory = cli_output::stdout;
}

#[test]
fn phase_boundaries_and_raw_request_jsonl_use_separate_encodings() {
    for phase in [
        Phase::Preflight,
        Phase::Pass,
        Phase::Warmup,
        Phase::Cell,
        Phase::Request,
    ] {
        assert_eq!(
            progress::decode(&wire(&Record::boundary(
                phase,
                Event::Started,
                "fixture".into()
            )))
            .unwrap()
            .phase,
            phase
        );
    }
    let raw = b"{\"request_id\":\"s:0\",\"prompt_tokens\":32768}\n";
    let value: serde_json::Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(value["request_id"], "s:0");
    assert!(progress::decode(raw).is_none());
}

#[test]
fn executed_probe_keeps_raw_jsonl_parseable_and_emits_completed_fields() {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut request = Vec::new();
        loop {
            let mut bytes = [0; 4096];
            let count = stream.read(&mut bytes).unwrap();
            assert!(count > 0);
            request.extend_from_slice(&bytes[..count]);
            if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                let length = String::from_utf8_lossy(&request[..end])
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":32768,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n";
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
        stream.write_all(body).unwrap();
    });
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("cell.json");
    let raw = directory.path().join("raw.jsonl");
    let summary = directory.path().join("summary.json");
    std::fs::write(&input, serde_json::to_vec(&serde_json::json!({
        "trajectories":[{"session_id":"fixture","source_dataset":"fixture","agent_framework":"fixture",
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"x"}]}],
        "model":"fixture","base_url":format!("http://{address}/v1"),"concurrency":1,
        "max_output_tokens":2,"request_timeout_seconds":2,"qualification_probe":true
    })).unwrap()).unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "execute-cell", "--input"])
        .arg(input)
        .arg("--requests-output")
        .arg(&raw)
        .arg("--summary-output")
        .arg(&summary)
        .output()
        .unwrap();
    server.join().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let raw = std::fs::read_to_string(raw).unwrap();
    let records: Vec<serde_json::Value> = raw
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["request_id"], "fixture:0");
    assert_eq!(records[0]["prompt_tokens"], 32768);
    assert!(!raw.contains(progress::PREFIX));
    let progress: Vec<_> = result
        .stderr
        .split(|byte| *byte == b'\n')
        .filter_map(progress::decode)
        .collect();
    let probe = progress
        .iter()
        .find(|record| record.request.is_some() && record.event == Event::Completed)
        .unwrap();
    assert_eq!(probe.phase, Phase::Preflight);
    assert_eq!(probe.outcome, Some(Outcome::Success));
    assert_eq!(probe.metrics.as_ref().unwrap().prompt_tokens, Some(32768));
    assert!(probe.elapsed_seconds >= 0.0);
    let cell = progress
        .iter()
        .find(|record| record.request.is_none() && record.event == Event::Completed)
        .unwrap();
    assert_eq!(cell.metrics.as_ref().unwrap().completion_tokens, Some(1));
}

#[cfg(unix)]
#[test]
fn worker_records_reach_retained_server_cell_parent_output() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let reservation = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = reservation.local_addr().unwrap().port();
    let binary = directory.path().join("server-fixture");
    std::fs::write(
        &binary,
        format!(
            "#!/bin/sh\nexec '{}' --port {port} \"$@\"\n",
            env!("CARGO_BIN_EXE_laya-product-fixture")
        ),
    )
    .unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    let workload = directory.path().join("workload.json");
    let raw = directory.path().join("raw.jsonl");
    let summary = directory.path().join("summary.json");
    std::fs::write(&workload, serde_json::to_vec(&serde_json::json!({
        "trajectories":[{"session_id":"worker-session","source_dataset":"fixture","agent_framework":"fixture",
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"recorded"}]}],
        "model":"pending","base_url":format!("http://127.0.0.1:{port}/v1"),"concurrency":1,
        "max_output_tokens":2048,"request_timeout_seconds":2,"qualification_probe":true
    })).unwrap()).unwrap();
    drop(reservation);
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "server-cell", "--binary"])
        .arg(binary)
        .arg("--native-runtime-root")
        .arg(directory.path())
        .args(["--model", "fixture"])
        .arg("--workload")
        .arg(workload)
        .arg("--requests-output")
        .arg(&raw)
        .arg("--summary-output")
        .arg(summary)
        .arg("--server-log")
        .arg(directory.path().join("server.log"))
        .args(["--timeout", "10", "--startup-timeout", "3"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let records: Vec<_> = result
        .stderr
        .split(|byte| *byte == b'\n')
        .filter_map(progress::decode)
        .collect();
    assert!(records.iter().any(|record| {
        record
            .request
            .as_ref()
            .is_some_and(|request| request.request_id == "worker-session:0")
            && record.event == Event::Completed
    }));
    assert!(result.stdout.is_empty());
    for line in std::fs::read_to_string(raw).unwrap().lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(record["request_id"], "worker-session:0");
    }
}
