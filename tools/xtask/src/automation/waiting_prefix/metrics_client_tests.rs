use super::*;
use serde_json::{Value as Json, json};
use std::{
    io::{Read, Write},
    net::TcpListener,
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn endpoint(http: String) -> Endpoint {
    Endpoint {
        http,
        otlp_grpc: "http://127.0.0.1:14317".into(),
        run_id: "owned-run".into(),
        timeout_secs: 2,
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn export(finalized: bool) -> Json {
    let spans = [
        ("stage.openai_decode_token", 2000000, 2100000),
        ("stage.openai_generation_summary", 1000000, 3000000),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (name, start, end))| {
        json!({
            "run_id":"owned-run","request_id":"measured","stage_id":"stage-0",
            "trace_id":"trace", "span_id":index.to_string(),"name":name,
            "start_time_unix_nanos":start,"end_time_unix_nanos":end,
        })
    })
    .collect::<Vec<_>>();
    json!({"run":{"run_id":"owned-run","status":if finalized {"completed"} else {"running"},
        "finished_at_unix_nanos":if finalized {json!(4000000)} else {Json::Null}},
        "counts":{"spans":2},"telemetry_loss":{"dropped_events":0,"export_errors":0},
        "spans":spans})
}

fn server(responses: Vec<(u16, Json)>) -> (String, thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let mut requests = Vec::new();
        for (status, value) in responses {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "fixture accept deadline");
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("fixture accept: {error}"),
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
            loop {
                let mut buffer = [0_u8; 4096];
                let count = socket.read(&mut buffer).unwrap();
                assert_ne!(count, 0, "fixture request ended prematurely");
                bytes.extend_from_slice(&buffer[..count]);
                assert!(bytes.len() <= 128 * 1024);
                if let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = std::str::from_utf8(&bytes[..position]).unwrap();
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if bytes.len() >= position + 4 + length {
                        break;
                    }
                }
            }
            requests.push(String::from_utf8(bytes).unwrap());
            let body = serde_json::to_vec(&value).unwrap();
            write!(socket,"HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
            socket.write_all(&body).unwrap();
        }
        requests
    });
    (format!("http://{address}"), handle)
}

fn directory() -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "waiting-prefix-metrics-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    directory
}

#[test]
fn validates_local_roots_and_path_safe_run_identity() {
    let good = endpoint("http://127.0.0.1:18080".into());
    good.validate().unwrap();
    for http in [
        "https://127.0.0.1:18080",
        "http://localhost:18080",
        "http://127.0.0.1:0",
        "http://127.0.0.1",
        "http://127.0.0.1:18080/path",
        "http://127.0.0.1:18080/?x=1",
        "http://user@127.0.0.1:18080",
    ] {
        let bad = endpoint(http.into());
        assert!(bad.validate().is_err(), "{http}");
    }
    for id in ["", "../other", "with space", "run?x=1"] {
        let mut bad = good.clone();
        bad.run_id = id.into();
        assert!(bad.validate().is_err(), "{id}");
    }
}

#[test]
fn creates_drains_delayed_delivery_finalizes_and_retains_both_reports() {
    let mut pending = export(false);
    pending["spans"] = json!([]);
    pending["counts"]["spans"] = json!(0);
    let (url, handle) = server(vec![
        (200, json!({"run_id":"owned-run","status":"running"})),
        (200, pending),
        (200, export(false)),
        (200, json!({"run_id":"owned-run","status":"completed"})),
        (200, export(true)),
    ]);
    let directory = directory();
    let endpoint = endpoint(url);
    let cancellation = Cancellation::default();
    let rows = runtime().block_on(async {
        create(&endpoint, &json!({"round":1}), &cancellation)
            .await
            .unwrap();
        collect(&endpoint, &["measured".into()], &directory, &cancellation)
            .await
            .unwrap()
    });
    let requests = handle.join().unwrap();
    assert_eq!(requests.len(), 5);
    assert!(requests[0].starts_with("POST /v1/runs HTTP/1.1"));
    assert!(requests[3].starts_with("POST /v1/runs/owned-run/finalize HTTP/1.1"));
    assert_eq!(rows[0].server_ttft_ms, 1.0);
    let raw: Json =
        serde_json::from_slice(&std::fs::read(directory.join("metrics-report.json")).unwrap())
            .unwrap();
    assert_eq!(raw["run"]["status"], "completed");
    assert!(directory.join("metrics-timing.json").is_file());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn failed_finalization_retains_delivered_report_without_timing_success() {
    let (url, handle) = server(vec![
        (200, export(false)),
        (500, json!({"error":"fixture failure"})),
    ]);
    let directory = directory();
    let error = runtime()
        .block_on(collect(
            &endpoint(url),
            &["measured".into()],
            &directory,
            &Cancellation::default(),
        ))
        .unwrap_err();
    assert!(error.to_string().contains("500"));
    assert_eq!(handle.join().unwrap().len(), 2);
    assert!(directory.join("metrics-report.json").is_file());
    assert!(!directory.join("metrics-timing.json").exists());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn refuses_mismatched_create_identity_and_response_byte_overflow() {
    let (url, handle) = server(vec![(200, json!({"run_id":"other","status":"running"}))]);
    assert!(
        runtime()
            .block_on(create(&endpoint(url), &json!({}), &Cancellation::default()))
            .is_err()
    );
    handle.join().unwrap();
    let (url, handle) = server(vec![(200, json!({"oversized":"body"}))]);
    let admitted = endpoint(url);
    let error = runtime()
        .block_on(bounded(
            &admitted,
            &Cancellation::default(),
            &admitted.url("/status"),
            Method::GET,
            Vec::new(),
            4,
        ))
        .unwrap_err();
    assert!(error.to_string().contains("byte budget"));
    handle.join().unwrap();
}

#[test]
fn preexisting_cancellation_sends_no_collector_request() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = endpoint(format!("http://{}", listener.local_addr().unwrap()));
    let cancellation = Cancellation::default();
    cancellation.cancel();
    assert!(
        runtime()
            .block_on(create(&endpoint, &json!({}), &cancellation))
            .is_err()
    );
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn stalled_collector_releases_its_connection_at_the_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut endpoint = endpoint(format!("http://{}", listener.local_addr().unwrap()));
    endpoint.timeout_secs = 1;
    let handle = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut socket = loop {
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("{error}"),
            }
        };
        socket.set_nonblocking(false).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut buffer = [0_u8; 4096];
        let mut received = 0;
        loop {
            match socket.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => received += count,
                Err(error) if error.kind() == std::io::ErrorKind::ConnectionReset => break,
                Err(error) => panic!("client did not release collector connection: {error}"),
            }
        }
        assert!(received > 0);
    });
    let error = runtime()
        .block_on(create(&endpoint, &json!({}), &Cancellation::default()))
        .unwrap_err();
    assert!(error.to_string().contains("deadline"));
    handle.join().unwrap();
}
