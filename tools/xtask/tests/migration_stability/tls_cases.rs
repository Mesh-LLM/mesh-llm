#[path = "tls_fixture.rs"]
mod tls_fixture;
use super::{answer, arguments, call, fixture, process, sse, stream_call};
use serde_json::json;
use std::{collections::BTreeMap, ffi::OsString, sync::atomic::Ordering};
use tls_fixture::{Reply, Server};

fn runtime_environment() -> BTreeMap<OsString, process::Value> {
    #[cfg(windows)]
    {
        ["PATH", "SystemRoot", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"]
            .into_iter()
            .filter_map(|key| {
                std::env::var_os(key).map(|value| (key.into(), process::Value::Public(value)))
            })
            .collect()
    }
    #[cfg(not(windows))]
    {
        BTreeMap::new()
    }
}

fn trusted(server: &Server) -> BTreeMap<OsString, process::Value> {
    let mut environment = runtime_environment();
    environment.extend([
        (
            "CURL_CA_BUNDLE".into(),
            process::Value::Public(server.ca.clone().into_os_string()),
        ),
        ("NO_PROXY".into(), process::Value::Public("*".into())),
    ]);
    environment
}

#[test]
fn stability_https_cli_verifies_fixture_ca_and_preserves_real_tool_continuations() {
    let server = Server::new(vec![
        Reply::json(200, &call()),
        Reply::json(200, &answer("signal-7429")),
        Reply::stream(sse(&stream_call()), false),
        Reply::stream(
            sse(&[json!({"choices":[{"delta":{"content":"signal-7429"}}]})]),
            false,
        ),
    ]);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("results.jsonl");
    let relative_ca = "fixture ca with spaces.pem";
    std::fs::copy(&server.ca, directory.path().join(relative_ca)).unwrap();
    let mut environment = trusted(&server);
    environment.insert(
        "CURL_CA_BUNDLE".into(),
        process::Value::Public(relative_ca.into()),
    );
    let result = fixture::invoke_with(
        directory.path(),
        &arguments("tool-call", &server.base, &output),
        environment,
    );
    assert_eq!(
        result.process.status.unwrap().code(),
        Some(0),
        "{:?}",
        result.process
    );
    let rows = fixture::rows(&output);
    assert_eq!(rows.len(), 4);
    assert!(
        rows.iter()
            .all(|row| row["ok"] == true && row["status_code"] == 200)
    );
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert!(
        requests[0]
            .0
            .starts_with("POST /tenant/v1/chat/completions ")
    );
    assert_eq!(requests[1].1["messages"][3]["tool_call_id"], "fixture-call");
    assert_eq!(requests[3].1["messages"][3]["tool_call_id"], "stream-call");
}

#[test]
fn stability_https_cli_rejects_untrusted_certificate_without_a_tool_continuation() {
    let server = Server::new(vec![Reply::json(200, &call())]);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("results.jsonl");
    let mut args = arguments("tool-call", &server.base, &output);
    args.extend(["--skip-streaming".into(), "--timeout".into(), "2".into()]);
    let result = fixture::invoke_with(directory.path(), &args, runtime_environment());
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    let rows = fixture::rows(&output);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["ok"], false);
    assert!(
        rows[0]["detail"]
            .as_str()
            .unwrap()
            .contains("certificate verification failed")
    );
    assert!(server.requests.lock().unwrap().is_empty());
}

#[test]
fn stability_https_cli_done_closes_still_open_streams_before_continuing() {
    let server = Server::new(vec![
        Reply::json(200, &call()),
        Reply::json(200, &answer("signal-7429")),
        Reply::stream(sse(&stream_call()), true),
        Reply::stream(
            sse(&[json!({"choices":[{"delta":{"content":"signal-7429"}}]})]),
            true,
        ),
    ]);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("results.jsonl");
    let result = fixture::invoke_with(
        directory.path(),
        &arguments("tool-call", &server.base, &output),
        trusted(&server),
    );
    assert_eq!(
        result.process.status.unwrap().code(),
        Some(0),
        "{:?}",
        result.process
    );
    assert_eq!(fixture::rows(&output).len(), 4);
    assert_eq!(server.requests.lock().unwrap().len(), 4);
    assert!(server.held_connections_closed.load(Ordering::SeqCst));
}

#[test]
fn stability_https_cli_deadline_closes_incomplete_transfer_and_preserves_failure() {
    let server = Server::new(vec![Reply::incomplete()]);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("results.jsonl");
    let mut args = arguments("tool-call", &server.base, &output);
    args.extend(["--skip-streaming".into(), "--timeout".into(), "0.5".into()]);
    let result = fixture::invoke_with(directory.path(), &args, trusted(&server));
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    let rows = fixture::rows(&output);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["ok"], false);
    assert!(
        rows[0]["detail"]
            .as_str()
            .unwrap()
            .contains("deadline exceeded")
    );
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    assert!(server.held_connections_closed.load(Ordering::SeqCst));
}

#[test]
fn stability_https_cli_redirect_error_status_and_invalid_json_never_continue() {
    for reply in [
        Reply::json(302, &json!({"redirect":"unfollowed"})),
        Reply::json(503, &json!({"error":"unavailable"})),
        Reply::bytes(200, b"not JSON".to_vec()),
    ] {
        let server = Server::new(vec![reply]);
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("results.jsonl");
        let mut args = arguments("tool-call", &server.base, &output);
        args.push("--skip-streaming".into());
        let result = fixture::invoke_with(directory.path(), &args, trusted(&server));
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let rows = fixture::rows(&output);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["ok"], false);
        assert_eq!(server.requests.lock().unwrap().len(), 1);
    }
}

#[test]
fn stability_https_cli_bounds_unknown_length_response_without_a_continuation() {
    let server = Server::new(vec![Reply::stream(
        "x".repeat(16 * 1024 * 1024 + 8192),
        false,
    )]);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("results.jsonl");
    let mut args = arguments("tool-call", &server.base, &output);
    args.push("--skip-streaming".into());
    let result = fixture::invoke_with(directory.path(), &args, trusted(&server));
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    let rows = fixture::rows(&output);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["ok"], false);
    assert!(
        rows[0]["detail"]
            .as_str()
            .unwrap()
            .contains("exceeds 16 MiB")
    );
    assert_eq!(server.requests.lock().unwrap().len(), 1);
}

#[test]
fn stability_https_cli_accepts_exact_limit_unknown_length_json_response() {
    let mut body = call().to_string();
    body.extend(std::iter::repeat_n(' ', 16 * 1024 * 1024 - body.len()));
    let server = Server::new(vec![
        Reply::stream(body, false),
        Reply::json(200, &answer("signal-7429")),
    ]);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("results.jsonl");
    let mut args = arguments("tool-call", &server.base, &output);
    args.push("--skip-streaming".into());
    let result = fixture::invoke_with(directory.path(), &args, trusted(&server));
    assert_eq!(
        result.process.status.unwrap().code(),
        Some(0),
        "{:?}",
        result.process
    );
    assert_eq!(fixture::rows(&output).len(), 2);
    assert_eq!(server.requests.lock().unwrap().len(), 2);
}

#[cfg(unix)]
#[test]
fn stability_https_cli_interrupt_preserves_signal_and_cleans_owned_private_files() {
    use std::{
        fs, thread,
        time::{Duration, Instant},
    };
    let server = Server::new(vec![Reply::incomplete()]);
    let directory = tempfile::tempdir().unwrap();
    let private = directory.path().join("private temporary files");
    fs::create_dir(&private).unwrap();
    let output = directory.path().join("results.jsonl");
    let mut args = arguments("tool-call", &server.base, &output);
    args.push("--skip-streaming".into());
    let cancellation = process::Cancellation::default();
    let cancel = cancellation.clone();
    let requests = server.requests.clone();
    let response_started = server.response_started.clone();
    let interrupt = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        while requests.lock().unwrap().is_empty() || !response_started.load(Ordering::SeqCst) {
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }
        cancel.cancel();
        true
    });
    let mut environment = trusted(&server);
    environment.insert(
        "TMPDIR".into(),
        process::Value::Public(private.clone().into_os_string()),
    );
    let result =
        fixture::invoke_with_cancellation(directory.path(), &args, environment, &cancellation);
    assert!(
        interrupt.join().unwrap(),
        "interrupt did not observe the active TLS request"
    );
    assert_eq!(
        result.process.status.unwrap().code(),
        Some(143),
        "{:?}",
        result.process
    );
    assert!(result.process.cleanup.complete && !result.process.cleanup.forced);
    let rows = fixture::rows(&output);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["ok"], false);
    assert!(rows[0]["detail"].as_str().unwrap().contains("cancelled"));
    assert!(server.held_connections_closed.load(Ordering::SeqCst));
    assert_eq!(
        fs::read_dir(private).unwrap().count(),
        0,
        "owned TLS files survived interruption"
    );
}
