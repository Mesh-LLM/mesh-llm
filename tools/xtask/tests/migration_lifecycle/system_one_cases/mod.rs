mod server;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::Value as Json;
use std::{collections::BTreeMap, fs, time::Duration};
fn invoke(
    server: &server::Server,
    mode: &str,
    timeout: &str,
    cancellation: &Cancellation,
) -> (process::RawProcessReport, Json) {
    let scratch = tempfile::tempdir().unwrap();
    let output = scratch.path().join("report.json");
    let args = vec![
        "automation".into(),
        "system-one-cases".into(),
        "--base-url".into(),
        server.url.clone(),
        "--model".into(),
        "fixture-model".into(),
        "--mode".into(),
        mode.into(),
        "--timeout".into(),
        timeout.into(),
        "--json-out".into(),
        output.to_str().unwrap().into(),
    ];
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: args
                .into_iter()
                .map(|arg| Value::Public(arg.into()))
                .collect(),
            cwd: scratch.path().into(),
            environment: BTreeMap::new(),
        },
        &Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(
        report.process.cleanup.complete && report.process.failure.is_none(),
        "{:?}",
        report.process
    );
    server.finish();
    let json = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
    (report, json)
}
#[test]
fn actual_contract_matrix_exercises_fourteen_refusals_and_consumed_report_statuses() {
    let server = server::Server::new("contract", "");
    let (report, json) = invoke(&server, "contract", "2", &Cancellation::default());
    assert!(report.process.success());
    assert_eq!(json["status"], "pass");
    assert_eq!(json["cases"][0]["cases"], 14);
    assert_eq!(server.calls.lock().unwrap().len(), 14);
    for mutation in [
        "missing-refusal",
        "wrong-code",
        "malformed",
        "choice-min",
        "choice-max",
        "score-min",
        "score-max",
        "arch-message",
        "arch-envelope",
        "arch-success",
        "unsupported-success",
        "unsupported-code",
    ] {
        let server = server::Server::new("contract", mutation);
        let (report, json) = invoke(&server, "contract", "2", &Cancellation::default());
        assert_eq!(report.process.status.unwrap().code(), Some(1), "{mutation}");
        assert_eq!(json["status"], "fail");
    }
}
#[test]
fn actual_full_read_validates_ranges_alias_usage_and_repeated_interleaved_state() {
    let server = server::Server::new("full-read", "");
    let (report, json) = invoke(&server, "full-read", "2", &Cancellation::default());
    assert!(report.process.success(), "{:?}", report.process);
    assert_eq!(json["cases"].as_array().unwrap().len(), 6);
    assert_eq!(server.calls.lock().unwrap().len(), 8);
    for mutation in [
        "unnormalized",
        "boolean",
        "bad-score",
        "bad-choice",
        "not-argmax",
        "alias",
        "usage",
        "generated",
        "constant",
        "leaked",
        "repeat-invalid",
    ] {
        let server = server::Server::new("full-read", mutation);
        let (report, json) = invoke(&server, "full-read", "2", &Cancellation::default());
        assert_eq!(
            report.process.status.unwrap().code(),
            Some(1),
            "{mutation}: {json}"
        );
        assert_eq!(json["status"], "fail");
    }
}
#[test]
fn actual_transport_redirect_cap_deadline_and_cancellation_report_error_without_pass() {
    for mutation in ["redirect", "oversized", "incomplete", "stall"] {
        let server = server::Server::new("contract", mutation);
        let (report, json) = invoke(&server, "contract", "0.1", &Cancellation::default());
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        assert_eq!(json["status"], "error");
    }
    let server = server::Server::new("contract", "stall");
    let cancellation = Cancellation::default();
    let token = cancellation.clone();
    let calls = server.calls.clone();
    let trigger = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while calls.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        token.cancel();
    });
    let (report, json) = invoke(&server, "contract", "5", &cancellation);
    trigger.join().unwrap();
    assert_eq!(report.process.outcome, process::Outcome::Cancelled);
    assert_eq!(json["status"], "error");
    assert!(
        !json["cases"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["status"] == "pass")
    );
}

mod wrapper;

mod admission;

#[test]
fn actual_ipv6_loopback_contract_dials_bare_address_with_http_bracketed_authority() {
    let listener = match std::net::TcpListener::bind("[::1]:0") {
        Ok(listener) => listener,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::AddrNotAvailable | std::io::ErrorKind::Unsupported
            ) =>
        {
            eprintln!("IPv6 loopback unavailable: {error}");
            return;
        }
        Err(error) => panic!("IPv6 fixture admission failed: {error}"),
    };
    let server = server::Server::with_listener(listener, "contract", "");
    let (report, json) = invoke(&server, "contract", "2", &Cancellation::default());
    assert!(report.process.success(), "{:?}", report.process);
    assert_eq!(json["status"], "pass");
    assert_eq!(server.calls.lock().unwrap().len(), 14);
}
