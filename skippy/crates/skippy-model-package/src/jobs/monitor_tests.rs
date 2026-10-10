use super::{
    MonitorEnd, MonitorLimits, TransportLimits,
    transport_tests::{Peer, response, runtime},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
fn status(stage: &str, id: &str) -> Vec<u8> {
    response(200, format!("{{\"id\":\"{id}\",\"status\":{{\"stage\":\"{stage}\",\"message\":\"private-server-diagnostic\"}}}}").as_bytes())
}
fn limits() -> MonitorLimits {
    MonitorLimits {
        poll_interval: Duration::from_millis(2),
        ..MonitorLimits::default()
    }
}
#[test]
fn jobs_monitor_correlates_terminal_status_and_excludes_private_messages() {
    for (stage, expected) in [
        ("COMPLETED", MonitorEnd::Completed),
        ("ERROR", MonitorEnd::TerminalFailure),
        ("CANCELED", MonitorEnd::TerminalFailure),
    ] {
        let peer = Peer::many(vec![(status(stage, "job-1"), false)]);
        let receipt = runtime().block_on(peer.client(TransportLimits::default()).monitor_until(
            "owner",
            "job-1",
            Instant::now() + Duration::from_secs(3),
            std::future::pending(),
            limits(),
            |_| Ok(()),
        ));
        assert_eq!(receipt.end, expected);
        assert_eq!(receipt.polls, 1);
        let public = serde_json::to_string(&receipt).unwrap();
        assert!(!public.contains("private-server-diagnostic"));
        assert!(!receipt.remote_cancel_requested && !receipt.remote_cancel_confirmed);
    }
    let peer = Peer::many(vec![(status("COMPLETED", "foreign-job"), false)]);
    let receipt = runtime().block_on(peer.client(TransportLimits::default()).monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(3),
        std::future::pending(),
        limits(),
        |_| Ok(()),
    ));
    assert_eq!(receipt.end, MonitorEnd::CorrelationFailure);
    assert!(receipt.last_stage.is_none());
}
#[test]
fn jobs_monitor_reconnects_under_one_budget_and_keeps_observed_log_counts() {
    let peer = Peer::many(vec![
        (status("RUNNING", "job-1"), false),
        (response(200, b"data: first\n"), false),
        (status("RUNNING", "job-1"), false),
        (response(200, b"data: second\n"), false),
        (status("COMPLETED", "job-1"), false),
    ]);
    let mut rows = Vec::new();
    let receipt = runtime().block_on(peer.client(TransportLimits::default()).monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(4),
        std::future::pending(),
        limits(),
        |text| {
            rows.push(text.to_owned());
            Ok(())
        },
    ));
    assert_eq!(receipt.end, MonitorEnd::Completed);
    assert_eq!(rows, ["first", "second"]);
    assert_eq!(receipt.polls, 3);
    assert_eq!(receipt.reconnects, 2);
    assert_eq!(receipt.log_lines, 2);
    assert_eq!(receipt.log_bytes, 11);
}
#[test]
fn jobs_monitor_causal_cancel_after_real_log_drops_owned_stream_without_remote_cancel() {
    let peer = Peer::many(vec![
        (status("RUNNING", "job-1"), false),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\nConnection: close\r\n\r\ndata: observed\n"
                .to_vec(),
            true,
        ),
    ]);
    let cancelled = Arc::new(AtomicBool::new(false));
    let observer_cancel = cancelled.clone();
    let future_cancel = cancelled.clone();
    let cancellation = async move {
        while !future_cancel.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    };
    let receipt = runtime().block_on(peer.client(TransportLimits::default()).monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(3),
        cancellation,
        limits(),
        |text| {
            assert_eq!(text, "observed");
            observer_cancel.store(true, Ordering::SeqCst);
            Ok(())
        },
    ));
    assert!(cancelled.load(Ordering::SeqCst));
    assert_eq!(receipt.end, MonitorEnd::Cancelled);
    assert_eq!(receipt.log_lines, 1);
    assert!(!receipt.remote_cancel_requested && !receipt.remote_cancel_confirmed);
    let requests = peer.requests.try_iter().collect::<Vec<_>>();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|r| !String::from_utf8_lossy(r).starts_with("POST"))
    );
}
#[test]
fn jobs_monitor_pending_and_stream_deadlines_are_whole_operation_limits() {
    let peer = Peer::many(vec![(status("PENDING", "job-1"), false)]);
    let mut pending_limits = limits();
    pending_limits.poll_interval = Duration::from_secs(3);
    let receipt = runtime().block_on(peer.client(TransportLimits::default()).monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(1),
        std::future::pending(),
        pending_limits,
        |_| Ok(()),
    ));
    assert_eq!(receipt.end, MonitorEnd::Deadline);
    assert_eq!(receipt.polls, 1);
    assert_eq!(receipt.reconnects, 0);
    let peer = Peer::many(vec![
        (status("RUNNING", "job-1"), false),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\nConnection: close\r\n\r\ndata: observed\n"
                .to_vec(),
            true,
        ),
    ]);
    let receipt = runtime().block_on(peer.client(TransportLimits::default()).monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(1),
        std::future::pending(),
        limits(),
        |_| Ok(()),
    ));
    assert_eq!(receipt.end, MonitorEnd::Deadline);
    assert_eq!(receipt.log_lines, 1);
    assert_eq!(receipt.polls, 1);
}
#[test]
fn jobs_monitor_caps_reconnects_polls_and_logs_and_retains_prior_counts() {
    let peer = Peer::many(vec![
        (status("RUNNING", "job-1"), false),
        (response(200, b"data: first\ndata: second\n"), false),
    ]);
    let mut cap = limits();
    cap.max_log_lines = 1;
    let receipt = runtime().block_on(peer.client(TransportLimits::default()).monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(3),
        std::future::pending(),
        cap,
        |_| Ok(()),
    ));
    assert_eq!(receipt.end, MonitorEnd::LimitFailure);
    assert_eq!(receipt.log_lines, 1);
    let peer = Peer::many(vec![(status("PENDING", "job-1"), false)]);
    let mut cap = limits();
    cap.max_polls = 1;
    let receipt = runtime().block_on(peer.client(TransportLimits::default()).monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(3),
        std::future::pending(),
        cap,
        |_| Ok(()),
    ));
    assert_eq!(receipt.end, MonitorEnd::LimitFailure);
    assert_eq!(receipt.polls, 1);
    let peer = Peer::many(vec![
        (status("RUNNING", "job-1"), false),
        (response(200, b"data: first\n"), false),
        (status("RUNNING", "job-1"), false),
    ]);
    let mut cap = limits();
    cap.max_reconnects = 1;
    let receipt = runtime().block_on(peer.client(TransportLimits::default()).monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(3),
        std::future::pending(),
        cap,
        |_| Ok(()),
    ));
    assert_eq!(receipt.end, MonitorEnd::LimitFailure);
    assert_eq!(receipt.reconnects, 1);
    assert_eq!(receipt.log_lines, 1);
}
#[test]
fn jobs_monitor_precancel_and_invalid_limits_refuse_before_any_request() {
    let peer = Peer::many(vec![(status("COMPLETED", "job-1"), false)]);
    let client = peer.client(TransportLimits::default());
    let receipt = runtime().block_on(client.monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(3),
        std::future::ready(()),
        limits(),
        |_| Ok(()),
    ));
    assert_eq!(receipt.end, MonitorEnd::Cancelled);
    assert_eq!(receipt.polls, 0);
    let mut invalid = limits();
    invalid.max_polls = 0;
    let receipt = runtime().block_on(client.monitor_until(
        "owner",
        "job-1",
        Instant::now() + Duration::from_secs(3),
        std::future::pending(),
        invalid,
        |_| Ok(()),
    ));
    assert_eq!(receipt.end, MonitorEnd::InvalidLimits);
    assert_eq!(receipt.polls, 0);
    assert!(matches!(
        peer.requests.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
}
#[test]
fn jobs_cancel_acknowledgment_is_consumed_as_unconfirmed_cli_output_without_inspect() {
    let peer = Peer::many(vec![(response(202, b""), false)]);
    let client = peer.client(TransportLimits::default());
    let receipt = runtime()
        .block_on(client.cancel_receipt("owner", "job-1"))
        .unwrap();
    // This is the exact typed serializer consumed by run_cancel, not duplicated booleans.
    let output: serde_json::Value =
        serde_json::from_str(&serde_json::to_string_pretty(&receipt).unwrap()).unwrap();
    assert_eq!(output["namespace"], "owner");
    assert_eq!(output["jobId"], "job-1");
    assert_eq!(output["cancelRequested"], true);
    assert_eq!(output["cancelAccepted"], true);
    assert_eq!(output["cancelConfirmed"], false);
    assert_eq!(output["canceled"], false);
    let requests = peer.requests.try_iter().collect::<Vec<_>>();
    assert_eq!(requests.len(), 1);
    assert!(
        String::from_utf8_lossy(&requests[0])
            .starts_with("POST /api/jobs/owner/job-1/cancel HTTP/1.1")
    );
}
