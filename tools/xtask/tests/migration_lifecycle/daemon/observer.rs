use super::*;
use crate::process::{LineEnding, Stream};
use std::sync::mpsc;

struct Harness {
    observer: Observer,
    requests: Receiver<Request>,
    results: SyncSender<Result<Transfer, TransferError>>,
    id: RequestId,
}

impl Harness {
    fn new() -> Self {
        let (requests, work) = mpsc::sync_channel(1);
        let (results, responses) = mpsc::sync_channel(1);
        let id = RequestId::generate().unwrap();
        Self {
            observer: Observer::new(requests, responses, id, 2),
            requests: work,
            results,
            id,
        }
    }

    fn tick(&mut self, elapsed: u64) -> ProbeDecision<Rejection> {
        self.observer.tick(ProbeContext {
            pid: 42,
            elapsed: Duration::from_secs(elapsed),
            remaining: Duration::from_secs(10 - elapsed),
        })
    }

    fn status(&mut self) {
        assert!(matches!(self.tick(0), ProbeDecision::Pending));
        let request = self.requests.try_recv().unwrap();
        assert!(matches!(
            request.endpoint,
            Endpoint::Status { leader_pid: 42 }
        ));
        assert!(request.deadline <= Instant::now() + Duration::from_secs(5));
        self.results
            .send(Ok(Transfer {
                status_code: 200,
                body_bytes: 60,
            }))
            .unwrap();
        assert!(matches!(self.tick(0), ProbeDecision::Pending));
        let request = self.requests.try_recv().unwrap();
        assert!(
            matches!(request.endpoint, Endpoint::Models { request_id } if request_id.header() == self.id.header())
        );
    }

    fn record(&mut self, status: u16) -> ProbeDecision<Rejection> {
        let record = format!(
            r#"{{"request_id":"{}","source":"direct_http","route":"models","method":"GET","request_kind":"model_listing","event":"request_completed","outcome":"completed","status_code":{status}}}"#,
            self.id.header()
        );
        self.observer.line(ObservedLine {
            stream: Stream::Stdout,
            bytes: record.as_bytes(),
            ending: LineEnding::Lf,
        })
    }
}

#[test]
fn d11_correlation_before_transfer_is_provisional() {
    let mut harness = Harness::new();
    harness.status();
    assert!(matches!(harness.record(200), ProbeDecision::Pending));
    harness
        .results
        .send(Ok(Transfer {
            status_code: 200,
            body_bytes: 1,
        }))
        .unwrap();
    assert!(matches!(harness.tick(1), ProbeDecision::Candidate));
    assert!(harness.requests.try_recv().is_err());
}

#[test]
fn d11_transfer_before_correlation_is_provisional() {
    let mut harness = Harness::new();
    harness.status();
    harness
        .results
        .send(Ok(Transfer {
            status_code: 200,
            body_bytes: 1,
        }))
        .unwrap();
    assert!(matches!(harness.tick(1), ProbeDecision::Pending));
    assert!(matches!(harness.record(200), ProbeDecision::Candidate));
}

#[test]
fn d10_wrong_status_cannot_correlate() {
    let mut harness = Harness::new();
    harness.status();
    assert!(matches!(harness.record(201), ProbeDecision::Pending));
    harness
        .results
        .send(Ok(Transfer {
            status_code: 200,
            body_bytes: 1,
        }))
        .unwrap();
    assert!(matches!(harness.tick(1), ProbeDecision::Pending));
    assert_eq!(harness.observer.facts().phase, Phase::Attribution);
}

#[test]
fn d07_retry_spacing_and_attempt_limit() {
    let mut harness = Harness::new();
    assert!(matches!(harness.tick(0), ProbeDecision::Pending));
    harness.requests.try_recv().unwrap();
    harness.results.send(Err(TransferError::Transport)).unwrap();
    harness.tick(0);
    harness.tick(0);
    assert!(harness.requests.try_recv().is_err());
    harness.tick(1);
    harness.requests.try_recv().unwrap();
    harness.results.send(Err(TransferError::Timeout)).unwrap();
    harness.tick(1);
    harness.tick(9);
    assert!(harness.requests.try_recv().is_err());
    assert_eq!(harness.observer.facts().status_attempts, 2);
}

#[test]
fn d20_worker_disconnect_is_rejected() {
    let mut harness = Harness::new();
    harness.tick(0);
    drop(harness.results);
    assert!(matches!(
        harness.observer.tick(ProbeContext {
            pid: 42,
            elapsed: Duration::ZERO,
            remaining: Duration::from_secs(10)
        }),
        ProbeDecision::Rejected(Rejection::WorkerFailed)
    ));
}
