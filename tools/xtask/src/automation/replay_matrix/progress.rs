//! Typed operational progress. These records never enter request evidence files.
use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    io::{self, Write},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

pub(crate) const PREFIX: &str = "replay-progress-v1 ";
pub(crate) const HEARTBEAT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    Preflight,
    Pass,
    Warmup,
    Cell,
    Request,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Event {
    Started,
    Heartbeat,
    Completed,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Outcome {
    Success,
    Error,
    Timeout,
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub request_id: String,
    pub session_id: String,
    pub assistant_turn: usize,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Metrics {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub cache_pct: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub workload_output_tokens_per_second: Option<f64>,
    pub ttft_p50_seconds: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record {
    pub phase: Phase,
    pub event: Event,
    pub identity: String,
    pub request: Option<Request>,
    pub concurrency: Option<usize>,
    pub sessions: Option<usize>,
    pub cohorts: Option<usize>,
    pub turns: Option<usize>,
    pub elapsed_seconds: f64,
    pub outcome: Option<Outcome>,
    pub metrics: Option<Metrics>,
}

impl Record {
    pub(crate) fn boundary(phase: Phase, event: Event, identity: String) -> Self {
        Self {
            phase,
            event,
            identity,
            request: None,
            concurrency: None,
            sessions: None,
            cohorts: None,
            turns: None,
            elapsed_seconds: 0.0,
            outcome: None,
            metrics: None,
        }
    }
}

pub(crate) fn emit(record: &Record) -> io::Result<()> {
    let encoded = serde_json::to_string(record).map_err(io::Error::other)?;
    let mut output = crate::cli_output::stderr();
    writeln!(output, "{PREFIX}{encoded}")?;
    output.flush()
}

/// Only bounded typed records are forwarded. Child telemetry and prose are ignored.
pub(crate) fn decode(bytes: &[u8]) -> Option<Record> {
    if bytes.len() > 8192 {
        return None;
    }
    let text = std::str::from_utf8(bytes).ok()?.strip_prefix(PREFIX)?;
    let record: Record = serde_json::from_str(text).ok()?;
    if !record.elapsed_seconds.is_finite()
        || record.elapsed_seconds < 0.0
        || record.identity.len() > 1024
        || record.request.as_ref().is_some_and(|request| {
            request.request_id.len() > 1024 || request.session_id.len() > 1024
        })
    {
        return None;
    }
    Some(record)
}

/// One timer lives in the request future. Dropping it on timeout or cancellation
/// also drops its timer, so there is no detached heartbeat task to clean up.
pub(crate) async fn monitor<F, C, S>(
    future: F,
    mut record: Record,
    interval: Duration,
    clock: C,
    mut sink: S,
) -> F::Output
where
    F: Future,
    C: Fn() -> Duration,
    S: FnMut(&Record),
{
    assert!(!interval.is_zero(), "heartbeat interval must be positive");
    let mut timer = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    tokio::pin!(future);
    loop {
        tokio::select! {
            biased;
            result = &mut future => return result,
            _ = timer.tick() => {
                record.event = Event::Heartbeat;
                record.elapsed_seconds = clock().as_secs_f64();
                sink(&record);
            }
        }
    }
}

pub(crate) async fn in_flight<F: Future>(future: F, record: Record) -> F::Output {
    let started = Instant::now();
    monitor(
        future,
        record,
        HEARTBEAT,
        || started.elapsed(),
        |record| {
            let _ = emit(record);
        },
    )
    .await
}

/// Scoped session output owner. Callbacks only decode and enqueue; all console
/// I/O happens on this single worker, outside process-coordinator callbacks.
pub(crate) struct Forwarder<'scope> {
    sender: Option<mpsc::SyncSender<Record>>,
    worker: thread::ScopedJoinHandle<'scope, io::Result<()>>,
}
impl<'scope> Forwarder<'scope> {
    pub(crate) fn new<'env>(scope: &'scope thread::Scope<'scope, 'env>) -> Self {
        Self::with_sink(scope, emit)
    }
    pub(crate) fn with_sink<'env, S>(
        scope: &'scope thread::Scope<'scope, 'env>,
        mut sink: S,
    ) -> Self
    where
        S: FnMut(&Record) -> io::Result<()> + Send + 'scope,
    {
        let (sender, receiver) = mpsc::sync_channel::<Record>(256);
        let worker = scope.spawn(move || {
            for record in receiver {
                sink(&record)?;
            }
            Ok(())
        });
        Self {
            sender: Some(sender),
            worker,
        }
    }
    pub(crate) fn line(&self, bytes: &[u8]) -> Result<bool, String> {
        let Some(record) = decode(bytes) else {
            return Ok(false);
        };
        self.sender
            .as_ref()
            .ok_or("progress writer closed")?
            .try_send(record)
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => "progress output queue full".to_owned(),
                mpsc::TrySendError::Disconnected(_) => "progress output writer closed".to_owned(),
            })?;
        Ok(true)
    }
    pub(crate) fn finish(mut self) -> io::Result<()> {
        self.sender.take();
        self.worker
            .join()
            .map_err(|_| io::Error::other("progress output worker panicked"))?
    }
}
