//! The OTLP/gRPC collector sink: finished spans, batched by an exporter task
//! on the caller's runtime, retried from a bounded buffer.

use std::{
    collections::VecDeque,
    sync::{Arc, atomic::Ordering},
    thread,
    time::{Duration, Instant},
};

use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTraceServiceRequest, trace_service_client::TraceServiceClient,
};
use opentelemetry_proto::tonic::metrics::v1::Metric;
use skippy_protocol::StageConfig;
use tokio::{
    sync::mpsc,
    time::{sleep, timeout},
};

use super::{
    TelemetryCounters,
    otlp_encoding::{event_to_span, resource_spans_by_component},
    sink::{OpenSpan, SpanRecord, SpanStart, TraceSink},
    trace_context::{Frame, SpanEvent},
};

const EXPORT_TIMEOUT: Duration = Duration::from_secs(10);
const EXPORT_BATCH_MAX: usize = 128;
const RETRY_BUFFER_CAPACITY: usize = 8192;
const RETRY_BACKOFF: Duration = Duration::from_millis(100);

pub(super) struct OtlpCollectorSink {
    tx: mpsc::Sender<SpanRecord>,
    stats: Arc<TelemetryCounters>,
}

impl OtlpCollectorSink {
    /// Start exporting to `endpoint` from a task on the current runtime.
    pub(super) fn start(
        endpoint: String,
        capacity: usize,
        stats: Arc<TelemetryCounters>,
        config: Arc<StageConfig>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        tokio::spawn(telemetry_export_loop(endpoint, rx, stats.clone(), config));
        Self { tx, stats }
    }
}

impl TraceSink for OtlpCollectorSink {
    fn create_span_with(&self, _: &SpanStart) -> OpenSpan {
        OpenSpan::NONE
    }

    fn end_span(&self, _: OpenSpan, span: &SpanRecord) {
        self.record_span(span);
    }

    fn discard_span(&self, _: OpenSpan) {}

    fn record_span(&self, span: &SpanRecord) {
        match self.tx.try_send(span.clone()) {
            Ok(()) => {
                self.stats.queued.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Events reach the collector inside their span.
    fn record_event(&self, _: &SpanEvent, _: &Frame) {}

    /// Metrics have their own OTLP sink.
    fn record_metrics(&self, _: &[Metric]) {}

    fn folds_events_into_spans(&self) -> bool {
        true
    }

    fn flush(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        without_starving_runtime(|| {
            while self.stats.sent.load(Ordering::Relaxed)
                < self.stats.queued.load(Ordering::Relaxed)
                && Instant::now() < deadline
            {
                thread::sleep(Duration::from_millis(20));
            }
        });
    }
}

/// Run a blocking `wait` for the gRPC exporter, a task on the caller's
/// runtime. Blocking a runtime worker in place can starve that task: woken
/// from this worker, it sits in the worker's LIFO slot, which other workers
/// never steal. `block_in_place` hands the worker's queue to another thread
/// first. A current-thread runtime cannot run the exporter while blocked, so
/// waiting there would only stall.
fn without_starving_runtime(wait: impl FnOnce()) {
    match tokio::runtime::Handle::try_current().map(|handle| handle.runtime_flavor()) {
        Ok(tokio::runtime::RuntimeFlavor::MultiThread) => tokio::task::block_in_place(wait),
        Ok(_) => {}
        Err(_) => wait(),
    }
}

async fn telemetry_export_loop(
    endpoint: String,
    mut rx: mpsc::Receiver<SpanRecord>,
    stats: Arc<TelemetryCounters>,
    config: Arc<StageConfig>,
) {
    let mut client = None;
    let mut retry_buffer = VecDeque::new();

    loop {
        let Some(first_event) = next_export_event(&mut rx, &mut retry_buffer).await else {
            break;
        };
        let mut batch = Vec::with_capacity(EXPORT_BATCH_MAX);
        batch.push(first_event);
        fill_export_batch(&mut rx, &mut retry_buffer, &mut batch);

        let event_count = batch.len() as u64;
        if export_batch(&endpoint, &mut client, &config, &stats, &batch).await {
            stats.sent.fetch_add(event_count, Ordering::Relaxed);
        } else {
            stats
                .export_errors
                .fetch_add(event_count, Ordering::Relaxed);
            let dropped = requeue_failed_batch(&mut retry_buffer, batch);
            if dropped > 0 {
                stats.dropped.fetch_add(dropped as u64, Ordering::Relaxed);
            }
            sleep(RETRY_BACKOFF).await;
        }
    }
}

async fn next_export_event(
    rx: &mut mpsc::Receiver<SpanRecord>,
    retry_buffer: &mut VecDeque<SpanRecord>,
) -> Option<SpanRecord> {
    match retry_buffer.pop_front() {
        Some(event) => Some(event),
        None => rx.recv().await,
    }
}

fn fill_export_batch(
    rx: &mut mpsc::Receiver<SpanRecord>,
    retry_buffer: &mut VecDeque<SpanRecord>,
    batch: &mut Vec<SpanRecord>,
) {
    while batch.len() < EXPORT_BATCH_MAX {
        if let Some(event) = retry_buffer.pop_front() {
            batch.push(event);
            continue;
        }
        match rx.try_recv() {
            Ok(event) => batch.push(event),
            Err(_) => break,
        }
    }
}

async fn export_batch(
    endpoint: &str,
    client: &mut Option<TraceServiceClient<tonic::transport::Channel>>,
    config: &StageConfig,
    stats: &TelemetryCounters,
    events: &[SpanRecord],
) -> bool {
    if client.is_none() {
        match timeout(
            EXPORT_TIMEOUT,
            TraceServiceClient::connect(endpoint.to_string()),
        )
        .await
        {
            Ok(Ok(connected)) => {
                *client = Some(connected);
            }
            Ok(Err(_)) | Err(_) => return false,
        }
    }

    let request = ExportTraceServiceRequest {
        resource_spans: resource_spans_by_component(
            config,
            events
                .iter()
                .map(|event| event_to_span(event, stats))
                .collect(),
        ),
    };

    let export = client
        .as_mut()
        .expect("client is connected above")
        .export(request);
    match timeout(EXPORT_TIMEOUT, export).await {
        Ok(Ok(_)) => true,
        Ok(Err(_)) | Err(_) => {
            *client = None;
            false
        }
    }
}

fn requeue_failed_batch(retry_buffer: &mut VecDeque<SpanRecord>, batch: Vec<SpanRecord>) -> usize {
    let overflow = retry_buffer
        .len()
        .saturating_add(batch.len())
        .saturating_sub(RETRY_BUFFER_CAPACITY);
    for _ in 0..overflow {
        retry_buffer.pop_back();
    }
    for event in batch.into_iter().rev() {
        retry_buffer.push_front(event);
    }
    overflow
}

#[cfg(test)]
mod tests {
    use super::without_starving_runtime;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    /// Block on a one-worker runtime until a task spawned from that worker
    /// runs, or `patience` passes; true when the task ran.
    fn spawned_task_runs_while_waiting(
        patience: Duration,
        wait: fn(Box<dyn FnOnce() + '_>),
    ) -> bool {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async move {
            tokio::spawn(async move {
                let ran = Arc::new(AtomicBool::new(false));
                let task_ran = Arc::clone(&ran);
                tokio::spawn(async move { task_ran.store(true, Ordering::SeqCst) });
                let deadline = Instant::now() + patience;
                wait(Box::new(|| {
                    while !ran.load(Ordering::SeqCst) && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }));
                ran.load(Ordering::SeqCst)
            })
            .await
            .unwrap()
        })
    }

    #[test]
    fn waiting_on_a_runtime_worker_lets_the_workers_tasks_run() {
        assert!(spawned_task_runs_while_waiting(
            Duration::from_secs(5),
            |wait| without_starving_runtime(wait)
        ));
    }

    #[test]
    fn waiting_in_place_on_a_runtime_worker_starves_its_tasks() {
        // The failure the helper exists for; if this ever passes, tokio has
        // made the LIFO slot stealable.
        assert!(!spawned_task_runs_while_waiting(
            Duration::from_millis(200),
            |wait| wait()
        ));
    }

    #[test]
    fn a_current_thread_runtime_does_not_stall_waiting() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let started = Instant::now();
        runtime.block_on(async {
            without_starving_runtime(|| std::thread::sleep(Duration::from_secs(5)));
        });
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
