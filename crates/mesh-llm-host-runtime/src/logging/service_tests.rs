//! Tests for LoggingService — extracted to keep service.rs under 1000 LOC.

use crate::logging::{
    BusEntry, Clock, FailOpenWriter, LoggingDynamicLimits, PersistSink, RegistryConfig,
    TerminalOutcome,
};
use crate::logging::{
    OperationalAuditRecord, ReplayBus, RequestRegistry, RequestSummaryEntry, RequestSummaryMetadata,
};
use mesh_llm_events::logging::events::LifecycleEvent;
use mesh_llm_events::logging::identifiers::{AttemptId, EventId, RequestId};
use mesh_llm_events::logging::proxy::ProxyRecord;
use mesh_llm_events::logging::replay::ReplayChannel;
use mesh_llm_events::{OutputEvent, OutputSink, clear_output_sink, set_output_sink};

// Re-import service.rs types. These are private to the logging module but accessible via super.
#[allow(unused_imports)]
use crate::logging::service::{BusEnqueueError, LoggingService, ServiceConfig, SystemClock};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::time::Duration;
use tokio::sync::{Notify, mpsc};

mod configuration;
mod delivery_shutdown;
mod lifecycle_registry;

#[derive(Default)]
struct RecordingOutputSink {
    events: std::sync::Mutex<Vec<OutputEvent>>,
}

impl RecordingOutputSink {
    fn take_events(&self) -> Vec<OutputEvent> {
        std::mem::take(
            &mut *self
                .events
                .lock()
                .expect("recording output sink mutex poisoned"),
        )
    }
}

impl OutputSink for RecordingOutputSink {
    fn emit_event(&self, event: OutputEvent) -> std::io::Result<()> {
        self.events
            .lock()
            .expect("recording output sink mutex poisoned")
            .push(event);
        Ok(())
    }
}

struct OutputSinkResetGuard;

impl Drop for OutputSinkResetGuard {
    fn drop(&mut self) {
        clear_output_sink();
    }
}

// ---------------------------------------------------------------------------
// Test infrastructure: Vec-backed sink + deterministic clock
// ---------------------------------------------------------------------------

/// Record type for the test Vec-backed persistence sink. Captures all persisted data deterministically without I/O.
#[derive(Clone, Debug)]
enum TestRecord {
    Summary(RequestSummaryEntry),
    Event {
        request_id: String,
        event_id: String,
        channel: ReplayChannel,
        sequence: u64,
        occurred_at: String,
        payload_json: String,
    },
    ArtifactPointer(String, serde_json::Value), // (request_id, data)
    ProxyRecord(String),                        // JSON string
    AuditEntry {
        level: String,
        message: String,
        entry_id: Option<String>,
        occurred_at: Option<String>,
    },
    WebhookDelivery {
        request_id: Option<String>,
        status_code: u16,
        error: Option<String>,
    },
    CleanupRun(u64), // deleted_count
}

/// Vec-backed persistence sink for deterministic testing. All writes are recorded in a shared Mutex<Vec<TestRecord>> — no I/O, no sleeps.
struct TestSink {
    records: std::sync::Mutex<Vec<TestRecord>>,
    fail_flag: Arc<AtomicU64>, // if > 0, all operations return Err
    audit_notifications: Option<mpsc::UnboundedSender<(String, String)>>,
    audit_attempt_notifications: Option<mpsc::UnboundedSender<()>>,
}

impl TestSink {
    fn new() -> Self {
        Self {
            records: std::sync::Mutex::new(Vec::new()),
            fail_flag: Arc::new(AtomicU64::new(0)),
            audit_notifications: None,
            audit_attempt_notifications: None,
        }
    }

    fn recording() -> (Self, mpsc::UnboundedReceiver<(String, String)>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut sink = Self::new();
        sink.audit_notifications = Some(tx);
        (sink, rx)
    }

    fn failing_with_attempt_notifications() -> (Self, mpsc::UnboundedReceiver<()>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut sink = Self::new();
        sink.set_failing();
        sink.audit_attempt_notifications = Some(tx);
        (sink, rx)
    }

    /// Set the sink to return Err on all subsequent operations (simulates store failure).
    fn set_failing(&self) {
        self.fail_flag.store(1, AtomicOrdering::Release);
    }

    /// Clear the failing flag.
    #[allow(dead_code)]
    fn clear_fail(&self) {
        self.fail_flag.store(0, AtomicOrdering::Release);
    }

    /// Get all records captured so far (for test assertions).
    #[allow(dead_code)]
    fn records(&self) -> Vec<TestRecord> {
        self.records.lock().unwrap().clone()
    }

    /// Count of audit entries with a specific level.
    #[allow(dead_code)]
    fn audit_count_by_level(&self, level: &str) -> usize {
        self.records()
            .iter()
            .filter(|r| matches!(r, TestRecord::AuditEntry { level: lvl, .. } if lvl == level))
            .count()
    }

    /// Reset records to empty (for multi-phase tests).
    #[allow(dead_code)]
    fn clear(&self) {
        self.records.lock().unwrap().clear();
    }
}

#[async_trait::async_trait]
impl PersistSink for TestSink {
    async fn persist_summary(&self, entry: RequestSummaryEntry) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        let mut records = self.records.lock().unwrap();
        if let Some(TestRecord::Summary(existing)) = records.iter_mut().find(|record| {
            matches!(record, TestRecord::Summary(existing) if existing.request_id == entry.request_id)
        }) {
            *existing = entry;
        } else {
            records.push(TestRecord::Summary(entry));
        }
        Ok(())
    }

    async fn persist_event(
        &self,
        request_id: String,
        event_id: String,
        channel: ReplayChannel,
        sequence: u64,
        occurred_at: String,
        payload_json: String,
    ) -> Result<(), String> {
        if let Some(tx) = &self.audit_attempt_notifications {
            let _ = tx.send(());
        }
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records.lock().unwrap().push(TestRecord::Event {
            request_id,
            event_id,
            channel,
            sequence,
            occurred_at,
            payload_json,
        });
        Ok(())
    }

    async fn persist_artifact_pointer(
        &self,
        request_id: String,
        artifact_data: serde_json::Value,
    ) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::ArtifactPointer(request_id, artifact_data));
        Ok(())
    }

    async fn persist_proxy_record(&self, proxy_json: String) -> Result<(), String> {
        if let Some(tx) = &self.audit_attempt_notifications {
            let _ = tx.send(());
        }
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::ProxyRecord(proxy_json));
        Ok(())
    }

    async fn persist_audit_entry(&self, record: OperationalAuditRecord) -> Result<(), String> {
        if let Some(tx) = &self.audit_attempt_notifications {
            let _ = tx.send(());
        }
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        if let Some(tx) = &self.audit_notifications {
            let _ = tx.send((
                record
                    .severity()
                    .map_or("none", crate::logging::OperationalAuditSeverity::as_str)
                    .to_string(),
                record
                    .detail_json()
                    .unwrap_or_else(|| record.code())
                    .to_string(),
            ));
        }
        self.records.lock().unwrap().push(TestRecord::AuditEntry {
            level: record
                .severity()
                .map_or("none", crate::logging::OperationalAuditSeverity::as_str)
                .to_string(),
            message: record
                .detail_json()
                .unwrap_or_else(|| record.code())
                .to_string(),
            entry_id: record.entry_id().map(str::to_owned),
            occurred_at: record.occurred_at().map(str::to_owned),
        });
        Ok(())
    }

    async fn persist_webhook_delivery(
        &self,
        request_id: Option<String>,
        status_code: u16,
        error: Option<String>,
    ) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::WebhookDelivery {
                request_id,
                status_code,
                error,
            });
        Ok(())
    }

    async fn persist_cleanup_run(&self, deleted_count: u64) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::CleanupRun(deleted_count));
        Ok(())
    }
}

/// A sink that blocks only its first audit persistence until the test releases
/// it. This deterministically fills the service's bounded worker channel
/// without sleeps or unobserved background work.
struct BlockingAuditSink {
    first_write: AtomicBool,
    started: mpsc::UnboundedSender<()>,
    completed: mpsc::UnboundedSender<String>,
    release: Arc<Notify>,
}

impl BlockingAuditSink {
    fn new() -> (
        Self,
        mpsc::UnboundedReceiver<()>,
        mpsc::UnboundedReceiver<String>,
        Arc<Notify>,
    ) {
        let (started_tx, started_rx) = mpsc::unbounded_channel();
        let (completed_tx, completed_rx) = mpsc::unbounded_channel();
        let release = Arc::new(Notify::new());
        (
            Self {
                first_write: AtomicBool::new(true),
                started: started_tx,
                completed: completed_tx,
                release: Arc::clone(&release),
            },
            started_rx,
            completed_rx,
            release,
        )
    }
}

#[async_trait::async_trait]
impl PersistSink for BlockingAuditSink {
    async fn persist_summary(&self, _entry: RequestSummaryEntry) -> Result<(), String> {
        Ok(())
    }

    async fn persist_event(
        &self,
        _request_id: String,
        _event_id: String,
        _channel: ReplayChannel,
        _sequence: u64,
        _occurred_at: String,
        _payload_json: String,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn persist_artifact_pointer(
        &self,
        _request_id: String,
        _artifact_data: serde_json::Value,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn persist_proxy_record(&self, _proxy_json: String) -> Result<(), String> {
        Ok(())
    }

    async fn persist_audit_entry(&self, record: OperationalAuditRecord) -> Result<(), String> {
        if self.first_write.swap(false, AtomicOrdering::AcqRel) {
            let _ = self.started.send(());
            self.release.notified().await;
        }
        let _ = self.completed.send(
            record
                .detail_json()
                .unwrap_or_else(|| record.code())
                .to_string(),
        );
        Ok(())
    }

    async fn persist_webhook_delivery(
        &self,
        _request_id: Option<String>,
        _status_code: u16,
        _error: Option<String>,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn persist_cleanup_run(&self, _deleted_count: u64) -> Result<(), String> {
        Ok(())
    }
}

/// Deterministic counter clock for tests. Each call increments a counter, producing unique timestamps without wall-clock dependency.
struct TestClock {
    counter: AtomicU64,
}

impl TestClock {
    fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }
}

impl Clock for TestClock {
    fn now(&self) -> String {
        let n = self
            .counter
            .fetch_update(
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
                |current| current.checked_add(1),
            )
            .expect("test clock counter overflow");
        format!("2025-01-01T00:00:00.{n:09}Z")
    }
}

#[test]
fn test_clock_timestamps_remain_unique_past_a_minute() {
    let clock = TestClock::new();
    let first = clock.now();
    for _ in 0..59 {
        clock.now();
    }
    let sixty_first = clock.now();

    assert_eq!(first, "2025-01-01T00:00:00.000000000Z");
    assert_eq!(sixty_first, "2025-01-01T00:00:00.000000060Z");
    assert_ne!(first, sixty_first);
}

fn make_service() -> LoggingService {
    let sink = Arc::new(TestSink::new());
    let clock = Box::new(TestClock::new());
    let config = ServiceConfig {
        queue_capacity: 128,
        event_buffer_size: 128,
        artifact_command_max_bytes: ServiceConfig::default().artifact_command_max_bytes,
        registry_config: RegistryConfig {
            max_active: 50,
            max_recent: 100,
        },
        ..ServiceConfig::default()
    };
    LoggingService::new(config, sink, clock)
}

fn recorded_lifecycle_events(service: &LoggingService) -> Vec<(serde_json::Value, LifecycleEvent)> {
    recorded_lifecycle_events_including_admitted(service)
        .into_iter()
        .filter(|(_, event)| !matches!(event, LifecycleEvent::Admitted { .. }))
        .collect()
}

fn recorded_lifecycle_events_including_admitted(
    service: &LoggingService,
) -> Vec<(serde_json::Value, LifecycleEvent)> {
    service
        .bus_ref()
        .drain()
        .into_iter()
        .map(|entry| {
            let envelope: serde_json::Value = serde_json::from_str(&entry.payload).unwrap();
            let event = serde_json::from_str(envelope["payload"].as_str().unwrap()).unwrap();
            (envelope, event)
        })
        .collect()
}

fn canonical_persistence_failure_fallbacks(service: &LoggingService) -> usize {
    service
        .bus_ref()
        .drain()
        .into_iter()
        .filter_map(|entry| serde_json::from_str::<serde_json::Value>(&entry.payload).ok())
        .filter_map(|entry| entry.get("canonical_envelope").cloned())
        .filter_map(|envelope| {
            mesh_llm_events::logging::envelope::CanonicalEnvelope::from_json_str(
                &envelope.to_string(),
            )
            .ok()
        })
        .filter(|envelope| {
            envelope.channel == ReplayChannel::System
                && matches!(envelope.event, LifecycleEvent::AuditError { ref message }
                    if message == "logging persistence delivery failed")
        })
        .count()
}

// ---------------------------------------------------------------------------
// Test Scenario 1: One terminal record for each of complete/fail/reject/cancel/drop
// ---------------------------------------------------------------------------

#[test]
#[serial_test::serial]
fn canonical_events_reach_the_output_sink_once_with_safe_local_projection() {
    let sink = Arc::new(RecordingOutputSink::default());
    let _reset_guard = OutputSinkResetGuard;
    set_output_sink(sink.clone());

    let service = make_service();
    let request_id = RequestId::new();
    service
        .enqueue_event(
            request_id,
            ReplayChannel::Requests,
            serde_json::to_string(&LifecycleEvent::Failed {
                error: "prompt=private Bearer secret-token".to_string(),
                status_code: None,
            })
            .expect("lifecycle event serializes"),
        )
        .expect("canonical event enqueue is fail-open");

    let events = sink.take_events();
    let matching_envelopes = events
        .iter()
        .filter_map(|event| match event {
            OutputEvent::CanonicalLog(envelope) if envelope.request_id == request_id => {
                Some(envelope.as_ref())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [envelope] = matching_envelopes.as_slice() else {
        panic!("expected one canonical output event for {request_id:?}, got {events:?}");
    };
    assert_eq!(envelope.request_id, request_id);
    assert!(matches!(envelope.event, LifecycleEvent::Failed { .. }));
    assert_eq!(envelope.presentation_event_name(), "request_failed");
    assert_eq!(envelope.presentation_message(), "request failed");
    let local_summary = envelope.presentation_local_summary();
    assert!(local_summary.contains(&request_id.as_uuid().to_string()));
    assert!(local_summary.contains(&envelope.event_id.as_uuid().to_string()));
    assert!(!local_summary.contains("private"));
    assert!(!local_summary.contains("secret-token"));

    let (guard, _) = service.register_request(request_id);
    service
        .transition_terminal(request_id, &guard, TerminalOutcome::Completed)
        .expect("first terminal transition succeeds");
    assert!(
        service
            .transition_terminal(
                request_id,
                &guard,
                TerminalOutcome::Failed("ignored".into())
            )
            .is_err()
    );

    let terminal_events = sink.take_events();
    assert_eq!(
        terminal_events
            .iter()
            .filter(|event| matches!(
                event,
                OutputEvent::CanonicalLog(envelope)
                    if envelope.request_id == request_id
                        && envelope.presentation_outcome().is_some()
            ))
            .count(),
        1,
        "terminal lifecycle ownership must still dedupe at the presentation sink"
    );
}

#[test]
#[serial_test::serial]
fn canonical_projection_uses_request_summary_context_for_probe_and_terminal_lines() {
    let sink = Arc::new(RecordingOutputSink::default());
    let _reset_guard = OutputSinkResetGuard;
    set_output_sink(sink.clone());

    let service = make_service();
    let request_id = RequestId::new();
    let metadata = RequestSummaryMetadata::from_parts(
        Some("healthz"),
        None,
        Some("openai_frontend"),
        Some("healthz"),
    )
    .with_source(Some("direct_http"))
    .with_method(Some("GET"));
    let (guard, _) = service.register_request_with_metadata(request_id, metadata);

    let admitted = sink
        .take_events()
        .into_iter()
        .find_map(|event| match event {
            OutputEvent::CanonicalLog(envelope)
                if envelope.request_id == request_id
                    && matches!(envelope.event, LifecycleEvent::Admitted { .. }) =>
            {
                Some(envelope)
            }
            _ => None,
        })
        .expect("admitted projection");
    assert_eq!(
        admitted.presentation_message(),
        "probe admitted route=healthz source=direct_http provider=openai_frontend engine=healthz method=GET"
    );
    assert!(matches!(
        admitted.event,
        LifecycleEvent::Admitted {
            model: None,
            method: Some(ref method)
        } if method == "GET"
    ));

    service
        .transition_terminal(
            request_id,
            &guard,
            TerminalOutcome::CompletedWithStatus(200),
        )
        .expect("probe terminal transition");
    let completed = sink
        .take_events()
        .into_iter()
        .find_map(|event| match event {
            OutputEvent::CanonicalLog(envelope)
                if envelope.request_id == request_id
                    && matches!(envelope.event, LifecycleEvent::Completed { .. }) =>
            {
                Some(envelope)
            }
            _ => None,
        })
        .expect("completed projection");
    assert_eq!(
        completed.presentation_message(),
        "probe request completed route=healthz source=direct_http provider=openai_frontend engine=healthz method=GET status=200"
    );
}

#[test]
#[serial_test::serial]
fn canonical_projection_classifies_management_polling_without_guessing_the_client() {
    let sink = Arc::new(RecordingOutputSink::default());
    let _reset_guard = OutputSinkResetGuard;
    set_output_sink(sink.clone());

    let service = Arc::new(make_service());
    let request_id = RequestId::new();
    let lifecycle = crate::logging::ManagementRequestLifecycle::register(
        Arc::clone(&service),
        request_id,
        "management_get_status",
    );

    let admitted = sink
        .take_events()
        .into_iter()
        .find_map(|event| match event {
            OutputEvent::CanonicalLog(envelope)
                if envelope.request_id == request_id
                    && matches!(envelope.event, LifecycleEvent::Admitted { .. }) =>
            {
                Some(envelope)
            }
            _ => None,
        })
        .expect("management admitted projection");
    assert_eq!(
        admitted.presentation_message(),
        "management admitted route=management_get_status source=direct_http provider=management_api engine=management_get_status method=GET"
    );

    lifecycle.finish_status(200);
    let completed = sink
        .take_events()
        .into_iter()
        .find_map(|event| match event {
            OutputEvent::CanonicalLog(envelope)
                if envelope.request_id == request_id
                    && matches!(envelope.event, LifecycleEvent::Completed { .. }) =>
            {
                Some(envelope)
            }
            _ => None,
        })
        .expect("management completed projection");
    assert_eq!(
        completed.presentation_message(),
        "management request completed route=management_get_status source=direct_http provider=management_api engine=management_get_status method=GET status=200"
    );
}

#[tokio::test]
async fn attempt_records_are_delivered_under_the_parent_request() {
    let sink = Arc::new(TestSink::new());
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    let (guard, admitted_event_id) = svc.register_request(request_id);
    let attempt_id = svc.start_attempt(request_id, &guard);
    svc.complete_attempt(request_id, attempt_id, Some(200));
    svc.transition_terminal(request_id, &guard, TerminalOutcome::Completed)
        .unwrap();

    assert_eq!(svc.pump_sync().await, 5);
    let events = sink
        .records()
        .into_iter()
        .filter_map(|record| match record {
            TestRecord::Event {
                request_id: recorded_request_id,
                event_id,
                payload_json,
                ..
            } => {
                assert_eq!(recorded_request_id, request_id.as_uuid().to_string());
                let envelope =
                    mesh_llm_events::logging::envelope::CanonicalEnvelope::from_json_str(
                        &payload_json,
                    )
                    .expect("canonical lifecycle envelope");
                assert_eq!(event_id, envelope.event_id.as_uuid().to_string());
                Some((event_id, envelope.event))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(events[0].0, admitted_event_id.as_uuid().to_string());
    assert_eq!(
        events
            .into_iter()
            .map(|(_, event)| event)
            .collect::<Vec<_>>(),
        vec![
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
            LifecycleEvent::AttemptStarted {
                attempt_id: Some(attempt_id),
            },
            LifecycleEvent::AttemptCompleted {
                attempt_id: Some(attempt_id),
                status_code: Some(200),
            },
            LifecycleEvent::Completed {
                status_code: None,
                duration_ms: None,
                usage: None,
            },
        ]
    );
    assert_eq!(
        sink.records()
            .iter()
            .filter(|record| matches!(record, TestRecord::Summary(_)))
            .count(),
        1,
        "the parent request owns one summary despite multiple attempts"
    );
}

/// A freshly-registered request has no truthful metadata yet: `register_request`
/// stores `RequestSummaryMetadata::default()` in the active registry, and the
/// admitted event's own registry-entry fallback must not treat that empty
/// placeholder as real metadata. Doing so previously fabricated a presentation
/// context whose absent route/model/method still stamped `kind=unknown` onto
/// every message for the request, exactly like the two other empty-metadata
/// fallbacks already guard against.
///
/// `CanonicalEnvelope::presentation_context` is deliberately skipped by serde
/// (durable/replay truth is the canonical event plus summary metadata), so
/// this reads the live replay-bus entry's rendered `presentation_summary`
/// text directly rather than round-tripping the envelope through JSON, which
/// would silently strip the very field this test guards.
#[tokio::test]
async fn newly_registered_request_with_no_metadata_omits_presentation_context() {
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(TestSink::new()),
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    let (_guard, admitted_event_id) = svc.register_request(request_id);

    let window = svc.bus_ref().replay_window();
    let live_payload: serde_json::Value = window
        .records
        .iter()
        .find_map(|record| {
            let payload: serde_json::Value =
                serde_json::from_str(&record.entry.payload).expect("parse live bus payload");
            (payload.get("event_id").and_then(serde_json::Value::as_str)
                == Some(&admitted_event_id.as_uuid().to_string()))
            .then_some(payload)
        })
        .expect("admitted event reached the replay bus");

    let presentation_summary = live_payload
        .get("presentation_summary")
        .and_then(serde_json::Value::as_str)
        .expect("canonical events carry a rendered presentation summary");
    assert!(
        !presentation_summary.contains("kind=unknown"),
        "empty registry metadata must not stamp kind=unknown onto every message for the request: {presentation_summary}"
    );
}

fn proxy_record(request_id: RequestId, attempt_id: AttemptId) -> ProxyRecord {
    ProxyRecord {
        attempt_id,
        request_id,
        target: "remote".to_string(),
        provider: Some("openai_frontend".to_string()),
        engine: Some("responses".to_string()),
        started_at: "2026-08-04T12:00:00Z".to_string(),
        completed_at: Some("2026-08-04T12:00:01Z".to_string()),
        status_code: Some(502),
        error: Some("timeout".to_string()),
    }
}

#[tokio::test]
async fn proxy_record_delivery_is_bounded_and_does_not_own_a_terminal() {
    let sink = Arc::new(TestSink::new());
    let service = LoggingService::new(
        ServiceConfig::default(),
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    let attempt_id = AttemptId::new();

    assert!(
        service
            .enqueue_proxy_record(proxy_record(request_id, attempt_id))
            .is_ok()
    );
    assert_eq!(service.pump_sync().await, 1);
    assert_eq!(service.registry_ref().active_count(), 0);
    assert_eq!(service.registry_ref().recent_count(), 0);
    assert_eq!(service.bus_ref().len(), 0);

    let records = sink.records();
    let proxy_json = records
        .into_iter()
        .find_map(|record| match record {
            TestRecord::ProxyRecord(proxy_json) => Some(proxy_json),
            _ => None,
        })
        .expect("one proxy record persists");
    let persisted: ProxyRecord = serde_json::from_str(&proxy_json).expect("bounded proxy JSON");
    assert_eq!(persisted.attempt_id, attempt_id);
    assert_eq!(persisted.request_id, request_id);
    assert_eq!(persisted.target, "remote");
    assert_eq!(persisted.provider.as_deref(), Some("openai_frontend"));
    assert_eq!(persisted.engine.as_deref(), Some("responses"));
    assert_eq!(persisted.error.as_deref(), Some("timeout"));
    assert_eq!(persisted.status_code, Some(502));
    assert_eq!(persisted.started_at, "2026-08-04T12:00:00Z");
    assert_eq!(
        persisted.completed_at.as_deref(),
        Some("2026-08-04T12:00:01Z")
    );
}

#[tokio::test]
async fn proxy_record_drops_unknown_target_and_strips_untrusted_metadata() {
    let sink = Arc::new(TestSink::new());
    let service = LoggingService::new(
        ServiceConfig::default(),
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    let attempt_id = AttemptId::new();
    let rejected = ProxyRecord {
        target: "https://host.invalid/path".to_string(),
        ..proxy_record(request_id, attempt_id)
    };

    assert!(service.enqueue_proxy_record(rejected).is_ok());
    assert_eq!(service.persistence_queue_drops(), 1);
    assert_eq!(service.pump_sync().await, 0);
    assert!(sink.records().is_empty());

    let stripped = ProxyRecord {
        provider: Some("untrusted_provider".to_string()),
        engine: Some("untrusted_engine".to_string()),
        error: Some("untrusted_error_text".to_string()),
        ..proxy_record(request_id, AttemptId::new())
    };
    assert!(service.enqueue_proxy_record(stripped).is_ok());
    assert_eq!(service.pump_sync().await, 1);
    let proxy_json = sink
        .records()
        .into_iter()
        .find_map(|record| match record {
            TestRecord::ProxyRecord(proxy_json) => Some(proxy_json),
            _ => None,
        })
        .expect("sanitized proxy record persists");
    assert!(!proxy_json.contains("untrusted_provider"));
    assert!(!proxy_json.contains("untrusted_engine"));
    assert!(!proxy_json.contains("untrusted_error_text"));
    let persisted: ProxyRecord = serde_json::from_str(&proxy_json).expect("sanitized proxy JSON");
    assert_eq!(persisted.target, "remote");
    assert!(persisted.provider.is_none());
    assert!(persisted.engine.is_none());
    assert!(persisted.error.is_none());
}

#[tokio::test]
async fn proxy_record_queue_saturation_stays_fail_open() {
    let sink = Arc::new(TestSink::new());
    let service = LoggingService::new(
        ServiceConfig {
            queue_capacity: 64,
            ..ServiceConfig::default()
        },
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();

    for _ in 0..65 {
        assert!(
            service
                .enqueue_proxy_record(proxy_record(request_id, AttemptId::new()))
                .is_ok()
        );
    }
    assert_eq!(service.persistence_queue_drops(), 1);
    assert_eq!(service.pump_sync().await, 64);
    assert_eq!(
        sink.records()
            .iter()
            .filter(|record| matches!(record, TestRecord::ProxyRecord(_)))
            .count(),
        64
    );
}

#[tokio::test]
async fn proxy_record_sink_failure_stays_fail_open() {
    let (sink, mut attempts) = TestSink::failing_with_attempt_notifications();
    let service = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    assert!(service.spawn());

    assert!(
        service
            .enqueue_proxy_record(proxy_record(RequestId::new(), AttemptId::new()))
            .is_ok()
    );
    attempts
        .recv()
        .await
        .expect("proxy persistence is attempted");
    attempts
        .recv()
        .await
        .expect("fallback audit persistence is attempted");
    assert_eq!(service.persistence_failures(), 2);
    assert_eq!(service.persistence_queue_drops(), 0);
    assert!(service.shutdown().await);
}

#[tokio::test]
async fn worker_delivery_preserves_admission_when_terminal_uses_the_priority_lane() {
    let sink = Arc::new(TestSink::new());
    let service = LoggingService::new(
        ServiceConfig::default(),
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(TestClock::new()),
    );
    assert!(service.spawn());

    let request_id = RequestId::new();
    let (guard, admitted_event_id) = service.register_request(request_id);
    service
        .transition_terminal(request_id, &guard, TerminalOutcome::Completed)
        .expect("one terminal outcome");
    assert!(service.shutdown().await);

    let records = sink.records();
    assert!(matches!(records[0], TestRecord::Summary(_)));
    let event_records = records
        .into_iter()
        .filter_map(|record| match record {
            TestRecord::Event {
                event_id,
                payload_json,
                ..
            } => {
                let envelope =
                    mesh_llm_events::logging::envelope::CanonicalEnvelope::from_json_str(
                        &payload_json,
                    )
                    .expect("canonical lifecycle envelope");
                Some((event_id, envelope.event))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(event_records.len(), 2);
    assert!(event_records.iter().any(|(event_id, event)| {
        event_id == &admitted_event_id.as_uuid().to_string()
            && matches!(event, LifecycleEvent::Admitted { .. })
    }));
    assert!(
        event_records
            .iter()
            .any(|(_, event)| matches!(event, LifecycleEvent::Completed { .. }))
    );
}

#[tokio::test]
async fn terminal_after_its_evicted_summary_is_durably_recovered() {
    let root = tempfile::tempdir().expect("temporary log-store root");
    let store = Arc::new(
        mesh_llm_log_store::LogStore::open(root.path(), Arc::new(mesh_llm_log_store::RealClock))
            .expect("open log store"),
    );
    let service = LoggingService::new(
        ServiceConfig {
            queue_capacity: 64,
            event_buffer_size: 64,
            ..ServiceConfig::default()
        },
        Arc::new(crate::logging::LogStoreSink::new(store)),
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    let (guard, _) = service.register_request(request_id);

    // The bounded persistence queue intentionally evicts the summary and then
    // its admitted event. The terminal carries the authoritative summary so
    // its durable write cannot be stranded by those observational evictions.
    for index in 0..63 {
        assert!(service.write_error_audit(format!("fill-{index}")));
    }
    service
        .transition_terminal(request_id, &guard, TerminalOutcome::Completed)
        .expect("one terminal outcome");

    assert_eq!(service.pump_sync().await, 64);
    assert_eq!(
        service.persistence_queue_drops(),
        2,
        "the evicted summary and admitted record are counted"
    );
    assert_eq!(service.persistence_failures(), 0);
}

#[tokio::test]
async fn durable_sink_reopens_one_summary_with_ordered_retry_events() {
    let root = tempfile::tempdir().expect("temporary log-store root");
    let store = Arc::new(
        mesh_llm_log_store::LogStore::open(root.path(), Arc::new(mesh_llm_log_store::RealClock))
            .expect("open log store"),
    );
    let service = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(crate::logging::LogStoreSink::new(Arc::clone(&store))),
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    let (guard, admitted_event_id) = service.register_request(request_id);
    let failed_attempt = service.start_attempt(request_id, &guard);
    service.fail_attempt(request_id, failed_attempt, "upstream timeout".into());
    let successful_attempt = service.start_attempt(request_id, &guard);
    service.complete_attempt(request_id, successful_attempt, Some(200));
    service
        .transition_terminal(request_id, &guard, TerminalOutcome::Completed)
        .expect("one terminal outcome");

    assert_eq!(service.pump_sync().await, 7);

    let request_key = request_id.as_uuid().to_string();
    let summary = store
        .get_summary(&request_key)
        .expect("query summary")
        .expect("durable summary");
    assert_eq!(summary.state, "completed");
    assert!(summary.terminal_at.is_some());

    let lifecycle_rows = store
        .list_lifecycle_events(10, None)
        .expect("query lifecycle events");
    assert_eq!(lifecycle_rows.items.len(), 6);
    assert!(
        lifecycle_rows
            .items
            .iter()
            .all(|row| row.request_id == request_key)
    );
    assert!(lifecycle_rows.items.iter().any(|row| {
        row.event_id == admitted_event_id.as_uuid().to_string() && row.request_id == request_key
    }));

    let payloads = {
        let connection = store.conn();
        let mut statement = connection
            .prepare(
                "SELECT payload_json FROM lifecycle_events \
                 WHERE request_id = ? ORDER BY occurred_at ASC, event_id ASC",
            )
            .expect("prepare lifecycle query");
        statement
            .query_map([&request_key], |row| row.get::<_, String>(0))
            .expect("query lifecycle payloads")
            .collect::<Result<Vec<_>, _>>()
            .expect("read lifecycle payloads")
    };
    let envelopes = payloads
        .iter()
        .map(|payload| {
            mesh_llm_events::logging::envelope::CanonicalEnvelope::from_json_str(payload)
                .expect("stored canonical envelope")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        envelopes
            .iter()
            .map(|envelope| envelope.sequence)
            .collect::<Vec<_>>(),
        vec![1, 1, 2, 3, 4, 2],
        "request and operations channels preserve their independent sequences"
    );
    assert_eq!(
        envelopes
            .iter()
            .map(|envelope| envelope.event.clone())
            .collect::<Vec<_>>(),
        vec![
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
            LifecycleEvent::AttemptStarted {
                attempt_id: Some(failed_attempt),
            },
            LifecycleEvent::AttemptFailed {
                attempt_id: Some(failed_attempt),
                error: Some("upstream timeout".into()),
            },
            LifecycleEvent::AttemptStarted {
                attempt_id: Some(successful_attempt),
            },
            LifecycleEvent::AttemptCompleted {
                attempt_id: Some(successful_attempt),
                status_code: Some(200),
            },
            LifecycleEvent::Completed {
                status_code: None,
                duration_ms: None,
                usage: None,
            },
        ]
    );
    assert!(
        envelopes
            .iter()
            .all(|envelope| envelope.event_id.as_uuid() != request_id.as_uuid())
    );
    assert_eq!(
        envelopes[0].event_id, admitted_event_id,
        "register_request returns the canonical admitted envelope ID"
    );

    let reopened = mesh_llm_log_store::LogStore::reopen_at(
        root.path(),
        Arc::new(mesh_llm_log_store::RealClock),
    )
    .expect("reopen durable log store");
    assert_eq!(
        reopened
            .get_summary(&request_key)
            .expect("query reopened summary")
            .expect("reopened durable summary")
            .state,
        "completed"
    );
    assert_eq!(
        reopened
            .list_lifecycle_events(10, None)
            .expect("query reopened lifecycle events")
            .items
            .len(),
        6
    );
}

#[tokio::test]
async fn durable_sink_persists_a_dropped_terminal_with_its_summary() {
    let root = tempfile::tempdir().expect("temporary log-store root");
    let store = Arc::new(
        mesh_llm_log_store::LogStore::open(root.path(), Arc::new(mesh_llm_log_store::RealClock))
            .expect("open log store"),
    );
    let service = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(crate::logging::LogStoreSink::new(Arc::clone(&store))),
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    let (guard, _) = service.register_request(request_id);

    drop(guard);
    assert_eq!(service.pump_sync().await, 3);

    let request_key = request_id.as_uuid().to_string();
    assert_eq!(
        store
            .get_summary(&request_key)
            .expect("query summary")
            .expect("durable summary")
            .state,
        "dropped"
    );
    assert_eq!(
        store
            .list_lifecycle_events(10, None)
            .expect("query dropped lifecycle event")
            .items
            .len(),
        2
    );
}

#[tokio::test]
async fn test_bounded_shutdown() {
    let svc = make_service();

    // Register some requests and enqueue events.
    for i in 0..10 {
        let rid = RequestId::new();
        let _guard = svc.register_request(rid);
        let payload = serde_json::json!({ "i": i }).to_string();
        svc.enqueue_event(rid, ReplayChannel::Requests, payload)
            .unwrap();
    }

    // Drain the bus before shutdown.
    let drained = svc.pump_sync().await;
    assert!(drained > 0);

    // Shutdown (without spawn) freezes the manual delivery mode so no later
    // hand-off can be parked after the bounded drain boundary.
    let result = svc.shutdown().await;
    assert!(result);

    // Second shutdown is a no-op (restart-safe).
    let second_result = svc.shutdown().await;
    assert!(!second_result);

    // Registry should still have entries (shutdown doesn't clear them — that's explicit via registry.clear()).
}

#[test]
#[allow(clippy::await_holding_lock)] // test-only: safe since single-threaded runtime context
fn test_spawn_then_shutdown() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        let sink = Arc::new(TestSink::new());
        let svc = Arc::new(std::sync::Mutex::new(LoggingService::new(
            ServiceConfig::default(),
            sink,
            Box::<SystemClock>::default(),
        )));

        // spawn() on blocking thread to avoid deadlock with tokio runtime.
        let first_spawned: bool = tokio::task::spawn_blocking({
            let s = Arc::clone(&svc);
            move || {
                let inner = s.lock().unwrap();
                inner.spawn()
            }
        })
        .await
        .unwrap();

        assert!(first_spawned, "first spawn should return true");

        // Second spawn is a no-op.
        let second_spawned: bool = tokio::task::spawn_blocking({
            let s = Arc::clone(&svc);
            move || {
                let inner = s.lock().unwrap();
                inner.spawn()
            }
        })
        .await
        .unwrap();

        assert!(!second_spawned, "second spawn should return false");

        // Shutdown drains + stops → returns true.
        {
            let inner = svc.lock().unwrap();
            let result = inner.shutdown().await;
            assert!(result, "first shutdown should succeed");
        }

        // Second shutdown after first completes → false (already stopped, restart-safe).
        {
            let inner = svc.lock().unwrap();
            let second_result = inner.shutdown().await;
            assert!(!second_result, "second shutdown should be no-op");
        }
    });
}

#[test]
#[allow(clippy::await_holding_lock)] // test-only: safe since single-threaded runtime context
fn test_restart_safe_shutdown() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        let sink = Arc::new(TestSink::new());
        let config = ServiceConfig::default();

        // Create, spawn, shutdown in scope.
        {
            let svc1 = Arc::new(std::sync::Mutex::new(LoggingService::new(
                config.clone(),
                sink.clone(),
                Box::<SystemClock>::default(),
            )));

            let spawned: bool = tokio::task::spawn_blocking({
                let s = Arc::clone(&svc1);
                move || {
                    let inner = s.lock().unwrap();
                    inner.spawn()
                }
            })
            .await
            .unwrap();

            assert!(spawned, "first spawn should succeed");

            // Shutdown.
            {
                let inner = svc1.lock().unwrap();
                let result = inner.shutdown().await;
                assert!(result, "shutdown should succeed");
            }
        } // Drop the service — worker task should clean up.

        // Re-create a new service (restart-safe: old one dropped).
        let svc2 = LoggingService::new(config, sink, Box::<SystemClock>::default());
        assert!(!svc2.is_spawned(), "fresh service should not be spawned");
    });
}

// ---------------------------------------------------------------------------
// Test Scenario 8a: Request-path completion despite full queue (enqueue on full returns Ok with drop-oldest)
// ---------------------------------------------------------------------------

#[test]
fn test_request_path_completion_despite_full_queue() {
    let config = ServiceConfig {
        queue_capacity: 5, // Very small to force overflow quickly.
        event_buffer_size: 5,
        artifact_command_max_bytes: ServiceConfig::default().artifact_command_max_bytes,
        registry_config: RegistryConfig::default(),
        ..ServiceConfig::default()
    };

    let svc = LoggingService::new(
        config,
        Arc::new(TestSink::new()),
        Box::new(TestClock::new()),
    );

    let rid = RequestId::new();
    let _guard = svc.register_request(rid);

    // Fill the queue completely.
    for i in 0..5 {
        assert!(
            svc.enqueue_event(rid, ReplayChannel::Requests, format!("event_{}", i))
                .is_ok()
        );
    }

    assert_eq!(svc.bus_ref().len(), 5);

    // Now enqueue more — should succeed (drop-oldest applies internally), never blocking.
    for i in 0..100 {
        let result = svc.enqueue_event(rid, ReplayChannel::Requests, format!("overflow_{}", i));
        assert!(result.is_ok(), "enqueue must always return Ok (fail-open)");

        // Bus stays at capacity.
        assert_eq!(svc.bus_ref().len(), 5);
    }

    // The admitted event already occupied one slot before the explicit fill,
    // so 101 old replay entries were evicted; no incoming event was rejected.
    let evictions = svc.bus_ref().evictions.load(AtomicOrdering::Relaxed);
    assert_eq!(evictions, 101);
    assert_eq!(svc.bus_ref().drops.load(AtomicOrdering::Relaxed), 0);
    assert_eq!(svc.total_drops(), 0);

    // Request path completes — no panic, no deadlock.
}

#[test]
fn test_zero_capacity_service_records_rejection_without_eviction() {
    let config = ServiceConfig {
        queue_capacity: 0,
        event_buffer_size: 0,
        artifact_command_max_bytes: ServiceConfig::default().artifact_command_max_bytes,
        registry_config: RegistryConfig::default(),
        ..ServiceConfig::default()
    };
    let svc = LoggingService::new(
        config,
        Arc::new(TestSink::new()),
        Box::new(TestClock::new()),
    );

    assert!(
        svc.enqueue_event(
            RequestId::new(),
            ReplayChannel::Requests,
            "{\"event\":\"rejected\"}".into(),
        )
        .is_ok()
    );

    assert_eq!(svc.bus_ref().len(), 0);
    assert_eq!(svc.bus_ref().drops.load(AtomicOrdering::Relaxed), 1);
    assert_eq!(svc.bus_ref().evictions.load(AtomicOrdering::Relaxed), 0);
    assert_eq!(svc.total_drops(), 1);
}

#[tokio::test]
async fn test_zero_queue_capacity_is_clamped_for_manual_delivery() {
    let sink = Arc::new(TestSink::new());
    let svc = LoggingService::new_with_dynamic_limits(
        ServiceConfig {
            queue_capacity: 0,
            event_buffer_size: 1,
            artifact_command_max_bytes: ServiceConfig::default().artifact_command_max_bytes,
            registry_config: RegistryConfig::default(),
            ..ServiceConfig::default()
        },
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(TestClock::new()),
        LoggingDynamicLimits {
            retention_ttl_secs: 60,
            replay_capacity: 1,
        },
    );

    svc.enqueue_event(
        RequestId::new(),
        ReplayChannel::Requests,
        "{\"event\":\"accepted\"}".into(),
    )
    .expect("replay-enabled service accepts the entry");

    assert_eq!(svc.persistence_queue_drops(), 0);
    assert_eq!(svc.persistence_outstanding(), 1);
    assert_eq!(svc.pump_sync().await, 1);
    assert_eq!(svc.persistence_outstanding(), 0);
    assert_eq!(sink.records().len(), 1, "accepted entry is persisted once");
}

#[tokio::test]
async fn test_zero_capacity_service_can_spawn_without_panicking() {
    let svc = LoggingService::new(
        ServiceConfig {
            queue_capacity: 0,
            event_buffer_size: 0,
            artifact_command_max_bytes: ServiceConfig::default().artifact_command_max_bytes,
            registry_config: RegistryConfig::default(),
            ..ServiceConfig::default()
        },
        Arc::new(TestSink::new()),
        Box::new(TestClock::new()),
    );

    assert!(svc.spawn());
    assert!(svc.shutdown().await);
}

// ---------------------------------------------------------------------------
// Test Scenario 8b: Request-path completion despite store worker failure (sink returns Err)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_request_path_completion_despite_sink_failure() {
    use crate::logging::lifecycle::TerminalOutcome;

    let sink = Arc::new(TestSink::new());
    let svc = LoggingService::new(
        ServiceConfig::default(),
        sink.clone(),
        Box::new(TestClock::new()),
    );

    // Make the sink start failing.
    sink.set_failing();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    // Enqueue should still succeed — fail-open writer absorbs the sink error.
    for i in 0..10 {
        let result = svc.enqueue_event(rid, ReplayChannel::Requests, format!("fail_{}", i));
        assert!(
            result.is_ok(),
            "enqueue must return Ok even when sink fails"
        );
    }

    // Transition terminal — should still work despite failing sink.
    let result = svc.transition_terminal(rid, &guard, TerminalOutcome::Completed);
    assert!(result.is_ok());

    // Request path completes without panic or deadlock.
}

// ---------------------------------------------------------------------------
// Persistence delivery: a replay entry has one delivery owner, independent of
// the replay window. These use channels rather than sleeps so worker progress
// is observed directly.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_spawned_worker_persists_each_accepted_entry_once_in_order() {
    let (sink, mut persisted) = TestSink::recording();
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();

    assert!(svc.spawn());
    let payloads = ["first", "second", "third"];
    for payload in payloads {
        svc.enqueue_event(
            request_id,
            ReplayChannel::Requests,
            serde_json::json!({ "event": payload }).to_string(),
        )
        .unwrap();
    }

    // A worker owns delivery once spawned, so a manual pump cannot duplicate it.
    assert_eq!(svc.pump_sync().await, 0);
    assert_eq!(svc.bus_ref().len(), payloads.len());

    for (index, expected_payload) in payloads.iter().enumerate() {
        let (level, message) = persisted.recv().await.expect("worker record");
        assert_eq!(level, "info");
        let envelope: serde_json::Value = serde_json::from_str(&message).unwrap();
        assert_eq!(envelope["request_id"], request_id.as_uuid().to_string());
        assert_eq!(envelope["sequence"], (index + 1) as u64);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(envelope["payload"].as_str().unwrap())
                .unwrap()["event"],
            *expected_payload
        );
    }
    assert!(
        persisted.try_recv().is_err(),
        "each accepted entry must be persisted exactly once"
    );
    assert_eq!(svc.persistence_queue_drops(), 0);
    assert_eq!(svc.persistence_failures(), 0);
    assert!(svc.shutdown().await);
}

#[tokio::test]
async fn test_manual_pump_persists_exact_entry_without_consuming_replay() {
    let (sink, mut persisted) = TestSink::recording();
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    let payload = serde_json::json!({ "event": "manual" }).to_string();

    svc.enqueue_event(request_id, ReplayChannel::Operations, payload.clone())
        .unwrap();
    let expected = serde_json::json!({
        "request_id": request_id.as_uuid(),
        "channel": ReplayChannel::Operations,
        "sequence": 1,
        "occurred_at": "2025-01-01T00:00:00.000000000Z",
        "payload": payload,
    })
    .to_string();

    assert_eq!(svc.pump_sync().await, 1);
    let (level, message) = persisted.recv().await.expect("manual pump record");
    assert_eq!(level, "info");
    assert_eq!(message, expected, "sink receives the exact bus entry");
    assert_eq!(svc.bus_ref().len(), 1, "persistence must not drain replay");
    assert_eq!(svc.pump_sync().await, 0, "manual entry is not redelivered");
    assert!(persisted.try_recv().is_err());
}

#[tokio::test]
async fn manual_shutdown_freezes_drains_pre_spawn_entries_and_is_restart_safe() {
    let (sink, mut persisted) = TestSink::recording();
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    for event in ["one", "two"] {
        svc.enqueue_event(
            request_id,
            ReplayChannel::System,
            serde_json::json!({ "event": event }).to_string(),
        )
        .unwrap();
    }
    assert_eq!(svc.persistence_outstanding(), 2);

    assert!(
        svc.shutdown().await,
        "manual ownership must be shutdown-capable"
    );
    for expected in ["one", "two"] {
        let (_, message) = persisted.recv().await.expect("manual shutdown delivery");
        assert!(message.contains(expected));
    }
    assert_eq!(svc.persistence_outstanding(), 0);
    assert_eq!(svc.persistence_shutdown_losses(), 0);

    // The freeze is final for the old owner: post-freeze work reaches replay
    // but is accurately accounted as an unavailable persistence hand-off.
    svc.enqueue_event(
        request_id,
        ReplayChannel::System,
        "{\"event\":\"after-freeze\"}".into(),
    )
    .unwrap();
    assert_eq!(svc.persistence_queue_drops(), 1);
    assert_eq!(svc.persistence_outstanding(), 0);

    assert!(svc.spawn(), "a stopped service starts a new worker safely");
    svc.enqueue_event(
        request_id,
        ReplayChannel::System,
        "{\"event\":\"after-restart\"}".into(),
    )
    .unwrap();
    let (_, message) = persisted.recv().await.expect("restarted worker delivery");
    assert!(message.contains("after-restart"));
    assert!(svc.shutdown().await);
}

#[tokio::test]
async fn manual_pump_shutdown_race_aborts_once_without_outstanding_underflow() {
    let (sink, mut started, mut completed, _release) = BlockingAuditSink::new();
    let svc = Arc::new(
        LoggingService::new(
            ServiceConfig::default(),
            Arc::new(sink),
            Box::new(TestClock::new()),
        )
        .with_shutdown_drain_timeout(Duration::ZERO),
    );

    assert!(svc.write_error_audit("manual-stall".into()));
    let pump = {
        let svc = Arc::clone(&svc);
        tokio::spawn(async move { svc.pump_sync().await })
    };
    started.recv().await.expect("manual pump entered sink");
    assert_eq!(svc.persistence_outstanding(), 1);
    assert_eq!(
        svc.pump_sync().await,
        0,
        "second pump cannot duplicate ownership"
    );

    assert!(svc.shutdown().await);
    assert_eq!(svc.persistence_outstanding(), 0);
    assert_eq!(svc.persistence_shutdown_losses(), 1);
    assert_eq!(pump.await.expect("pump task joins after abort"), 0);
    assert!(completed.try_recv().is_err());

    // A stale cancelled pump must never decrement a new owner below zero.
    assert!(svc.spawn());
    assert!(svc.write_error_audit("after-manual-abort".into()));
    let message = completed.recv().await.expect("fresh worker audit delivery");
    assert!(message.contains("after-manual-abort"));
    assert_eq!(svc.persistence_outstanding(), 0);
    assert!(svc.shutdown().await);
}

#[tokio::test]
async fn error_audit_uses_canonical_system_delivery_before_and_after_spawn() {
    let (sink, mut persisted) = TestSink::recording();
    let svc = Arc::new(LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    ));

    assert!(svc.write_error_audit("before-worker".into()));
    assert_eq!(svc.pump_sync().await, 1);
    let (_, before) = persisted.recv().await.expect("manual audit persistence");
    let before: mesh_llm_events::logging::envelope::CanonicalEnvelope =
        serde_json::from_str(&before).expect("canonical audit envelope");
    assert_eq!(before.channel, ReplayChannel::System);
    assert_eq!(before.sequence, 1);
    assert!(
        matches!(before.event, LifecycleEvent::AuditError { ref message } if message == "before-worker")
    );

    assert!(svc.spawn());
    assert!(svc.write_error_audit("after-worker".into()));
    let (_, after) = persisted.recv().await.expect("worker audit persistence");
    let after: mesh_llm_events::logging::envelope::CanonicalEnvelope =
        serde_json::from_str(&after).expect("canonical audit envelope");
    assert_eq!(after.channel, ReplayChannel::System);
    assert_eq!(after.sequence, 2);
    assert!(
        matches!(after.event, LifecycleEvent::AuditError { ref message } if message == "after-worker")
    );

    let nested_service = Arc::clone(&svc);
    assert!(svc.writer_ref().try_record_error(move || {
        assert!(!nested_service.write_error_audit("suppressed".into()));
    }));
    assert_eq!(
        svc.writer_ref()
            .fallback_guard_suppressions
            .load(AtomicOrdering::Relaxed),
        1
    );
    assert_eq!(
        svc.writer_ref()
            .fallback_persistence_suppressions
            .load(AtomicOrdering::Relaxed),
        0
    );
    assert!(
        persisted.try_recv().is_err(),
        "recursion emits no second audit"
    );
    assert!(svc.shutdown().await);
}

#[tokio::test]
async fn noncanonical_lifecycle_payload_is_sanitized_before_operational_audit_persistence() {
    let sink = Arc::new(TestSink::new());
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(TestClock::new()),
    );
    let home = dirs::home_dir().expect("test runner has a home directory");
    let raw_payload = format!(
        r#"{{"type":"unknown_lifecycle","authorization":"Bearer sk-test-secret","path":"{}"}}"#,
        home.join("private/request.json").display()
    );

    assert!(
        svc.enqueue_event(RequestId::new(), ReplayChannel::System, raw_payload)
            .is_ok(),
        "untrusted payloads must remain fail-open"
    );
    assert_eq!(svc.pump_sync().await, 1);

    let records = sink.records();
    assert!(matches!(
        records.as_slice(),
        [TestRecord::AuditEntry { level, message, .. }]
            if level == "info"
                && message.contains("[REDACTED]")
                && !message.contains("sk-test-secret")
                && !message.contains(&home.display().to_string())
    ));
}

#[tokio::test]
async fn test_worker_channel_saturation_is_nonblocking_and_counted_once() {
    let (sink, mut started, mut completed, release) = BlockingAuditSink::new();
    let svc = LoggingService::new(
        ServiceConfig {
            queue_capacity: 64,
            ..ServiceConfig::default()
        },
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    let request_id = RequestId::new();
    assert!(svc.spawn());

    // Hold the worker on its first entry, then fill its 64-slot hand-off.
    svc.enqueue_event(request_id, ReplayChannel::System, "{\"event\":0}".into())
        .unwrap();
    started
        .recv()
        .await
        .expect("worker started first persistence");
    for index in 1..=65 {
        assert!(
            svc.enqueue_event(
                request_id,
                ReplayChannel::System,
                format!("{{\"event\":{index}}}"),
            )
            .is_ok(),
            "request path must remain fail-open when worker queue is full"
        );
    }
    assert_eq!(svc.persistence_queue_drops(), 1);
    assert_eq!(svc.total_drops(), 0, "worker delivery drops are separate");
    assert_eq!(svc.bus_ref().drops.load(AtomicOrdering::Relaxed), 0);

    release.notify_one();
    for _ in 0..65 {
        completed
            .recv()
            .await
            .expect("accepted persistence completion");
    }
    assert!(completed.try_recv().is_err());
    assert!(svc.shutdown().await);
}

#[tokio::test]
async fn terminal_handoff_survives_normal_worker_queue_saturation() {
    let (sink, mut started, _completed, release) = BlockingAuditSink::new();
    let svc = LoggingService::new(
        ServiceConfig {
            queue_capacity: 64,
            ..ServiceConfig::default()
        },
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    assert!(svc.spawn());

    // Block the worker, then fill its normal lane. Registering before the
    // saturation gives the terminal a real lifecycle parent while the test
    // verifies that only the extra normal entry is rejected.
    svc.enqueue_event(
        RequestId::new(),
        ReplayChannel::System,
        "{\"event\":0}".into(),
    )
    .expect("first normal entry is accepted");
    started
        .recv()
        .await
        .expect("worker started first normal persistence");
    let request_id = RequestId::new();
    let (guard, _) = svc.register_request(request_id);
    for index in 1..=62 {
        svc.enqueue_event(
            RequestId::new(),
            ReplayChannel::System,
            format!("{{\"event\":{index}}}"),
        )
        .expect("normal queue fill remains fail-open");
    }
    svc.enqueue_event(
        RequestId::new(),
        ReplayChannel::System,
        "{\"event\":65}".into(),
    )
    .expect("overflowing request path remains fail-open");
    assert_eq!(svc.persistence_queue_drops(), 1);

    svc.transition_terminal(request_id, &guard, TerminalOutcome::Completed)
        .expect("terminal transition is accepted on the reserved lane");
    assert_eq!(
        svc.persistence_queue_drops(),
        1,
        "a terminal transition is not dropped behind saturated normal traffic"
    );

    release.notify_one();
    assert!(svc.shutdown().await);
    assert_eq!(svc.persistence_outstanding(), 0);
}

#[tokio::test]
async fn test_sink_failure_is_counted_without_changing_enqueue_success() {
    let (sink, mut attempts) = TestSink::failing_with_attempt_notifications();
    let svc = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestClock::new()),
    );
    assert!(svc.spawn());

    assert!(
        svc.enqueue_event(
            RequestId::new(),
            ReplayChannel::Requests,
            "{\"event\":\"failing\"}".into(),
        )
        .is_ok()
    );
    attempts
        .recv()
        .await
        .expect("original sink persistence attempt");
    attempts
        .recv()
        .await
        .expect("fallback sink persistence attempt");
    assert_eq!(svc.persistence_failures(), 2);
    assert_eq!(svc.persistence_queue_drops(), 0);
    assert_eq!(svc.total_drops(), 0);
    assert!(svc.shutdown().await);
}

#[test]
fn test_writer_fail_open_no_panic_on_sink_error() {
    let sink = Arc::new(TestSink::new());
    sink.set_failing();

    let svc = LoggingService::new(ServiceConfig::default(), sink, Box::new(TestClock::new()));

    // Error audit write should not panic even when the underlying operations fail.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for i in 0..100 {
            svc.write_error_audit(format!("error_{}", i));
        }
    }));

    assert!(result.is_ok(), "write_error_audit must never panic");
}

// ---------------------------------------------------------------------------
// Additional: Recursion guard prevents self-logging loops
// ---------------------------------------------------------------------------

#[test]
fn test_recursion_guard_blocks_nested_error_path() {
    let svc = make_service();
    let writer = svc.writer_ref();

    assert!(!writer.is_in_error_path());

    // First entry succeeds.
    assert!(writer.try_record_error(|| {}));

    // After exit, can enter again (not nested anymore).
}

#[test]
fn test_recursion_guard_depth_prevents_cross_thread_duplication() {
    let writer = FailOpenWriter::new();

    // Simulate depth guard behavior.
    assert!(writer.try_record_error(|| {}));
}

// ---------------------------------------------------------------------------
// Additional: Bus drop-oldest preserves recent entries under pressure
// ---------------------------------------------------------------------------

#[test]
fn test_drop_oldest_preserves_recent() {
    let bus = ReplayBus::new(3);

    for i in 0..10 {
        bus.push(format!("entry_{}", i));
    }

    assert_eq!(bus.len(), 3); // At capacity.

    let entries = bus.drain();
    assert_eq!(entries.len(), 3);

    // Last three entries should be preserved (indices 7, 8, 9).
    assert_eq!(entries[0].payload, "entry_7");
    assert_eq!(entries[1].payload, "entry_8");
    assert_eq!(entries[2].payload, "entry_9");

    // Oldest entries (0-6) were evicted.
}

// ---------------------------------------------------------------------------
// Audit identity sharing: live bus frame and durable row share entry_id/occurred_at
// ---------------------------------------------------------------------------

#[tokio::test]
async fn audit_enqueue_shares_identity_with_durable_row() {
    let sink = Arc::new(TestSink::new());
    let service = LoggingService::new(
        ServiceConfig::default(),
        Arc::clone(&sink) as Arc<dyn PersistSink>,
        Box::new(TestClock::new()),
    );

    let record = OperationalAuditRecord::builder("runtime", "startup_complete").build();
    assert!(service.write_operational_audit(record));

    assert_eq!(service.pump_sync().await, 1);

    let audit_records: Vec<_> = sink
        .records()
        .into_iter()
        .filter_map(|r| match r {
            TestRecord::AuditEntry {
                level,
                message,
                entry_id,
                occurred_at,
            } => Some((level, message, entry_id, occurred_at)),
            _ => None,
        })
        .collect();
    assert_eq!(audit_records.len(), 1);
    let (_, _, durable_entry_id, durable_occurred_at) = &audit_records[0];

    let bus_entries = service.bus_ref().audit_replay_window();
    assert_eq!(bus_entries.records.len(), 1);
    let live_payload: serde_json::Value =
        serde_json::from_str(&bus_entries.records[0].entry.payload).expect("parse live payload");
    let live_entry_id = live_payload
        .get("entry_id")
        .and_then(|v| v.as_str())
        .expect("entry_id");
    let live_occurred_at = live_payload
        .get("occurred_at")
        .and_then(|v| v.as_str())
        .expect("occurred_at");

    assert!(!live_entry_id.is_empty());
    assert_eq!(live_occurred_at, "2025-01-01T00:00:00.000000000Z");
    assert_eq!(
        live_payload.get("source").and_then(|v| v.as_str()),
        Some("runtime")
    );
    assert_eq!(
        live_payload.get("code").and_then(|v| v.as_str()),
        Some("startup_complete")
    );

    assert_eq!(durable_entry_id.as_deref(), Some(live_entry_id));
    assert_eq!(durable_occurred_at.as_deref(), Some(live_occurred_at));
}
