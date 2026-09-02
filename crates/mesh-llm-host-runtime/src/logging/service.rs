//! LoggingService facade owning bus + registry + lifecycle guard factory.
//!
//! The service coordinates all logging components and exposes a simple API for request-path callers.
//! Persistence delivery and canonical event projection live in the focused
//! `service` child modules; the enqueue path never blocks.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mesh_llm_events::logging::envelope::CanonicalEnvelope;
use mesh_llm_events::logging::events::LifecycleEvent;
use mesh_llm_events::logging::identifiers::{AttemptId, EventId, RequestId};
use mesh_llm_events::logging::proxy::ProxyRecord;
use mesh_llm_events::logging::replay::ReplayChannel;
use mesh_llm_events::logging::timestamp::canonical_logging_timestamp;
use tokio::task::AbortHandle;

const PERSISTENCE_SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

mod artifact_persistence;
use artifact_persistence::DEFAULT_ARTIFACT_MEMORY_BUDGET_BYTES;
pub(crate) use artifact_persistence::{ArtifactCaptureContent, ArtifactUnavailableReason};
pub use artifact_persistence::{ArtifactCaptureEntry, ArtifactPersistenceStatus};

mod event_delivery;
mod operational_audit;
mod persistence_delivery;
use event_delivery::{EventDelivery, ServiceLifecycleRecorder, enqueue_event_with_delivery};
pub use operational_audit::{
    OperationalAuditContext, OperationalAuditPathType, OperationalAuditRecord,
    OperationalAuditRecordBuilder, OperationalAuditSeverity, OperationalAuditSubjectKind,
};
pub(crate) use persistence_delivery::PersistenceWorkerState;
pub use persistence_delivery::WorkerHandle;
use persistence_delivery::{
    DeliveryMode, PersistenceEntry, offer_persistence_to, offer_summary_persistence,
    record_persistence_queue_drop,
};

pub use super::service_errors::BusEnqueueError;

pub use super::bus::{BusEntry, ReplayBus};
use super::lifecycle::LifecycleRecorder;
pub use super::lifecycle::{DuplicateTerminalError, LifecycleGuard, TerminalOutcome};
use super::limits::{DynamicLoggingLimits, LoggingDynamicLimits};
use super::metrics::{
    LoggingArtifactCaptureStatus, LoggingCleanupOutcome, LoggingMetric, LoggingMetrics,
    LoggingMetricsSink, LoggingTerminalOutcome,
};
use super::output_projection::emit_accepted_canonical_event;
use super::policy::sanitize_lifecycle_event;
use super::registry::RequestSummaryEventSnapshots;
pub use super::registry::{RegistryConfig, RequestRegistry, RequestSummaryEntry};
use super::request_metadata::RequestSummaryMetadata;
pub use super::sequences::SequenceGenerators;
pub use super::writer::FailOpenWriter;

/// Trait for persistence sinks. The real LogStore implements this in a later todo (Todo 7+).
/// For now, tests provide a Vec-backed implementation.
#[async_trait::async_trait]
pub trait PersistSink: Send + Sync {
    /// Persist a request summary record.
    async fn persist_summary(&self, entry: RequestSummaryEntry) -> Result<(), String>;

    /// Persist a lifecycle event payload (JSON string).
    async fn persist_event(
        &self,
        request_id: String,
        event_id: String,
        channel: ReplayChannel,
        sequence: u64,
        occurred_at: String,
        payload_json: String,
    ) -> Result<(), String>;

    /// Persist an artifact pointer (metadata only; content handled by ArtifactFileStore).
    async fn persist_artifact_pointer(
        &self,
        request_id: String,
        artifact_data: serde_json::Value,
    ) -> Result<(), String>;

    /// Persist one logging-owned OpenAI artifact command. Production sinks do
    /// all redaction, filesystem, and SQLite work on their serial blocking
    /// worker. The default keeps test and external sinks source-compatible.
    async fn persist_artifact_capture(
        &self,
        _entry: ArtifactCaptureEntry,
    ) -> Result<ArtifactPersistenceStatus, String> {
        Err("artifact capture persistence is not configured".to_string())
    }

    /// Persist a proxy transport record.
    async fn persist_proxy_record(&self, proxy_json: String) -> Result<(), String>;

    /// Persist one typed static operational audit record.
    async fn persist_audit_entry(&self, record: OperationalAuditRecord) -> Result<(), String>;

    /// Persist a webhook delivery record.
    async fn persist_webhook_delivery(
        &self,
        request_id: Option<String>,
        status_code: u16,
        error: Option<String>,
    ) -> Result<(), String>;

    /// Persist a cleanup run summary.
    async fn persist_cleanup_run(&self, deleted_count: u64) -> Result<(), String>;
}

/// Clock provider for deterministic timestamps (injected by the service constructor).
pub trait Clock: Send + Sync {
    /// Return an ISO 8601 timestamp string. Tests inject a counter-based clock; production uses chrono::Utc.
    fn now(&self) -> String;
}

/// Production clock using system time.
#[derive(Clone, Debug)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> String {
        use chrono::SecondsFormat;

        chrono::Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true)
    }
}

fn canonical_clock_timestamp(clock: &dyn Clock) -> String {
    let timestamp = clock.now();
    canonical_logging_timestamp(&timestamp).unwrap_or(timestamp)
}

impl Default for SystemClock {
    fn default() -> Self {
        Self
    }
}

/// Configuration for the logging service. Derived from [`mesh_llm_config::LoggingConfig`] but simplified for runtime use.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Maximum Unicode characters in each payload-free local presentation
    /// summary line, fixed for the process lifetime.
    pub summary_line_limit: usize,
    /// Hard ceiling on entries held by the in-memory replay buffer, fixed for
    /// the process lifetime.
    pub event_buffer_size: usize,
    /// Initial and live current replay target, dynamically adjustable up to
    /// `event_buffer_size`.
    pub replay_capacity: usize,
    /// Maximum pending entries in the persistence/dispatch delivery queue.
    /// This does not control replay retention.
    pub queue_capacity: usize,
    /// Registry configuration (max_active, max_recent).
    pub registry_config: RegistryConfig,
    /// Maximum body bytes copied into one in-memory artifact command. The
    /// sink independently owns and enforces the durable policy limits.
    pub artifact_command_max_bytes: usize,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            summary_line_limit: 2_048,
            event_buffer_size: 10_000,
            replay_capacity: 128,
            queue_capacity: 4096, // matches config defaults from Todo 2.
            registry_config: RegistryConfig::default(),
            artifact_command_max_bytes: 256 * 1024,
        }
    }
}

impl ServiceConfig {
    /// Build the static service contract from the validated runtime policy.
    pub fn from_policy(policy: &super::policy::PrivacyPolicy) -> Self {
        Self {
            summary_line_limit: policy.summary_line_limit,
            event_buffer_size: policy.event_buffer_size,
            replay_capacity: policy.replay_capacity,
            queue_capacity: policy.queue_capacity,
            artifact_command_max_bytes: match policy.capture_mode {
                super::policy::PolicyCaptureMode::MetadataOnly => {
                    Self::default().artifact_command_max_bytes
                }
                super::policy::PolicyCaptureMode::RedactedWithLimits(byte_limit, _) => byte_limit,
            },
            ..Self::default()
        }
    }
}

/// The LoggingService facade coordinating all logging components.
pub struct LoggingService {
    /// Bounded replay bus for nonblocking enqueue with drop-oldest overflow policy.
    bus: Arc<ReplayBus>,

    /// Process-local optional metrics adapter. Logging owns this closed
    /// vocabulary and never depends on an OTLP implementation.
    metrics: LoggingMetrics,

    /// Sequence generators per ReplayChannel (monotonic, shared across clones).
    sequences: SequenceGenerators,

    /// Active/recent request registry.
    registry: Arc<RequestRegistry>,

    /// Fail-open writer with recursion guard for error-audit fallback.
    writer: Arc<FailOpenWriter>,

    /// Persistence sink (LogStore in production; Vec-backed in tests).
    sink: Option<Arc<dyn PersistSink>>,

    /// Clock provider for deterministic timestamps.
    clock: Arc<dyn Clock>,

    /// Worker handle for controlled shutdown of the persistence task.
    worker_handle: Mutex<Option<WorkerHandle>>,

    /// Serializes the state transitions that publish or claim a delivery
    /// owner. It closes the worker-handle publication window between spawn and
    /// shutdown and also makes manual freeze/pump ownership atomic.
    transition_lock: Mutex<()>,

    /// A running manual pump's cancellation handle. It is installed before a
    /// shutdown can observe `ManualPumping`, so bounded shutdown can always
    /// stop a stalled pump without relying on caller cancellation.
    manual_abort_handle: Arc<Mutex<Option<AbortHandle>>>,

    /// One-time delivery state, kept separate from the replay window.
    delivery: Arc<Mutex<DeliveryMode>>,

    /// Whether spawn() has been called (prevents double-spawn).
    spawned: Arc<AtomicBool>,

    /// A retired runtime state must never be able to resurrect a persistence
    /// worker through an `Arc<LoggingService>` captured by an earlier runtime
    /// invocation. Retirement is permanent for this service instance; a new
    /// process-local runtime receives a new service.
    startable: AtomicBool,

    /// Accepted entries that could not be handed to the bounded persistence
    /// channel. This intentionally excludes replay-window evictions.
    persistence_queue_drops: Arc<AtomicU64>,

    /// Persistence attempts that reached a sink but the sink rejected.
    persistence_failures: Arc<AtomicU64>,

    /// Accepted persistence entries that remain owned by a manual queue or a
    /// worker. A timed-out shutdown moves this exact count into the bounded
    /// loss counter before aborting the worker.
    persistence_outstanding: Arc<AtomicU64>,

    /// Accepted entries lost only because a shutdown drain timed out. This is
    /// separate from ordinary queue saturation and sink failure accounting.
    persistence_shutdown_losses: Arc<AtomicU64>,

    /// Body bytes retained by accepted artifact commands. This independent
    /// byte budget prevents a large count-bound persistence queue from
    /// retaining an unbounded aggregate of request/response bodies.
    artifact_memory_in_flight: Arc<AtomicU64>,
    artifact_memory_budget_bytes: u64,

    /// Fixed upper bound for the worker drain/join phase. Tests inject zero to
    /// exercise the abort/accounting path without wall-clock sleeps.
    shutdown_drain_timeout: Duration,

    /// Weakly referenced by request guards so an explicit terminal transition
    /// and final-handle Drop share the same service delivery path.
    lifecycle_recorder: Arc<dyn LifecycleRecorder>,

    /// Service configuration for observability.
    #[allow(dead_code)]
    config: ServiceConfig,

    /// Coherent live values for the only dynamically applicable logging
    /// settings. The replay bus capacity is adjusted before this snapshot is
    /// published, so readers never see a live snapshot that the bus has not
    /// reached yet.
    dynamic_limits: DynamicLoggingLimits,
}

const PERSISTENCE_FAILURE_AUDIT: &str = "logging persistence delivery failed";
const PROXY_TARGET_CLASSES: &[&str] = &["local", "remote", "external", "none"];
const PROXY_PROVIDER_LABELS: &[&str] = &[
    "openai_frontend",
    "management_api",
    "local",
    "remote",
    "external",
];
const PROXY_ENGINE_LABELS: &[&str] = &[
    "models",
    "chat_completion",
    "chat_completion_stream",
    "completion",
    "completion_stream",
    "responses",
    "responses_stream",
    "local",
    "remote",
    "external",
];
const PROXY_ERROR_LABELS: &[&str] = &[
    "timeout",
    "unavailable",
    "connection_failed",
    "client_disconnected",
    "cancelled",
    "rejected",
    "upstream_status",
];

pub(super) fn validate_proxy_record(record: &ProxyRecord) -> bool {
    is_allowed_proxy_label(PROXY_TARGET_CLASSES, &record.target)
        && record
            .provider
            .as_deref()
            .is_none_or(|value| is_allowed_proxy_label(PROXY_PROVIDER_LABELS, value))
        && record
            .engine
            .as_deref()
            .is_none_or(|value| is_allowed_proxy_label(PROXY_ENGINE_LABELS, value))
        && record
            .error
            .as_deref()
            .is_none_or(|value| is_allowed_proxy_label(PROXY_ERROR_LABELS, value))
}

fn sanitize_proxy_record(mut record: ProxyRecord) -> Option<ProxyRecord> {
    if !is_allowed_proxy_label(PROXY_TARGET_CLASSES, &record.target) {
        return None;
    }
    record.provider = record
        .provider
        .filter(|value| is_allowed_proxy_label(PROXY_PROVIDER_LABELS, value));
    record.engine = record
        .engine
        .filter(|value| is_allowed_proxy_label(PROXY_ENGINE_LABELS, value));
    record.error = record
        .error
        .filter(|value| is_allowed_proxy_label(PROXY_ERROR_LABELS, value));
    Some(record)
}

fn is_allowed_proxy_label(allowed: &[&str], value: &str) -> bool {
    allowed.contains(&value)
}

/// A fallback audit is itself a canonical System event. Its sink failure is
/// deliberately terminal for the fallback path; producing another fallback
/// would create an unbounded self-logging loop.
fn is_fallback_audit(entry: &PersistenceEntry) -> bool {
    let (PersistenceEntry::Bus(entry) | PersistenceEntry::Terminal { entry, .. }) = entry else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&entry.payload)
        .ok()
        .and_then(|record| record.get("canonical_envelope").cloned())
        .and_then(|envelope| CanonicalEnvelope::from_json_str(&envelope.to_string()).ok())
        .is_some_and(|envelope| matches!(envelope.event, LifecycleEvent::AuditError { .. }))
}

fn record_persistence_failure(
    writer: &FailOpenWriter,
    event_delivery: &EventDelivery,
    entry: &PersistenceEntry,
) {
    if is_fallback_audit(entry) {
        writer.record_fallback_suppressed();
        return;
    }

    let event_delivery = event_delivery.clone();
    let _ = writer.try_record_error(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Ok(payload) = serde_json::to_string(&LifecycleEvent::AuditError {
                message: PERSISTENCE_FAILURE_AUDIT.into(),
            }) {
                event_delivery.enqueue(RequestId::new(), ReplayChannel::System, payload);
            }
        }));
    });
}

impl LoggingService {
    pub(crate) const fn artifact_command_max_bytes(&self) -> usize {
        self.config.artifact_command_max_bytes
    }

    /// Return a logging-owned timestamp for bounded persistence metadata.
    ///
    /// Request-path callers reach this only through lifecycle owners, so an
    /// unowned or disabled observer cannot manufacture a durable record.
    pub(crate) fn proxy_record_timestamp(&self) -> String {
        canonical_clock_timestamp(self.clock.as_ref())
    }

    /// Create a new logging service with the given sink and clock. In production, `sink` is the real LogStore; in tests, it's a Vec-backed mock.
    pub fn new(config: ServiceConfig, sink: Arc<dyn PersistSink>, clock: Box<dyn Clock>) -> Self {
        let dynamic_limits = LoggingDynamicLimits {
            retention_ttl_secs: mesh_llm_config::LoggingConfig::default().retention_ttl_secs,
            replay_capacity: config.replay_capacity.min(config.event_buffer_size),
        };
        Self::new_with_dynamic_limits(config, sink, clock, dynamic_limits)
    }

    /// Create a service with its replay bus and live limits initialized from
    /// validated host configuration.
    pub fn new_with_dynamic_limits(
        config: ServiceConfig,
        sink: Arc<dyn PersistSink>,
        clock: Box<dyn Clock>,
        dynamic_limits: LoggingDynamicLimits,
    ) -> Self {
        let bounded_limits = Self::bounded_dynamic_limits(&config, dynamic_limits);
        let bus = Arc::new(ReplayBus::new(bounded_limits.replay_capacity));
        let metrics = bus.metrics();
        let sequences = SequenceGenerators::new();
        let registry = Arc::new(RequestRegistry::new(config.registry_config.clone()));
        let writer = Arc::new(FailOpenWriter::new());
        let clock: Arc<dyn Clock> = Arc::from(clock);
        let delivery = Arc::new(Mutex::new(DeliveryMode::Manual {
            pending: VecDeque::new(),
            capacity: config.queue_capacity.max(1),
        }));
        let persistence_queue_drops = Arc::new(AtomicU64::new(0));
        let persistence_outstanding = Arc::new(AtomicU64::new(0));
        let lifecycle_recorder: Arc<dyn LifecycleRecorder> = Arc::new(ServiceLifecycleRecorder {
            registry: Arc::clone(&registry),
            event_delivery: EventDelivery {
                bus: Arc::clone(&bus),
                registry: Arc::clone(&registry),
                metrics: metrics.clone(),
                sequences: sequences.clone(),
                summary_line_limit: config.summary_line_limit,
                sink_enabled: true,
                clock: Arc::clone(&clock),
                delivery: Arc::clone(&delivery),
                persistence_queue_drops: Arc::clone(&persistence_queue_drops),
                persistence_outstanding: Arc::clone(&persistence_outstanding),
            },
        });

        Self {
            bus,
            metrics,
            sequences,
            registry,
            writer,
            sink: Some(sink),
            clock,
            worker_handle: Mutex::new(None),
            transition_lock: Mutex::new(()),
            manual_abort_handle: Arc::new(Mutex::new(None)),
            delivery,
            spawned: Arc::new(AtomicBool::new(false)),
            startable: AtomicBool::new(true),
            persistence_queue_drops,
            persistence_failures: Arc::new(AtomicU64::new(0)),
            persistence_outstanding,
            persistence_shutdown_losses: Arc::new(AtomicU64::new(0)),
            artifact_memory_in_flight: Arc::new(AtomicU64::new(0)),
            artifact_memory_budget_bytes: DEFAULT_ARTIFACT_MEMORY_BUDGET_BYTES,
            shutdown_drain_timeout: PERSISTENCE_SHUTDOWN_DRAIN_TIMEOUT,
            lifecycle_recorder,
            config,
            dynamic_limits: DynamicLoggingLimits::new(bounded_limits),
        }
    }

    /// Create a service without any persistence sink (events are buffered but never persisted). Useful for testing or disabled logging.
    pub fn new_disabled(config: ServiceConfig) -> Self {
        let replay_capacity = config.replay_capacity.min(config.event_buffer_size);
        let bus = Arc::new(ReplayBus::new(replay_capacity));
        let metrics = bus.metrics();
        let sequences = SequenceGenerators::new();
        let registry = Arc::new(RequestRegistry::new(config.registry_config.clone()));
        let writer = Arc::new(FailOpenWriter::new());
        let clock: Arc<dyn Clock> = Arc::new(SystemClock);
        let delivery = Arc::new(Mutex::new(DeliveryMode::Manual {
            pending: VecDeque::new(),
            capacity: config.queue_capacity.max(1),
        }));
        let persistence_queue_drops = Arc::new(AtomicU64::new(0));
        let persistence_outstanding = Arc::new(AtomicU64::new(0));
        let lifecycle_recorder: Arc<dyn LifecycleRecorder> = Arc::new(ServiceLifecycleRecorder {
            registry: Arc::clone(&registry),
            event_delivery: EventDelivery {
                bus: Arc::clone(&bus),
                registry: Arc::clone(&registry),
                metrics: metrics.clone(),
                sequences: sequences.clone(),
                summary_line_limit: config.summary_line_limit,
                sink_enabled: false,
                clock: Arc::clone(&clock),
                delivery: Arc::clone(&delivery),
                persistence_queue_drops: Arc::clone(&persistence_queue_drops),
                persistence_outstanding: Arc::clone(&persistence_outstanding),
            },
        });

        Self {
            bus,
            metrics,
            sequences,
            registry,
            writer,
            sink: None,
            clock,
            worker_handle: Mutex::new(None),
            transition_lock: Mutex::new(()),
            manual_abort_handle: Arc::new(Mutex::new(None)),
            delivery,
            spawned: Arc::new(AtomicBool::new(false)),
            startable: AtomicBool::new(true),
            persistence_queue_drops,
            persistence_failures: Arc::new(AtomicU64::new(0)),
            persistence_outstanding,
            persistence_shutdown_losses: Arc::new(AtomicU64::new(0)),
            artifact_memory_in_flight: Arc::new(AtomicU64::new(0)),
            artifact_memory_budget_bytes: DEFAULT_ARTIFACT_MEMORY_BUDGET_BYTES,
            shutdown_drain_timeout: PERSISTENCE_SHUTDOWN_DRAIN_TIMEOUT,
            lifecycle_recorder,
            config,
            dynamic_limits: DynamicLoggingLimits::new(LoggingDynamicLimits {
                retention_ttl_secs: mesh_llm_config::LoggingConfig::default().retention_ttl_secs,
                replay_capacity,
            }),
        }
    }

    /// Return the coherent dynamic limit pair currently applied to this
    /// running service. Retention scheduling consumes this later; Todo 6 does
    /// not start a cleanup worker.
    pub fn dynamic_limits(&self) -> LoggingDynamicLimits {
        self.dynamic_limits.snapshot()
    }

    /// Attach the optional process-local telemetry adapter. `None` keeps all
    /// logging metric emissions disabled; attaching the adapter publishes the
    /// current bounded persistence gauge without replaying historical data.
    pub(crate) fn set_metrics_sink(&self, sink: Option<Arc<dyn LoggingMetricsSink>>) {
        self.metrics.set_sink(sink);
        self.metrics.record(LoggingMetric::PersistenceOutstanding {
            current: self.persistence_outstanding.load(Ordering::Acquire),
        });
    }

    pub(crate) fn metrics(&self) -> LoggingMetrics {
        self.metrics.clone()
    }

    pub(crate) fn record_cleanup_outcome(&self, outcome: LoggingCleanupOutcome) {
        self.metrics.record(LoggingMetric::Cleanup { outcome });
    }

    pub(crate) fn record_artifact_capture_status(&self, status: LoggingArtifactCaptureStatus) {
        self.metrics
            .record(LoggingMetric::ArtifactCapture { status });
    }

    /// Apply both dynamically supported logging limits to this running
    /// service. Shrinking replay capacity evicts only the oldest buffered
    /// entries and accounts for each eviction. This is nonblocking except for
    /// the short in-memory mutexes guarding replay and the published snapshot.
    pub fn apply_dynamic_limits(&self, next: LoggingDynamicLimits) {
        let bus = Arc::clone(&self.bus);
        let next = Self::bounded_dynamic_limits(&self.config, next);
        let _ = self.dynamic_limits.apply(next, move |capacity| {
            bus.set_capacity(capacity);
            Ok::<_, std::convert::Infallible>(())
        });
    }

    fn bounded_dynamic_limits(
        config: &ServiceConfig,
        mut limits: LoggingDynamicLimits,
    ) -> LoggingDynamicLimits {
        limits.replay_capacity = limits.replay_capacity.min(config.event_buffer_size);
        limits
    }

    #[cfg(test)]
    pub(crate) fn poison_worker_handle_for_test(&self) {
        let _worker_handle = match self.worker_handle.lock() {
            Ok(worker_handle) => worker_handle,
            Err(poisoned) => poisoned.into_inner(),
        };
        panic!("poison worker handle lock for recovery coverage");
    }

    /// Enqueue a lifecycle event for the given request. This is fail-open: if the bus is full, drop counters increment and Ok(()) returns — the caller should NOT block or retry. Returns `Ok(())` always (the writer absorbs failures).
    pub fn enqueue_event(
        &self,
        request_id: RequestId,
        channel: ReplayChannel,
        payload_json: String,
    ) -> Result<(), BusEnqueueError> {
        let event_delivery = self.event_delivery();
        let _ = enqueue_event_with_delivery(
            &event_delivery,
            request_id,
            channel,
            payload_json,
            None,
            None,
            None,
        );
        Ok(())
    }

    /// Enqueue one bounded proxy attempt for durable persistence. It does not
    /// publish a lifecycle event or transition the parent request, so proxy
    /// observation cannot acquire duplicate terminal ownership.
    pub fn enqueue_proxy_record(&self, record: ProxyRecord) -> Result<(), BusEnqueueError> {
        if self.sink.is_none() {
            return Ok(());
        }

        let Some(record) = sanitize_proxy_record(record) else {
            record_persistence_queue_drop(&self.persistence_queue_drops, &self.metrics);
            return Ok(());
        };
        let proxy_json = match serde_json::to_string(&record) {
            Ok(proxy_json) => proxy_json,
            Err(_) => {
                record_persistence_queue_drop(&self.persistence_queue_drops, &self.metrics);
                return Ok(());
            }
        };
        {
            offer_persistence_to(
                &self.delivery,
                &self.persistence_queue_drops,
                &self.persistence_outstanding,
                &self.metrics,
                PersistenceEntry::ProxyRecord(proxy_json),
            );
        }
        Ok(())
    }

    fn event_delivery(&self) -> EventDelivery {
        EventDelivery {
            bus: Arc::clone(&self.bus),
            registry: Arc::clone(&self.registry),
            metrics: self.metrics.clone(),
            sequences: self.sequences.clone(),
            summary_line_limit: self.config.summary_line_limit,
            sink_enabled: self.sink.is_some(),
            clock: Arc::clone(&self.clock),
            delivery: Arc::clone(&self.delivery),
            persistence_queue_drops: Arc::clone(&self.persistence_queue_drops),
            persistence_outstanding: Arc::clone(&self.persistence_outstanding),
        }
    }

    /// Register a new request in the active registry and emit an admitted event on the Requests channel. Returns a LifecycleGuard for tracking terminal transitions.
    pub fn register_request(&self, request_id: RequestId) -> (LifecycleGuard, EventId) {
        self.register_request_with_metadata(request_id, RequestSummaryMetadata::default())
    }

    /// Register a request with metadata available at its trusted ingress boundary.
    pub(crate) fn register_request_with_metadata(
        &self,
        request_id: RequestId,
        metadata: RequestSummaryMetadata,
    ) -> (LifecycleGuard, EventId) {
        let guard =
            LifecycleGuard::for_request(request_id, Arc::downgrade(&self.lifecycle_recorder));

        // Register summary in active set.
        let created_at = canonical_clock_timestamp(self.clock.as_ref());
        let admitted_model = metadata.model().map(str::to_owned);
        let admitted_method = metadata.method().map(str::to_owned);
        let summary = RequestSummaryEntry {
            request_id: request_id.as_uuid().to_string(),
            state: "active".into(),
            created_at: created_at.clone(),
            terminal_at: None,
            metadata,
        };
        self.registry.register_active(summary.clone());
        if self.sink.is_some() {
            offer_summary_persistence(
                &self.delivery,
                &self.persistence_queue_drops,
                &self.persistence_outstanding,
                &self.metrics,
                summary,
            );
        }

        // The summary must precede the admitted envelope on the single
        // persistence delivery path so the typed lifecycle row can satisfy
        // its SQLite summary foreign key. Returning this exact ID lets callers
        // correlate registration with the canonical envelope and durable row.
        let event_id = enqueue_event_with_delivery(
            &self.event_delivery(),
            request_id,
            ReplayChannel::Requests,
            serde_json::to_string(&LifecycleEvent::Admitted {
                model: admitted_model,
                method: admitted_method,
            })
            .expect("LifecycleEvent serialization is infallible"),
            Some(created_at),
            None,
            None,
        );

        (guard, event_id)
    }

    /// Merge newly available truthful metadata into the request projection.
    /// Evicted or absent requests simply retain no update.
    pub(crate) fn merge_request_metadata(
        &self,
        request_id: RequestId,
        metadata: RequestSummaryMetadata,
    ) {
        let Some(summary) = self
            .registry
            .merge_metadata(&request_id.as_uuid().to_string(), metadata)
        else {
            return;
        };
        if self.sink.is_some() {
            offer_summary_persistence(
                &self.delivery,
                &self.persistence_queue_drops,
                &self.persistence_outstanding,
                &self.metrics,
                summary,
            );
        }
    }

    pub(crate) fn merge_authenticated_remote_caller(
        &self,
        request_id: RequestId,
        metadata: RequestSummaryMetadata,
    ) -> bool {
        let Some(summary) = self
            .registry
            .merge_authenticated_remote_caller(&request_id.as_uuid().to_string(), metadata)
        else {
            return false;
        };
        if self.sink.is_some() {
            offer_summary_persistence(
                &self.delivery,
                &self.persistence_queue_drops,
                &self.persistence_outstanding,
                &self.metrics,
                summary,
            );
        }
        true
    }

    /// Record the beginning of a transport attempt under an existing request.
    /// The returned branded identifier is used by its completion or failure and
    /// never changes the parent request lifecycle.
    pub fn start_attempt(&self, request_id: RequestId, guard: &LifecycleGuard) -> AttemptId {
        let attempt_id = guard.record_attempt();
        self.enqueue_lifecycle_event(
            request_id,
            ReplayChannel::Operations,
            LifecycleEvent::AttemptStarted {
                attempt_id: Some(attempt_id),
            },
        );
        attempt_id
    }

    /// Record a successful transport attempt without terminalizing its parent request.
    pub fn complete_attempt(
        &self,
        request_id: RequestId,
        attempt_id: AttemptId,
        status_code: Option<u16>,
    ) {
        self.enqueue_lifecycle_event(
            request_id,
            ReplayChannel::Operations,
            LifecycleEvent::AttemptCompleted {
                attempt_id: Some(attempt_id),
                status_code,
            },
        );
    }

    /// Record a failed transport attempt without terminalizing its parent request.
    pub fn fail_attempt(&self, request_id: RequestId, attempt_id: AttemptId, error: String) {
        self.enqueue_lifecycle_event(
            request_id,
            ReplayChannel::Operations,
            LifecycleEvent::AttemptFailed {
                attempt_id: Some(attempt_id),
                error: Some(error),
            },
        );
    }

    /// Transition a request to a terminal outcome. Moves the summary from active → recent in the registry and emits a terminal lifecycle event on the bus. Returns `Err(DuplicateTerminalError)` if already terminated (idempotent rejection).
    pub fn transition_terminal(
        &self,
        request_id: RequestId,
        guard: &LifecycleGuard,
        outcome: TerminalOutcome,
    ) -> Result<(), DuplicateTerminalError> {
        let _ = request_id;
        guard.terminate(outcome)
    }

    fn enqueue_lifecycle_event(
        &self,
        request_id: RequestId,
        channel: ReplayChannel,
        event: LifecycleEvent,
    ) {
        if let Ok(payload) = serde_json::to_string(&event) {
            let _ = self.enqueue_event(request_id, channel, payload);
        }
    }

    /// Write an error audit entry using the fail-open writer's recursion guard. Returns `true` if written, `false` if blocked by recursion detection (caller should proceed silently). Never panics.
    pub fn write_error_audit(&self, message: String) -> bool {
        // Best-effort audit write — fail-open. Use the same canonical System
        // event path as every other record so replay and persistence cannot
        // diverge. The recursion guard prevents self-logging loops.
        let event_delivery = self.event_delivery();

        self.writer.try_record_error(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Ok(payload) = serde_json::to_string(&LifecycleEvent::AuditError { message })
                {
                    event_delivery.enqueue(RequestId::new(), ReplayChannel::System, payload);
                }
            }));
        })
    }

    /// Write one bounded operational audit record through the same replay and
    /// persistence hand-off as lifecycle records.
    pub fn write_operational_audit(&self, record: OperationalAuditRecord) -> bool {
        let event_delivery = self.event_delivery();
        self.writer.try_record_error(move || {
            event_delivery.enqueue_audit(record);
        })
    }

    /// Get total rejected entries and writer write drops. Replay-window evictions
    /// are tracked separately by the bus and are not rejected new events.
    #[allow(dead_code)]
    pub fn total_drops(&self) -> u64 {
        self.bus.drops.load(Ordering::Relaxed) + self.writer.write_drops.load(Ordering::Relaxed)
    }

    /// Number of accepted entries dropped because the dedicated persistence
    /// hand-off was full or unavailable. Replay evictions are excluded.
    #[allow(dead_code)]
    pub fn persistence_queue_drops(&self) -> u64 {
        self.persistence_queue_drops.load(Ordering::Relaxed)
    }

    /// Number of persistence attempts rejected by the sink. These failures do
    /// not alter request serving or the replay window.
    #[allow(dead_code)]
    pub fn persistence_failures(&self) -> u64 {
        self.persistence_failures.load(Ordering::Relaxed)
    }

    /// Number of accepted persistence entries abandoned only after a bounded
    /// shutdown drain timed out. Queue saturation and sink failures are
    /// reported separately.
    #[allow(dead_code)]
    pub fn persistence_shutdown_losses(&self) -> u64 {
        self.persistence_shutdown_losses.load(Ordering::Relaxed)
    }

    /// Number of one-time persistence entries currently owned by the manual
    /// queue or worker. This is intended for local health/tests only.
    #[allow(dead_code)]
    pub fn persistence_outstanding(&self) -> u64 {
        self.persistence_outstanding.load(Ordering::Relaxed)
    }

    /// Get the bus for direct access (tests).
    #[allow(dead_code)]
    pub fn bus_ref(&self) -> Arc<ReplayBus> {
        Arc::clone(&self.bus)
    }

    /// Get the registry for direct access (tests).
    #[allow(dead_code)]
    pub fn registry_ref(&self) -> Arc<RequestRegistry> {
        Arc::clone(&self.registry)
    }

    /// Get sequence generators reference.
    #[allow(dead_code)]
    pub fn sequences_ref(&self) -> &SequenceGenerators {
        &self.sequences
    }

    /// Permanently prevent this concrete service from starting another worker.
    ///
    /// The owning [`LoggingRuntimeState`](super::runtime_state::LoggingRuntimeState)
    /// calls this before it begins asynchronous cleanup. This narrow atomic
    /// boundary makes an already-captured service handle safe while replacement
    /// releases global locks and awaits bounded worker shutdown.
    pub(crate) fn retire(&self) {
        self.startable.store(false, Ordering::Release);
    }
    pub(crate) fn is_startable(&self) -> bool {
        self.startable.load(Ordering::Acquire)
    }
    #[cfg(test)]
    pub(crate) fn with_shutdown_drain_timeout(mut self, timeout: Duration) -> Self {
        self.shutdown_drain_timeout = timeout;
        self
    }
    /// Check if the service is currently spawned and running. For observability / tests.
    #[allow(dead_code)]
    pub fn is_spawned(&self) -> bool {
        self.spawned.load(Ordering::Acquire)
    }
    /// Clone writer for external observation of drop counters.
    #[allow(dead_code)]
    pub fn writer_ref(&self) -> Arc<FailOpenWriter> {
        Arc::clone(&self.writer)
    }
    #[cfg(test)]
    pub(crate) fn worker_handle_lock_for_test(
        &self,
    ) -> std::sync::MutexGuard<'_, Option<WorkerHandle>> {
        self.worker_handle
            .lock()
            .expect("worker handle lock starts healthy")
    }
}
