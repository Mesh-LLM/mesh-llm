//! Host-owned durable logging state.
//!
//! This is intentionally a narrow startup boundary. It opens the durable
//! metadata store and the independently fail-open artifact capture facade,
//! but it does not start workers or instrument request producers.

use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use mesh_llm_events::logging::identifiers::EventId;
use mesh_llm_log_store::{
    ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE, ArtifactContent, ArtifactRecord,
    ArtifactRedactor, AuditEntryFilters, AuditEntryRow, Clock as StoreClock, EventRecord,
    FailOpenArtifactCapture, LogStore, LogStoreError, Page, PageQuery, ProxyQuery, ProxyRecord,
    QueryPage, RealClock, RequestQuery, RequestRecord,
};
#[cfg(test)]
use mesh_llm_log_store::{ArtifactCaptureDisabledReason, ArtifactCaptureOutcome};

#[cfg(test)]
use super::LoggingArtifactCaptureStatus;
use super::cleanup::{CleanupOutcome, CleanupWorker, CleanupWorkerState, CleanupWorkerStatus};
use super::foundation::LoggingFoundation;
use super::openai_lifecycle::{
    OpenAiArtifactCapture, OpenAiLifecycleAttachment, OpenAiLifecycleLoggingAdapter,
};
use super::operator_audit::OperatorAuditWriter;
use super::policy::{PolicyCaptureMode, redact_artifact_bytes};
use super::writer::FailOpenWriter;
use super::{
    ActiveRequestSnapshot, LogStoreSink, LoggingDynamicLimits, LoggingMetricsSink, LoggingService,
    ManagementRequestLifecycle, RandomWebhookJitter, RawMeshLifecycleOwners,
    RawMeshRemoteSuppressionLease, RawMeshRequestLifecycle, RequestSummaryMetadata,
    ReqwestWebhookTransport, ServiceConfig, SystemClock, SystemWebhookWorkerClock,
    WebhookDeliveryScheduler, WebhookDeliveryWorker,
};

const HEALTH_AUDIT_ACTOR: &str = "logging-runtime";

/// Internal capability state for local logging storage.
///
/// The only artifact-degradation value exposed from this state is the stable,
/// path-free circuit-breaker code. Errors and filesystem locations remain
/// private to the storage implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoggingRuntimeHealth {
    pub metadata_available: bool,
    /// Storage can be available while no ingress has installed a producer.
    pub artifact_capture_available: bool,
    pub artifact_capture_ready: bool,
    pub artifact_capture_degradation: Option<&'static str>,
}

/// Path-free runtime logging status for trusted-local API consumers.
///
/// This deliberately exposes only fixed state labels, counters, and the
/// stable artifact circuit-breaker code. Storage paths and backend errors stay
/// inside the logging implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LoggingRuntimeStatus {
    pub(crate) metadata_available: bool,
    pub(crate) metadata_state: &'static str,
    pub(crate) schema_version: Option<u32>,
    pub(crate) supported_schema_version: Option<u32>,
    pub(crate) capture_mode: &'static str,
    pub(crate) artifact_capture_available: bool,
    pub(crate) artifact_capture_ready: bool,
    pub(crate) artifact_capture_degradation: Option<&'static str>,
    pub(crate) persistence_worker_state: &'static str,
    pub(crate) persistence_queue_drops: u64,
    pub(crate) persistence_failures: u64,
    pub(crate) persistence_shutdown_losses: u64,
    pub(crate) persistence_outstanding: u64,
    pub(crate) cleanup_worker_state: &'static str,
    pub(crate) cleanup_shutdown_timeouts: u64,
    pub(crate) cleanup_last_outcome: Option<&'static str>,
    pub(crate) cleanup_last_deleted_count: Option<u64>,
}

/// Stable, path-free explanation for metadata availability. Schema versions
/// are safe to expose to trusted-local clients and make upgrade/downgrade
/// failures actionable without leaking the database location or SQLite text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LoggingMetadataState {
    Ready,
    Disabled,
    StorageUnavailable,
    SchemaIncompatible { found: u32, supported: u32 },
}

impl LoggingMetadataState {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Disabled => "disabled",
            Self::StorageUnavailable => "storage_unavailable",
            Self::SchemaIncompatible { .. } => "schema_incompatible",
        }
    }

    const fn schema_versions(self) -> (Option<u32>, Option<u32>) {
        match self {
            Self::SchemaIncompatible { found, supported } => (Some(found), Some(supported)),
            _ => (None, None),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LoggingQueryCounts {
    pub(crate) point_requests: usize,
    pub(crate) batch_requests: usize,
}

#[cfg(test)]
#[derive(Default)]
struct LoggingQueryCounter {
    point_requests: AtomicUsize,
    batch_requests: AtomicUsize,
}

#[cfg(test)]
impl LoggingQueryCounter {
    fn snapshot(&self) -> LoggingQueryCounts {
        LoggingQueryCounts {
            point_requests: self.point_requests.load(Ordering::Relaxed),
            batch_requests: self.batch_requests.load(Ordering::Relaxed),
        }
    }
}

impl LoggingRuntimeHealth {
    fn unavailable() -> Self {
        Self {
            metadata_available: false,
            artifact_capture_available: false,
            artifact_capture_ready: false,
            artifact_capture_degradation: None,
        }
    }
}

/// A fully specified test artifact write owned by the logging runtime.
///
/// Keeping this request at the boundary prevents callers from reaching into
/// the fail-open capture object and forgetting to publish its one-shot health
/// marker after a privacy failure.
#[cfg(test)]
pub struct ArtifactCaptureRequest<'a> {
    pub artifact_id: &'a str,
    pub request_id: &'a str,
    pub kind: &'a str,
    pub occurred_at: &'a str,
    pub content: &'a [u8],
    pub media_kind: Option<&'a str>,
    pub version: u32,
    pub truncated: bool,
}

struct RuntimeOpenAiArtifactCapture {
    service: Weak<LoggingService>,
}

impl OpenAiArtifactCapture for RuntimeOpenAiArtifactCapture {
    fn body_limit_bytes(&self) -> usize {
        self.service
            .upgrade()
            .map_or(0, |service| service.artifact_command_max_bytes())
    }

    fn capture_body(
        &self,
        request_id: mesh_llm_events::logging::identifiers::RequestId,
        kind: &'static str,
        content: &[u8],
        media_kind: Option<&str>,
    ) {
        if let Some(service) = self.service.upgrade() {
            service.enqueue_openai_artifact_body(request_id, kind, content, media_kind);
        }
    }

    fn capture_unavailable(
        &self,
        request_id: mesh_llm_events::logging::identifiers::RequestId,
        kind: &'static str,
        reason: super::ArtifactUnavailableReason,
    ) {
        if let Some(service) = self.service.upgrade() {
            service.enqueue_openai_artifact_unavailable(request_id, kind, reason);
        }
    }
}

/// The process-local durable logging resources created from a foundation.
pub struct LoggingRuntimeState {
    store: Option<Arc<LogStore>>,
    service: Option<Arc<LoggingService>>,
    raw_mesh_owners: Arc<RawMeshLifecycleOwners>,
    artifact_capture: Option<Arc<FailOpenArtifactCapture>>,
    #[cfg(test)]
    artifact_limits: Option<(usize, usize)>,
    artifact_export_enabled: bool,
    capture_mode: &'static str,
    metadata_state: LoggingMetadataState,
    export_limit_bytes: usize,
    health: Mutex<LoggingRuntimeHealth>,
    health_audit_writer: FailOpenWriter,
    operator_audit_writer: Arc<OperatorAuditWriter>,
    cleanup_worker: Mutex<Option<CleanupWorker>>,
    cleanup_status: Arc<Mutex<CleanupWorkerStatus>>,
    webhook_delivery_worker: Mutex<Option<WebhookDeliveryScheduler>>,
    webhook_config: Option<mesh_llm_config::LoggingWebhookConfig>,
    /// Serializes the synchronous start/retire boundary. The asynchronous
    /// cleanup and drain paths deliberately run after this short lock is
    /// released, so no ordinary lock is held across an await.
    activation_lock: Mutex<()>,
    /// Once a process-local state is replaced, any captured `Arc` must be
    /// unable to create another worker.
    retired: AtomicBool,
    cleanup_cadence: Duration,
    retention_max_rows: u64,
    webhook_dead_letter_retention_secs: u64,
    #[cfg(test)]
    cleanup_install_hook: Mutex<Option<Arc<CleanupInstallHook>>>,
    #[cfg(test)]
    cleanup_candidate_count: AtomicUsize,
    #[cfg(test)]
    query_counter: Arc<LoggingQueryCounter>,
}

mod query_facade;
mod workers;
pub(crate) use query_facade::LoggingQueryFacade;

/// Deterministic test-only pause after the scheduler is atomically published
/// and before its starter awaits readiness. Two barriers make replacement
/// cancel and join that installed scheduler without relying on wall-clock
/// scheduling.
#[cfg(test)]
struct CleanupInstallHook {
    candidate_created: tokio::sync::Barrier,
    resume_install: tokio::sync::Barrier,
}

#[cfg(test)]
impl CleanupInstallHook {
    fn new() -> Self {
        Self {
            candidate_created: tokio::sync::Barrier::new(2),
            resume_install: tokio::sync::Barrier::new(2),
        }
    }
}

impl LoggingRuntimeState {
    /// Open/migrate the local store and initialize independent artifact capture.
    ///
    /// A failure at either layer is fail-open for serving. In particular,
    /// artifact privacy failure never removes the already-open metadata store.
    pub fn initialize(
        foundation: &LoggingFoundation,
        config: &mesh_llm_config::LoggingConfig,
    ) -> Self {
        Self::initialize_with_capture_opener(foundation, config, |artifact_root, clock, store| {
            FailOpenArtifactCapture::open(
                artifact_root,
                clock,
                store,
                canonical_artifact_redactor(),
            )
        })
    }

    fn initialize_with_capture_opener<F>(
        foundation: &LoggingFoundation,
        config: &mesh_llm_config::LoggingConfig,
        open_capture: F,
    ) -> Self
    where
        F: FnOnce(
            PathBuf,
            Arc<dyn StoreClock>,
            Arc<LogStore>,
        ) -> Result<FailOpenArtifactCapture, LogStoreError>,
    {
        if !foundation.is_healthy() {
            tracing::warn!("Logging durable storage unavailable; continuing without logging");
            let metadata_state = if config.enabled {
                LoggingMetadataState::StorageUnavailable
            } else {
                LoggingMetadataState::Disabled
            };
            return Self::unavailable(metadata_state);
        }

        Self::initialize_healthy_foundation(foundation, config, open_capture)
    }

    fn initialize_healthy_foundation<F>(
        foundation: &LoggingFoundation,
        config: &mesh_llm_config::LoggingConfig,
        open_capture: F,
    ) -> Self
    where
        F: FnOnce(
            PathBuf,
            Arc<dyn StoreClock>,
            Arc<LogStore>,
        ) -> Result<FailOpenArtifactCapture, LogStoreError>,
    {
        let clock: Arc<dyn StoreClock> = Arc::new(RealClock);
        Self::initialize_healthy_foundation_with_clock(foundation, config, clock, open_capture)
    }

    fn initialize_healthy_foundation_with_clock<F>(
        foundation: &LoggingFoundation,
        config: &mesh_llm_config::LoggingConfig,
        clock: Arc<dyn StoreClock>,
        open_capture: F,
    ) -> Self
    where
        F: FnOnce(
            PathBuf,
            Arc<dyn StoreClock>,
            Arc<LogStore>,
        ) -> Result<FailOpenArtifactCapture, LogStoreError>,
    {
        let store = match Self::open_metadata_store(foundation, &clock) {
            Ok(store) => store,
            Err(metadata_state) => return Self::unavailable(metadata_state),
        };
        let policy = super::policy::build_policy(config);
        let (artifact_capture, artifact_limits) = match policy.capture_mode {
            PolicyCaptureMode::MetadataOnly => (None, None),
            PolicyCaptureMode::RedactedWithLimits(byte_limit, aggregate_limit) => (
                Self::open_artifact_capture(foundation, clock, &store, open_capture).map(Arc::new),
                Some((byte_limit, aggregate_limit)),
            ),
        };
        Self::from_open_store(store, artifact_capture, artifact_limits, config)
    }

    fn open_metadata_store(
        foundation: &LoggingFoundation,
        clock: &Arc<dyn StoreClock>,
    ) -> Result<Arc<LogStore>, LoggingMetadataState> {
        match LogStore::open(foundation.store_dir(), Arc::clone(clock)) {
            Ok(store) => Ok(Arc::new(store)),
            Err(LogStoreError::SchemaIncompatible { found, supported }) => {
                tracing::warn!(
                    found_schema_version = found,
                    supported_schema_version = supported,
                    "Logging database schema is incompatible; database left unchanged"
                );
                Err(LoggingMetadataState::SchemaIncompatible { found, supported })
            }
            Err(_) => {
                tracing::warn!("Logging durable storage unavailable; continuing without logging");
                Err(LoggingMetadataState::StorageUnavailable)
            }
        }
    }

    fn open_artifact_capture<F>(
        foundation: &LoggingFoundation,
        clock: Arc<dyn StoreClock>,
        store: &Arc<LogStore>,
        open_capture: F,
    ) -> Option<FailOpenArtifactCapture>
    where
        F: FnOnce(
            PathBuf,
            Arc<dyn StoreClock>,
            Arc<LogStore>,
        ) -> Result<FailOpenArtifactCapture, LogStoreError>,
    {
        match open_capture(
            foundation.artifact_dir().to_path_buf(),
            clock,
            Arc::clone(store),
        ) {
            Ok(capture) => Some(capture),
            Err(_) => {
                tracing::warn!(
                    "Logging artifact capture unavailable; metadata logging remains enabled"
                );
                None
            }
        }
    }

    fn from_open_store(
        store: Arc<LogStore>,
        artifact_capture: Option<Arc<FailOpenArtifactCapture>>,
        artifact_limits: Option<(usize, usize)>,
        config: &mesh_llm_config::LoggingConfig,
    ) -> Self {
        let policy = super::policy::build_policy(config);
        let artifact_capture_available = artifact_capture
            .as_ref()
            .is_some_and(|capture| !capture.is_disabled());
        let persistence_sink = if config.webhook.enabled {
            LogStoreSink::with_terminal_webhook_enqueue(
                Arc::clone(&store),
                config.webhook.max_attempts,
            )
        } else {
            LogStoreSink::new(Arc::clone(&store))
        };
        let persistence_sink = match (artifact_capture.as_ref(), artifact_limits) {
            (Some(capture), Some((byte_limit, aggregate_limit))) => persistence_sink
                .with_artifact_capture(Arc::clone(capture), byte_limit, aggregate_limit),
            _ => persistence_sink,
        };
        let state = Self {
            service: Some(Arc::new(LoggingService::new_with_dynamic_limits(
                ServiceConfig::from_policy(&policy),
                Arc::new(persistence_sink),
                Box::new(SystemClock),
                LoggingDynamicLimits::from_config(config),
            ))),
            raw_mesh_owners: Arc::new(RawMeshLifecycleOwners::default()),
            store: Some(store),
            artifact_capture,
            #[cfg(test)]
            artifact_limits,
            artifact_export_enabled: matches!(
                config.artifact.capture_mode,
                mesh_llm_config::CaptureMode::RedactedArtifacts
            ) && artifact_capture_available,
            capture_mode: match config.artifact.capture_mode {
                mesh_llm_config::CaptureMode::MetadataOnly => "metadata_only",
                mesh_llm_config::CaptureMode::RedactedArtifacts => "redacted_artifacts",
            },
            metadata_state: LoggingMetadataState::Ready,
            export_limit_bytes: config.export_limit_bytes as usize,
            health: Mutex::new(LoggingRuntimeHealth {
                metadata_available: true,
                artifact_capture_available,
                // Storage is usable only because the production ingress
                // attachment is installed below. Metadata-only and degraded
                // configurations still report false.
                artifact_capture_ready: artifact_capture_available,
                artifact_capture_degradation: None,
            }),
            health_audit_writer: FailOpenWriter::new(),
            operator_audit_writer: Arc::new(OperatorAuditWriter::new()),
            cleanup_worker: Mutex::new(None),
            cleanup_status: Arc::new(Mutex::new(CleanupWorkerStatus::default())),
            webhook_delivery_worker: Mutex::new(None),
            webhook_config: config.webhook.enabled.then(|| config.webhook.clone()),
            activation_lock: Mutex::new(()),
            retired: AtomicBool::new(false),
            cleanup_cadence: Duration::from_secs(config.cleanup_cadence_secs),
            retention_max_rows: config.retention_max_rows,
            webhook_dead_letter_retention_secs: config.webhook.dead_letter_retention_secs,
            #[cfg(test)]
            cleanup_install_hook: Mutex::new(None),
            #[cfg(test)]
            cleanup_candidate_count: AtomicUsize::new(0),
            #[cfg(test)]
            query_counter: Arc::new(LoggingQueryCounter::default()),
        };
        state.consume_artifact_capture_health_marker();
        state
    }

    fn unavailable(metadata_state: LoggingMetadataState) -> Self {
        Self {
            store: None,
            service: None,
            raw_mesh_owners: Arc::new(RawMeshLifecycleOwners::default()),
            artifact_capture: None,
            #[cfg(test)]
            artifact_limits: None,
            artifact_export_enabled: false,
            capture_mode: "unavailable",
            metadata_state,
            export_limit_bytes: mesh_llm_config::LoggingConfig::default().export_limit_bytes
                as usize,
            health: Mutex::new(LoggingRuntimeHealth::unavailable()),
            health_audit_writer: FailOpenWriter::new(),
            operator_audit_writer: Arc::new(OperatorAuditWriter::new()),
            cleanup_worker: Mutex::new(None),
            cleanup_status: Arc::new(Mutex::new(CleanupWorkerStatus::default())),
            webhook_delivery_worker: Mutex::new(None),
            webhook_config: None,
            activation_lock: Mutex::new(()),
            retired: AtomicBool::new(false),
            cleanup_cadence: Duration::from_secs(
                mesh_llm_config::LoggingConfig::default().cleanup_cadence_secs,
            ),
            retention_max_rows: mesh_llm_config::LoggingConfig::default().retention_max_rows,
            webhook_dead_letter_retention_secs: mesh_llm_config::LoggingConfig::default()
                .webhook
                .dead_letter_retention_secs,
            #[cfg(test)]
            cleanup_install_hook: Mutex::new(None),
            #[cfg(test)]
            cleanup_candidate_count: AtomicUsize::new(0),
            #[cfg(test)]
            query_counter: Arc::new(LoggingQueryCounter::default()),
        }
    }

    /// Return the internal health/capability projection without filesystem details.
    pub fn health(&self) -> LoggingRuntimeHealth {
        // The serial artifact worker may trip the privacy circuit breaker
        // after startup. Consume its one-shot marker at the next operator
        // health read so availability never remains falsely ready.
        self.consume_artifact_capture_health_marker();
        *self
            .health
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Install the optional process-local telemetry adapter after runtime
    /// configuration has explicitly enabled telemetry. Logging remains fully
    /// independent of the adapter and unavailable state stays fail-open.
    pub(crate) fn set_metrics_sink(&self, sink: Option<Arc<dyn LoggingMetricsSink>>) {
        if self.retired.load(Ordering::Acquire) {
            return;
        }
        if let Some(service) = self.service.as_ref() {
            service.set_metrics_sink(sink);
        }
    }

    /// Return a fixed-label, path-free snapshot suitable for trusted-local
    /// management status. This is intentionally not a mesh capability.
    pub(crate) fn status(&self) -> LoggingRuntimeStatus {
        let health = self.health();
        let cleanup = *self
            .cleanup_status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (
            persistence_worker_state,
            persistence_queue_drops,
            persistence_failures,
            persistence_shutdown_losses,
            persistence_outstanding,
        ) = self
            .service
            .as_ref()
            .map_or(("unavailable", 0, 0, 0, 0), |service| {
                (
                    service.persistence_worker_state().label(),
                    service.persistence_queue_drops(),
                    service.persistence_failures(),
                    service.persistence_shutdown_losses(),
                    service.persistence_outstanding(),
                )
            });

        let (schema_version, supported_schema_version) = self.metadata_state.schema_versions();
        LoggingRuntimeStatus {
            metadata_available: health.metadata_available,
            metadata_state: self.metadata_state.code(),
            schema_version,
            supported_schema_version,
            capture_mode: self.capture_mode,
            artifact_capture_available: health.artifact_capture_available,
            artifact_capture_ready: health.artifact_capture_ready,
            artifact_capture_degradation: health.artifact_capture_degradation,
            persistence_worker_state,
            persistence_queue_drops,
            persistence_failures,
            persistence_shutdown_losses,
            persistence_outstanding,
            cleanup_worker_state: cleanup_worker_state_label(cleanup.state),
            cleanup_shutdown_timeouts: cleanup.shutdown_timeouts,
            cleanup_last_outcome: cleanup.last_outcome.map(|outcome| outcome.code()),
            cleanup_last_deleted_count: cleanup
                .last_outcome
                .and_then(|outcome| outcome.deleted_count()),
        }
    }

    /// Test-only direct capture seam. Production ingress always uses the
    /// logging service's bounded serial persistence command.
    #[cfg(test)]
    pub fn write_artifact(
        &self,
        request: ArtifactCaptureRequest<'_>,
    ) -> Result<ArtifactCaptureOutcome, LogStoreError> {
        // Redact before bytes leave the host-owned boundary. The capture store
        // applies the same mandatory redactor again as defense in depth.
        let Some((byte_limit, aggregate_limit)) = self.artifact_limits else {
            return Ok(ArtifactCaptureOutcome::Disabled(
                ArtifactCaptureDisabledReason,
            ));
        };
        let redacted_content = redact_artifact_bytes(request.content);
        let outcome = match self.artifact_capture.as_ref() {
            Some(capture) => capture.write_artifact(
                request.artifact_id,
                request.request_id,
                request.kind,
                request.occurred_at,
                &redacted_content,
                request.media_kind,
                request.version,
                true,
                request.truncated,
                byte_limit,
                aggregate_limit,
            ),
            None => Ok(ArtifactCaptureOutcome::Disabled(
                ArtifactCaptureDisabledReason,
            )),
        };
        let status = match &outcome {
            Ok(ArtifactCaptureOutcome::Written(_)) => LoggingArtifactCaptureStatus::Written,
            Ok(ArtifactCaptureOutcome::Disabled(_)) => LoggingArtifactCaptureStatus::Disabled,
            Err(_) => LoggingArtifactCaptureStatus::Failed,
        };
        if let Some(service) = self.service.as_ref() {
            service.record_artifact_capture_status(status);
        }
        self.consume_artifact_capture_health_marker();
        outcome
    }

    /// Access to the typed store stays internal to host runtime ownership.
    pub(crate) fn store(&self) -> Option<Arc<LogStore>> {
        self.store.clone()
    }

    /// Create the only query/read handle used by trusted-local log API code.
    /// Disabled or degraded metadata storage yields no facade; artifact
    /// degradation alone still yields a facade for metadata-only queries.
    pub(crate) fn query_facade(&self) -> Option<LoggingQueryFacade> {
        LoggingQueryFacade::from_runtime(self)
    }

    pub(crate) const fn metadata_state(&self) -> LoggingMetadataState {
        self.metadata_state
    }

    /// Drain the serial persistence queue for an integration test without
    /// widening the production OpenAI transport's storage boundary.
    #[cfg(test)]
    pub(crate) async fn pump_persistence_for_test(&self) -> usize {
        match self.service.as_ref() {
            Some(service) => service.pump_sync().await,
            None => 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn query_counts_for_test(&self) -> LoggingQueryCounts {
        self.query_counter.snapshot()
    }

    /// Return the bounded semantic replay source without exposing the logging
    /// service's persistence or registry internals to the HTTP adapter.
    pub(crate) fn replay_bus(&self) -> Option<Arc<super::bus::ReplayBus>> {
        self.service.as_ref().map(|service| service.bus_ref())
    }

    /// Snapshot the host-owned OpenAI lifecycle observer for one frontend
    /// server instance. A disabled, unavailable, or retired runtime stays
    /// absent so request serving continues without logging.
    pub(crate) fn openai_lifecycle_observer(
        &self,
    ) -> Option<Arc<dyn openai_frontend::OpenAiLifecycleObserver>> {
        if self.retired.load(Ordering::Acquire) {
            return None;
        }
        let service = Arc::clone(self.service.as_ref()?);
        if !service.is_startable() {
            return None;
        }
        Some(Arc::new(OpenAiLifecycleLoggingAdapter::new(
            service,
            Arc::clone(&self.raw_mesh_owners),
        )))
    }

    /// Claim the metadata-only parent lifecycle for one raw mesh ingress
    /// request. The matching embedded frontend observer consults the same
    /// ownership registry and does not register a competing parent.
    pub(crate) fn register_raw_mesh_request(
        &self,
        request_id: mesh_llm_events::logging::identifiers::RequestId,
    ) -> Option<RawMeshRequestLifecycle> {
        self.register_raw_mesh_request_with_metadata(request_id, RequestSummaryMetadata::default())
    }

    fn register_raw_mesh_request_with_metadata(
        &self,
        request_id: mesh_llm_events::logging::identifiers::RequestId,
        metadata: RequestSummaryMetadata,
    ) -> Option<RawMeshRequestLifecycle> {
        if self.retired.load(Ordering::Acquire) {
            return None;
        }
        let service = Arc::clone(self.service.as_ref()?);
        if !service.is_startable() {
            return None;
        }
        RawMeshRequestLifecycle::register_with_metadata(
            service,
            Arc::clone(&self.raw_mesh_owners),
            request_id,
            metadata,
        )
    }

    /// Attach one parsed host OpenAI ingress to the canonical parent owner.
    ///
    /// The returned attachment remains usable when logging is unavailable; its
    /// route observer is then empty and all dispatch instrumentation fails open.
    pub(crate) fn openai_ingress_attachment(
        self: &Arc<Self>,
        request_id: mesh_llm_events::logging::identifiers::RequestId,
        metadata: RequestSummaryMetadata,
    ) -> OpenAiLifecycleAttachment {
        let parent = self.register_raw_mesh_request_with_metadata(request_id, metadata);
        let Some(parent) = parent else {
            return OpenAiLifecycleAttachment::unowned();
        };
        let Some(_) = self.artifact_capture.as_ref() else {
            return OpenAiLifecycleAttachment::new(Some(parent));
        };
        let Some(service) = self.service.as_ref() else {
            return OpenAiLifecycleAttachment::new(Some(parent));
        };
        OpenAiLifecycleAttachment::with_capture(
            Some(parent),
            Arc::new(RuntimeOpenAiArtifactCapture {
                service: Arc::downgrade(service),
            }),
        )
    }

    /// Suppress a duplicate embedded-frontend parent for one trusted remote
    /// HTTP tunnel. This is intentionally a lease only: it does not register
    /// any lifecycle events and failures stay fail-open for request serving.
    pub(crate) fn suppress_remote_tunneled_request(
        &self,
        request_id: mesh_llm_events::logging::identifiers::RequestId,
    ) -> Option<RawMeshRemoteSuppressionLease> {
        if self.retired.load(Ordering::Acquire) {
            return None;
        }
        let service = self.service.as_ref()?;
        if !service.is_startable() {
            return None;
        }
        RawMeshRemoteSuppressionLease::acquire(Arc::clone(&self.raw_mesh_owners), request_id)
    }

    /// Register one already-parsed management API request with bounded
    /// metadata. Disabled, unavailable, and retired logging stay fail-open.
    pub(crate) fn register_management_request(
        &self,
        request_id: mesh_llm_events::logging::identifiers::RequestId,
        method_route: &'static str,
    ) -> Option<ManagementRequestLifecycle> {
        if self.retired.load(Ordering::Acquire) {
            return None;
        }
        let service = Arc::clone(self.service.as_ref()?);
        if !service.is_startable() {
            return None;
        }
        Some(ManagementRequestLifecycle::register(
            service,
            request_id,
            method_route,
        ))
    }

    /// Record one static operational audit without exposing the service to
    /// producers. Disabled, unavailable, and retired logging remain fail-open
    /// for the calling runtime path. This is the durable seam the CLI audit
    /// bridge uses after logging runtime initialization.
    pub fn write_operational_audit(&self, record: super::service::OperationalAuditRecord) -> bool {
        if self.retired.load(Ordering::Acquire) {
            return false;
        }
        self.service.as_ref().is_some_and(|service| {
            service.is_startable() && service.write_operational_audit(record)
        })
    }

    /// Drain durable logging after a one-shot CLI command.
    ///
    /// The command process exits immediately after dispatch, so its queued
    /// audit envelopes need an explicit bounded durability boundary. Serving
    /// callers do not use this method; their runtime owns normal retirement.
    pub async fn shutdown_for_one_shot_cli(&self) -> bool {
        let Some(service) = self.service.as_ref() else {
            return false;
        };
        service.shutdown().await
    }

    #[cfg(test)]
    pub(crate) fn service_for_test(&self) -> Option<Arc<LoggingService>> {
        self.service.clone()
    }

    /// Apply the only two logging settings whose schema permits live mutation.
    /// A disabled or fail-open runtime has no service, so callers can truthfully
    /// report the config as staged instead of claiming a live update.
    pub fn apply_dynamic_limits(
        &self,
        limits: LoggingDynamicLimits,
    ) -> Result<(), LoggingRuntimeApplyError> {
        let Some(service) = self.service.as_ref() else {
            return Err(LoggingRuntimeApplyError::Unavailable);
        };
        service.apply_dynamic_limits(limits);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn dynamic_limits(&self) -> Option<LoggingDynamicLimits> {
        self.service
            .as_ref()
            .map(|service| service.dynamic_limits())
    }

    fn consume_artifact_capture_health_marker(&self) {
        let Some(capture) = self.artifact_capture.as_ref() else {
            return;
        };
        let Some(marker) = capture.take_health_marker() else {
            return;
        };
        let code = marker.reason().code();
        debug_assert_eq!(code, ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE);

        {
            let mut health = self
                .health
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            health.artifact_capture_available = false;
            health.artifact_capture_ready = false;
            health.artifact_capture_degradation = Some(code);
        }
        self.record_sanitized_health_audit(code);
    }

    fn record_sanitized_health_audit(&self, code: &'static str) {
        let Some(store) = self.store() else {
            return;
        };
        let action = code.to_string();
        let _ = self.health_audit_writer.try_record_error(move || {
            let entry_id = EventId::new().as_uuid().to_string();
            let occurred_at = store.now();
            let _ = store.insert_audit_entry(
                &entry_id,
                None,
                &occurred_at,
                HEALTH_AUDIT_ACTOR,
                &action,
                None,
            );
        });
    }

    #[cfg(test)]
    fn initialize_with_capture_opener_for_test<F>(
        foundation: &LoggingFoundation,
        config: &mesh_llm_config::LoggingConfig,
        open_capture: F,
    ) -> Self
    where
        F: FnOnce(
            PathBuf,
            Arc<dyn StoreClock>,
            Arc<LogStore>,
        ) -> Result<FailOpenArtifactCapture, LogStoreError>,
    {
        Self::initialize_with_capture_opener(foundation, config, open_capture)
    }

    #[cfg(test)]
    pub(crate) fn initialize_with_store_clock_for_test(
        foundation: &LoggingFoundation,
        config: &mesh_llm_config::LoggingConfig,
        clock: Arc<dyn StoreClock>,
    ) -> Self {
        if !foundation.is_healthy() {
            let metadata_state = if config.enabled {
                LoggingMetadataState::StorageUnavailable
            } else {
                LoggingMetadataState::Disabled
            };
            return Self::unavailable(metadata_state);
        }
        Self::initialize_healthy_foundation_with_clock(
            foundation,
            config,
            clock,
            |artifact_root, clock, store| {
                FailOpenArtifactCapture::open(
                    artifact_root,
                    clock,
                    store,
                    canonical_artifact_redactor(),
                )
            },
        )
    }
}

const fn cleanup_worker_state_label(state: CleanupWorkerState) -> &'static str {
    match state {
        CleanupWorkerState::NotStarted => "not_started",
        CleanupWorkerState::Running => "running",
        CleanupWorkerState::Stopping => "stopping",
        CleanupWorkerState::TimedOut => "timed_out",
        CleanupWorkerState::Stopped => "stopped",
    }
}

/// The path-free reason a live config apply could not reach a running service.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoggingRuntimeApplyError {
    Unavailable,
}

impl std::fmt::Display for LoggingRuntimeApplyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("logging runtime is unavailable for live configuration apply")
    }
}

fn canonical_artifact_redactor() -> ArtifactRedactor {
    Arc::new(redact_artifact_bytes)
}

#[cfg(test)]
mod tests;
