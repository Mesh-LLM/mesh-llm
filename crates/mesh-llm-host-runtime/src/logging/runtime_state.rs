//! Host-owned durable logging state.
//!
//! This is intentionally a narrow startup boundary. It opens the durable
//! metadata store and the independently fail-open artifact capture facade,
//! but it does not start workers or instrument request producers.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use mesh_llm_events::logging::identifiers::EventId;
use mesh_llm_log_store::{
    ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE, ArtifactCaptureDisabledReason,
    ArtifactCaptureOutcome, ArtifactRedactor, Clock as StoreClock, FailOpenArtifactCapture,
    LogStore, LogStoreError, RealClock,
};

use super::foundation::LoggingFoundation;
use super::policy::PolicyCaptureMode;
use super::policy::redact_artifact_bytes;
use super::writer::FailOpenWriter;
use super::{LogStoreSink, LoggingDynamicLimits, LoggingService, ServiceConfig, SystemClock};

const HEALTH_AUDIT_ACTOR: &str = "logging-runtime";

/// Internal capability state for local logging storage.
///
/// The only artifact-degradation value exposed from this state is the stable,
/// path-free circuit-breaker code. Errors and filesystem locations remain
/// private to the storage implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoggingRuntimeHealth {
    pub metadata_available: bool,
    pub artifact_capture_available: bool,
    pub artifact_capture_degradation: Option<&'static str>,
}

impl LoggingRuntimeHealth {
    fn unavailable() -> Self {
        Self {
            metadata_available: false,
            artifact_capture_available: false,
            artifact_capture_degradation: None,
        }
    }
}

/// A fully specified artifact write owned by the logging runtime.
///
/// Keeping this request at the boundary prevents callers from reaching into
/// the fail-open capture object and forgetting to publish its one-shot health
/// marker after a privacy failure.
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

/// The process-local durable logging resources created from a foundation.
pub struct LoggingRuntimeState {
    store: Option<Arc<LogStore>>,
    service: Option<Arc<LoggingService>>,
    artifact_capture: Option<FailOpenArtifactCapture>,
    artifact_limits: Option<(usize, usize)>,
    health: Mutex<LoggingRuntimeHealth>,
    health_audit_writer: FailOpenWriter,
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
            return Self::unavailable();
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
        let Some(store) = Self::open_metadata_store(foundation, &clock) else {
            return Self::unavailable();
        };
        let policy = super::policy::build_policy(config);
        let (artifact_capture, artifact_limits) = match policy.capture_mode {
            PolicyCaptureMode::MetadataOnly => (None, None),
            PolicyCaptureMode::RedactedWithLimits(byte_limit, aggregate_limit) => (
                Self::open_artifact_capture(foundation, clock, &store, open_capture),
                Some((byte_limit, aggregate_limit)),
            ),
        };
        Self::from_open_store(store, artifact_capture, artifact_limits, config)
    }

    fn open_metadata_store(
        foundation: &LoggingFoundation,
        clock: &Arc<dyn StoreClock>,
    ) -> Option<Arc<LogStore>> {
        match LogStore::open(foundation.store_dir(), Arc::clone(clock)) {
            Ok(store) => Some(Arc::new(store)),
            Err(_) => {
                tracing::warn!("Logging durable storage unavailable; continuing without logging");
                None
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
        artifact_capture: Option<FailOpenArtifactCapture>,
        artifact_limits: Option<(usize, usize)>,
        config: &mesh_llm_config::LoggingConfig,
    ) -> Self {
        let policy = super::policy::build_policy(config);
        let artifact_capture_available = artifact_capture
            .as_ref()
            .is_some_and(|capture| !capture.is_disabled());
        let state = Self {
            service: Some(Arc::new(LoggingService::new_with_dynamic_limits(
                ServiceConfig::from_policy(&policy),
                Arc::new(LogStoreSink::new(Arc::clone(&store))),
                Box::new(SystemClock),
                LoggingDynamicLimits::from_config(config),
            ))),
            store: Some(store),
            artifact_capture,
            artifact_limits,
            health: Mutex::new(LoggingRuntimeHealth {
                metadata_available: true,
                artifact_capture_available,
                artifact_capture_degradation: None,
            }),
            health_audit_writer: FailOpenWriter::new(),
        };
        state.consume_artifact_capture_health_marker();
        state
    }

    fn unavailable() -> Self {
        Self {
            store: None,
            service: None,
            artifact_capture: None,
            artifact_limits: None,
            health: Mutex::new(LoggingRuntimeHealth::unavailable()),
            health_audit_writer: FailOpenWriter::new(),
        }
    }

    /// Return the internal health/capability projection without filesystem details.
    pub fn health(&self) -> LoggingRuntimeHealth {
        *self
            .health
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Attempt an artifact write while publishing a one-time privacy failure.
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
        self.consume_artifact_capture_health_marker();
        outcome
    }

    /// Access to the typed store stays internal to host runtime ownership.
    pub(crate) fn store(&self) -> Option<Arc<LogStore>> {
        self.store.clone()
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
mod tests {
    use mesh_llm_log_store::ArtifactPrivacy;
    use std::path::Path;

    use super::*;

    #[derive(Default)]
    struct RejectPrivacy;

    impl ArtifactPrivacy for RejectPrivacy {
        fn prepare_directory(&self, _path: &Path) -> Result<(), LogStoreError> {
            Err(LogStoreError::PrivacyNotGuaranteed)
        }

        fn prepare_file(&self, _path: &Path) -> Result<(), LogStoreError> {
            Err(LogStoreError::PrivacyNotGuaranteed)
        }
    }

    #[derive(Default)]
    struct RejectArtifactFiles;

    impl ArtifactPrivacy for RejectArtifactFiles {
        fn prepare_directory(&self, _path: &Path) -> Result<(), LogStoreError> {
            Ok(())
        }

        fn prepare_file(&self, _path: &Path) -> Result<(), LogStoreError> {
            Err(LogStoreError::PrivacyNotGuaranteed)
        }
    }

    fn marker_audit_count(store: &LogStore) -> i64 {
        store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM audit_entries WHERE action = ?",
                [ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE],
                |row| row.get(0),
            )
            .expect("count marker audits")
    }

    fn artifact_config() -> mesh_llm_config::LoggingConfig {
        let mut config = mesh_llm_config::LoggingConfig::default();
        config.artifact.capture_mode = mesh_llm_config::CaptureMode::RedactedArtifacts;
        config.artifact.byte_limit_bytes = 4_096;
        config.artifact.aggregate_limit_bytes = 8_192;
        config
    }

    #[test]
    fn metadata_only_never_opens_or_writes_artifact_capture() {
        let root = tempfile::tempdir().expect("temporary logging root");
        let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
        let opened = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let opened_by_callback = Arc::clone(&opened);
        let state = LoggingRuntimeState::initialize_with_capture_opener_for_test(
            &foundation,
            &mesh_llm_config::LoggingConfig::default(),
            move |_, _, _| {
                opened_by_callback.store(true, std::sync::atomic::Ordering::SeqCst);
                Err(LogStoreError::PrivacyNotGuaranteed)
            },
        );
        assert!(!opened.load(std::sync::atomic::Ordering::SeqCst));
        let outcome = state
            .write_artifact(ArtifactCaptureRequest {
                artifact_id: "metadata-only",
                request_id: "request",
                kind: "body",
                occurred_at: "2025-01-01T00:00:00Z",
                content: b"must not be stored",
                media_kind: Some("text/plain"),
                version: 1,
                truncated: false,
            })
            .unwrap();
        assert!(matches!(outcome, ArtifactCaptureOutcome::Disabled(_)));
        assert_eq!(
            std::fs::read_dir(foundation.artifact_dir())
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn privacy_failure_disables_only_artifacts_and_records_one_sanitized_marker() {
        let root = tempfile::tempdir().expect("temporary logging root");
        let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
        let state = LoggingRuntimeState::initialize_with_capture_opener_for_test(
            &foundation,
            &artifact_config(),
            |artifact_root, clock, store| {
                FailOpenArtifactCapture::open_with_privacy(
                    artifact_root,
                    clock,
                    store,
                    canonical_artifact_redactor(),
                    Arc::new(RejectPrivacy),
                )
            },
        );

        assert_eq!(
            state.health(),
            LoggingRuntimeHealth {
                metadata_available: true,
                artifact_capture_available: false,
                artifact_capture_degradation: Some(ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE,),
            }
        );
        let store = state.store().expect("metadata store remains available");
        assert_eq!(marker_audit_count(&store), 1);

        store
            .insert_summary(
                "metadata-request",
                None,
                None,
                None,
                None,
                &store.now(),
                None,
                None,
                None,
            )
            .expect("metadata summary insert");
        store
            .insert_audit_entry(
                "metadata-audit",
                Some("metadata-request"),
                &store.now(),
                "test",
                "metadata_still_available",
                None,
            )
            .expect("metadata audit insert");

        let outcome = state
            .write_artifact(ArtifactCaptureRequest {
                artifact_id: "artifact-after-disable",
                request_id: "metadata-request",
                kind: "request_body",
                occurred_at: &store.now(),
                content: b"redacted",
                media_kind: Some("text/plain"),
                version: 1,
                truncated: false,
            })
            .expect("disabled capture is fail-open");
        assert!(matches!(outcome, ArtifactCaptureOutcome::Disabled(_)));
        let repeated_outcome = state
            .write_artifact(ArtifactCaptureRequest {
                artifact_id: "artifact-after-disable-again",
                request_id: "metadata-request",
                kind: "request_body",
                occurred_at: &store.now(),
                content: b"redacted",
                media_kind: Some("text/plain"),
                version: 1,
                truncated: false,
            })
            .expect("repeated disabled capture is fail-open");
        assert!(matches!(
            repeated_outcome,
            ArtifactCaptureOutcome::Disabled(_)
        ));
        assert_eq!(marker_audit_count(&store), 1);
        assert!(store.get_summary("metadata-request").unwrap().is_some());
        let total_audits: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM audit_entries", [], |row| row.get(0))
            .expect("count all audits");
        assert_eq!(total_audits, 2);
    }

    #[test]
    fn write_time_privacy_failure_publishes_one_marker_and_keeps_metadata_available() {
        let root = tempfile::tempdir().expect("temporary logging root");
        let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
        let state = LoggingRuntimeState::initialize_with_capture_opener_for_test(
            &foundation,
            &artifact_config(),
            |artifact_root, clock, store| {
                FailOpenArtifactCapture::open_with_privacy(
                    artifact_root,
                    clock,
                    store,
                    canonical_artifact_redactor(),
                    Arc::new(RejectArtifactFiles),
                )
            },
        );
        let store = state.store().expect("metadata store available");
        store
            .insert_summary(
                "write-time-request",
                None,
                None,
                None,
                None,
                &store.now(),
                None,
                None,
                None,
            )
            .expect("metadata summary insert");

        let occurred_at = store.now();
        let write = |artifact_id| ArtifactCaptureRequest {
            artifact_id,
            request_id: "write-time-request",
            kind: "request_body",
            occurred_at: &occurred_at,
            content: b"redacted",
            media_kind: Some("text/plain"),
            version: 1,
            truncated: false,
        };
        assert!(matches!(
            state.write_artifact(write("write-time-artifact")).unwrap(),
            ArtifactCaptureOutcome::Disabled(_)
        ));
        assert!(matches!(
            state
                .write_artifact(write("write-time-artifact-again"))
                .unwrap(),
            ArtifactCaptureOutcome::Disabled(_)
        ));

        assert_eq!(
            state.health().artifact_capture_degradation,
            Some(ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE)
        );
        assert!(state.health().metadata_available);
        assert_eq!(marker_audit_count(&store), 1);
    }

    #[test]
    fn raw_artifact_bytes_are_redacted_before_the_capture_boundary() {
        let root = tempfile::tempdir().expect("temporary logging root");
        let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
        let state = LoggingRuntimeState::initialize(&foundation, &artifact_config());
        let store = state.store().expect("metadata store available");
        store
            .insert_summary(
                "raw-artifact-request",
                None,
                None,
                None,
                None,
                &store.now(),
                None,
                None,
                None,
            )
            .expect("metadata summary insert");

        let secret = b"Bearer super-secret-token-0123456789";
        let outcome = state
            .write_artifact(ArtifactCaptureRequest {
                artifact_id: "raw-artifact",
                request_id: "raw-artifact-request",
                kind: "request_body",
                occurred_at: &store.now(),
                content: secret,
                media_kind: Some("text/plain"),
                version: 1,
                truncated: false,
            })
            .expect("artifact write");
        assert!(matches!(
            outcome,
            ArtifactCaptureOutcome::Written(receipt) if receipt.redacted
        ));

        let stored = std::fs::read(
            foundation
                .artifact_dir()
                .join("raw-artifact-request")
                .join("raw-artifact"),
        )
        .expect("read stored artifact");
        assert!(!stored.windows(secret.len()).any(|window| window == secret));
        assert_eq!(stored, b"[REDACTED]");
        assert!(
            store
                .get_artifact_pointer("raw-artifact")
                .expect("artifact pointer")
                .expect("pointer present")
                .redacted
        );
    }

    #[test]
    fn unavailable_foundation_is_sanitized_and_fail_open() {
        let root = tempfile::tempdir().expect("temporary logging root");
        let foundation = LoggingFoundation::init(false, Some(&root.path().to_path_buf()));
        let state = LoggingRuntimeState::initialize(&foundation, &Default::default());

        assert_eq!(state.health(), LoggingRuntimeHealth::unavailable());
        assert!(state.store().is_none());
        assert_eq!(
            state.apply_dynamic_limits(LoggingDynamicLimits {
                retention_ttl_secs: 7_200,
                replay_capacity: 256,
            }),
            Err(LoggingRuntimeApplyError::Unavailable)
        );
    }

    #[test]
    fn store_open_failure_is_fail_open_without_exposing_the_failed_path() {
        let root = tempfile::tempdir().expect("temporary logging root");
        let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
        std::fs::remove_dir_all(foundation.store_dir()).expect("remove store directory");
        std::fs::write(foundation.store_dir(), b"not a directory").expect("block store root");

        let state = LoggingRuntimeState::initialize(&foundation, &Default::default());

        assert_eq!(state.health(), LoggingRuntimeHealth::unavailable());
        assert!(state.store().is_none());
    }

    #[test]
    fn applies_retention_and_replay_limits_together_to_the_installed_service() {
        let root = tempfile::tempdir().expect("temporary logging root");
        let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
        let initial = mesh_llm_config::LoggingConfig {
            retention_ttl_secs: 3_600,
            replay_capacity: 4,
            ..Default::default()
        };
        let state = LoggingRuntimeState::initialize(&foundation, &initial);
        assert_eq!(
            state.dynamic_limits(),
            Some(LoggingDynamicLimits {
                retention_ttl_secs: 3_600,
                replay_capacity: 4,
            })
        );

        let next = LoggingDynamicLimits {
            retention_ttl_secs: 7_200,
            replay_capacity: 2,
        };
        state.apply_dynamic_limits(next).expect("healthy runtime");
        assert_eq!(state.dynamic_limits(), Some(next));
    }
}
