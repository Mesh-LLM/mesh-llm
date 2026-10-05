use mesh_llm_log_store::{ArtifactPrivacy, MaintenanceExecutionControl};
use std::path::Path;

use super::*;
use crate::logging::{CallerPathType, TerminalOutcome};

mod artifact_capture;
mod remote_caller_attribution;
mod service_lifecycle;

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

struct FixedStoreClock(&'static str);

impl StoreClock for FixedStoreClock {
    fn now(&self) -> String {
        self.0.to_string()
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
fn remote_tunnel_suppression_merges_authenticated_caller_into_existing_parent() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(true, Some(&root.path().to_path_buf()));
    let state = Arc::new(LoggingRuntimeState::initialize(
        &foundation,
        &mesh_llm_config::LoggingConfig::default(),
    ));
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let mut attachment = state.openai_ingress_attachment(
        request_id,
        RequestSummaryMetadata::from_openai_ingress_path("/v1/chat/completions")
            .with_source(Some("direct_http"))
            .with_method(Some("POST"))
            .with_caller_identity(
                None,
                Some("127.0.0.1:40123"),
                Some(CallerPathType::LocalHttp),
            ),
    );
    let endpoint_id = "61".repeat(32);
    let attribution = state
        .attribute_remote_tunneled_request(
            request_id,
            RequestSummaryMetadata::default().with_caller_identity(
                Some(&endpoint_id),
                Some("192.0.2.61:11204"),
                Some(CallerPathType::RemoteQuicHttp),
            ),
        )
        .expect("remote attribution lease");
    let lease = state
        .suppress_remote_tunneled_request(request_id)
        .expect("remote suppression lease");

    let active = state
        .service
        .as_ref()
        .expect("logging service")
        .registry_ref()
        .get_active(&request_id.as_uuid().to_string())
        .expect("active raw mesh parent");
    assert_eq!(
        active.metadata.caller_endpoint_id(),
        Some(endpoint_id.as_str())
    );
    assert_eq!(active.metadata.caller_addr(), Some("192.0.2.61:11204"));
    assert_eq!(active.metadata.caller_path_type(), Some("remote_quic_http"));
    assert_eq!(active.metadata.route(), Some("chat_completions"));
    assert_eq!(active.metadata.source(), Some("direct_http"));
    assert_eq!(active.metadata.method(), Some("POST"));

    drop(lease);
    drop(attribution);
    attachment.terminal(TerminalOutcome::Completed);
}

#[test]
fn unavailable_foundation_is_sanitized_and_fail_open() {
    let root = tempfile::tempdir().expect("temporary logging root");
    let foundation = LoggingFoundation::init(false, Some(&root.path().to_path_buf()));
    let config = mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    };
    let state = LoggingRuntimeState::initialize(&foundation, &config);

    assert_eq!(state.health(), LoggingRuntimeHealth::unavailable());
    assert!(state.store().is_none());
    assert_eq!(
        state.status(),
        LoggingRuntimeStatus {
            metadata_available: false,
            metadata_state: "disabled",
            schema_version: None,
            supported_schema_version: None,
            capture_mode: "unavailable",
            artifact_capture_available: false,
            artifact_capture_ready: false,
            artifact_capture_degradation: None,
            persistence_worker_state: "unavailable",
            persistence_queue_drops: 0,
            persistence_failures: 0,
            persistence_shutdown_losses: 0,
            persistence_outstanding: 0,
            cleanup_worker_state: "not_started",
            cleanup_shutdown_timeouts: 0,
            cleanup_last_outcome: None,
            cleanup_last_deleted_count: None,
        }
    );
    assert_eq!(
        state.apply_dynamic_limits(LoggingDynamicLimits {
            retention_ttl_secs: 7_200,
            replay_capacity: 256,
        }),
        Err(LoggingRuntimeApplyError::Unavailable)
    );
}

#[test]
fn incompatible_schema_is_reported_without_exposing_storage_details() {
    let incompatible_version = mesh_llm_log_store::LOG_STORE_SCHEMA_VERSION + 1;
    let state = LoggingRuntimeState::unavailable(LoggingMetadataState::SchemaIncompatible {
        found: incompatible_version,
        supported: mesh_llm_log_store::LOG_STORE_SCHEMA_VERSION,
    });
    let status = state.status();

    assert!(!status.metadata_available);
    assert_eq!(status.metadata_state, "schema_incompatible");
    assert_eq!(status.schema_version, Some(incompatible_version));
    assert_eq!(
        status.supported_schema_version,
        Some(mesh_llm_log_store::LOG_STORE_SCHEMA_VERSION)
    );
    assert!(state.store().is_none());
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
