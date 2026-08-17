//! Bounded OpenAI artifact commands admitted by [`LoggingService`].
//!
//! This module owns request-path memory bounds, closed unavailable reasons,
//! canonical lifecycle correlation, and the nonblocking queue hand-off. The
//! production sink owns redaction and all filesystem/SQLite work.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use mesh_llm_events::logging::identifiers::{ArtifactId, RequestId};

use super::{LoggingService, PersistenceEntry, offer_persistence_to};

/// Closed reason vocabulary for intentionally omitted artifact content.
/// Values are durable API data, so this type never accepts free-form text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArtifactUnavailableReason {
    StreamingResponseNotAssembled,
    ResponseBodyNotBounded,
    CaptureContentLimitExceeded,
    CaptureMemoryBudgetExceeded,
    ArtifactCaptureDisabled,
    ArtifactCaptureFailed,
}

impl ArtifactUnavailableReason {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::StreamingResponseNotAssembled => "streaming_response_not_assembled",
            Self::ResponseBodyNotBounded => "response_body_not_bounded",
            Self::CaptureContentLimitExceeded => "capture_content_limit_exceeded",
            Self::CaptureMemoryBudgetExceeded => "capture_memory_budget_exceeded",
            Self::ArtifactCaptureDisabled => "artifact_capture_disabled",
            Self::ArtifactCaptureFailed => "artifact_capture_failed",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ArtifactCaptureContent {
    Body(Arc<[u8]>),
    Unavailable(ArtifactUnavailableReason),
}

/// Independent process-local bound for queued artifact body bytes. The main
/// persistence channel is count-bounded, but its configured count can be very
/// large; this budget prevents accepted body commands from multiplying that
/// count into unbounded memory.
pub(super) const DEFAULT_ARTIFACT_MEMORY_BUDGET_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug)]
struct ArtifactMemoryCharge {
    bytes: u64,
    in_flight: Arc<AtomicU64>,
}

impl Drop for ArtifactMemoryCharge {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Clone-safe admission charge retained by every clone of one artifact
/// command. The charge is released exactly once when the final queued/worker
/// owner drops it, including queue eviction and failed `try_send` paths.
#[derive(Clone, Debug)]
pub(crate) struct ArtifactMemoryPermit(Arc<ArtifactMemoryCharge>);

impl ArtifactMemoryPermit {
    fn try_acquire(in_flight: &Arc<AtomicU64>, bytes: usize, limit: u64) -> Option<Self> {
        let bytes = u64::try_from(bytes).ok()?;
        if bytes == 0 {
            return None;
        }
        in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(bytes).filter(|next| *next <= limit)
            })
            .ok()?;
        Some(Self(Arc::new(ArtifactMemoryCharge {
            bytes,
            in_flight: Arc::clone(in_flight),
        })))
    }
}

/// Owned, bounded artifact command accepted by the persistence queue. It
/// contains decoded body bytes only; HTTP headers cannot enter this type.
#[derive(Clone, Debug)]
pub struct ArtifactCaptureEntry {
    pub(crate) artifact_id: String,
    pub(crate) request_id: String,
    pub(crate) kind: &'static str,
    pub(crate) occurred_at: String,
    pub(crate) content: ArtifactCaptureContent,
    pub(crate) media_kind: String,
    pub(crate) version: u32,
    pub(crate) _memory_permit: Option<ArtifactMemoryPermit>,
}

/// Bounded outcome returned by the serial persistence sink.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactPersistenceStatus {
    Written,
    Unavailable,
    FailedUnavailable,
}

impl LoggingService {
    /// Offer decoded request or client-visible response bytes to the bounded
    /// serial persistence owner. This only copies content that fits the
    /// in-memory command bound; larger bodies become explicit metadata-only
    /// artifacts instead of allocating unbounded queue memory.
    pub(crate) fn enqueue_openai_artifact_body(
        &self,
        request_id: RequestId,
        kind: &'static str,
        content: &[u8],
        media_kind: Option<&str>,
    ) {
        let media_kind = validated_media_kind(media_kind, Some(content));
        let (content, memory_permit) = if content.len() > self.config.artifact_command_max_bytes {
            (
                ArtifactCaptureContent::Unavailable(
                    ArtifactUnavailableReason::CaptureContentLimitExceeded,
                ),
                None,
            )
        } else if content.is_empty() {
            (ArtifactCaptureContent::Body(Arc::from(content)), None)
        } else if let Some(permit) = ArtifactMemoryPermit::try_acquire(
            &self.artifact_memory_in_flight,
            content.len(),
            self.artifact_memory_budget_bytes,
        ) {
            (
                ArtifactCaptureContent::Body(Arc::from(content)),
                Some(permit),
            )
        } else {
            (
                ArtifactCaptureContent::Unavailable(
                    ArtifactUnavailableReason::CaptureMemoryBudgetExceeded,
                ),
                None,
            )
        };
        self.enqueue_openai_artifact(request_id, kind, content, media_kind, memory_permit);
    }

    /// Offer a metadata-only record for streaming or otherwise unbounded
    /// client responses. Reasons are closed at the type boundary.
    pub(crate) fn enqueue_openai_artifact_unavailable(
        &self,
        request_id: RequestId,
        kind: &'static str,
        reason: ArtifactUnavailableReason,
    ) {
        self.enqueue_openai_artifact(
            request_id,
            kind,
            ArtifactCaptureContent::Unavailable(reason),
            "application/octet-stream".to_string(),
            None,
        );
    }

    fn enqueue_openai_artifact(
        &self,
        request_id: RequestId,
        kind: &'static str,
        content: ArtifactCaptureContent,
        media_kind: String,
        memory_permit: Option<ArtifactMemoryPermit>,
    ) {
        if self.sink.is_none() || !matches!(kind, "request" | "response") {
            return;
        }
        let request_id_text = request_id.as_uuid().to_string();
        let Some(summary) = self
            .registry
            .get_active(&request_id_text)
            .or_else(|| self.registry.get_recent(&request_id_text))
        else {
            return;
        };
        let entry = ArtifactCaptureEntry {
            artifact_id: ArtifactId::new().as_uuid().to_string(),
            request_id: request_id_text,
            kind,
            // Use the lifecycle service clock at the actual capture boundary.
            // The summary is carried separately to preserve the FK ordering.
            occurred_at: super::canonical_clock_timestamp(self.clock.as_ref()),
            content,
            media_kind,
            version: 1,
            _memory_permit: memory_permit,
        };
        offer_persistence_to(
            &self.delivery,
            &self.persistence_queue_drops,
            &self.persistence_outstanding,
            &self.metrics,
            PersistenceEntry::Artifact { entry, summary },
        );
    }

    #[cfg(test)]
    pub(crate) fn with_artifact_memory_budget_for_test(mut self, bytes: u64) -> Self {
        self.artifact_memory_budget_bytes = bytes;
        self
    }
}

fn validated_media_kind(media_kind: Option<&str>, content: Option<&[u8]>) -> String {
    let Some(essence) = media_kind
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty() && value.len() <= 127)
    else {
        return "application/octet-stream".to_string();
    };
    let mut parts = essence.split('/');
    let valid_token = |token: &str| {
        !token.is_empty()
            && token.bytes().all(|byte| {
                byte.is_ascii_alphanumeric()
                    || matches!(
                        byte,
                        b'!' | b'#' | b'$' | b'&' | b'^' | b'_' | b'.' | b'+' | b'-'
                    )
            })
    };
    match (parts.next(), parts.next(), parts.next()) {
        (Some(top), Some(sub), None) if valid_token(top) && valid_token(sub) => {
            let normalized = essence.to_ascii_lowercase();
            if (normalized == "application/json" || normalized.ends_with("+json"))
                && content
                    .is_some_and(|body| serde_json::from_slice::<serde_json::Value>(body).is_err())
            {
                "application/octet-stream".to_string()
            } else {
                normalized
            }
        }
        _ => "application/octet-stream".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use mesh_llm_events::logging::replay::ReplayChannel;

    use super::*;
    use crate::logging::{
        Clock, LogStoreSink, OperationalAuditRecord, PersistSink, RegistryConfig,
        RequestSummaryEntry, ServiceConfig, SystemClock,
    };

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum Recorded {
        Summary(String),
        Event(String),
        Artifact {
            occurred_at: String,
            content: &'static str,
            media_kind: String,
        },
    }

    #[derive(Default)]
    struct RecordingSink(Mutex<Vec<Recorded>>);

    impl RecordingSink {
        fn records(&self) -> Vec<Recorded> {
            self.0.lock().expect("records mutex").clone()
        }
    }

    #[async_trait]
    impl PersistSink for RecordingSink {
        async fn persist_summary(&self, entry: RequestSummaryEntry) -> Result<(), String> {
            self.0
                .lock()
                .expect("records mutex")
                .push(Recorded::Summary(entry.created_at));
            Ok(())
        }

        async fn persist_event(
            &self,
            _request_id: String,
            _event_id: String,
            _channel: ReplayChannel,
            _sequence: u64,
            occurred_at: String,
            _payload_json: String,
        ) -> Result<(), String> {
            self.0
                .lock()
                .expect("records mutex")
                .push(Recorded::Event(occurred_at));
            Ok(())
        }

        async fn persist_artifact_pointer(
            &self,
            _request_id: String,
            _artifact_data: serde_json::Value,
        ) -> Result<(), String> {
            Ok(())
        }

        async fn persist_artifact_capture(
            &self,
            entry: ArtifactCaptureEntry,
        ) -> Result<ArtifactPersistenceStatus, String> {
            let content = match entry.content {
                ArtifactCaptureContent::Body(_) => "body",
                ArtifactCaptureContent::Unavailable(
                    ArtifactUnavailableReason::CaptureContentLimitExceeded,
                ) => "limit_unavailable",
                ArtifactCaptureContent::Unavailable(_) => "unavailable",
            };
            self.0
                .lock()
                .expect("records mutex")
                .push(Recorded::Artifact {
                    occurred_at: entry.occurred_at,
                    content,
                    media_kind: entry.media_kind,
                });
            Ok(if content == "body" {
                ArtifactPersistenceStatus::Written
            } else {
                ArtifactPersistenceStatus::Unavailable
            })
        }

        async fn persist_proxy_record(&self, _proxy_json: String) -> Result<(), String> {
            Ok(())
        }

        async fn persist_audit_entry(&self, _record: OperationalAuditRecord) -> Result<(), String> {
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

    struct SequenceClock(Mutex<VecDeque<String>>);

    impl SequenceClock {
        fn new(timestamps: &[&str]) -> Self {
            Self(Mutex::new(
                timestamps
                    .iter()
                    .map(|value| (*value).to_string())
                    .collect(),
            ))
        }
    }

    impl Clock for SequenceClock {
        fn now(&self) -> String {
            self.0
                .lock()
                .expect("clock mutex")
                .pop_front()
                .unwrap_or_else(|| "2025-01-01T00:00:00.999999999Z".to_string())
        }
    }

    #[tokio::test]
    async fn artifact_follows_summary_and_admitted_with_a_service_clock_timestamp() {
        let sink = Arc::new(RecordingSink::default());
        let service = LoggingService::new(
            ServiceConfig::default(),
            Arc::clone(&sink) as Arc<dyn PersistSink>,
            Box::new(SequenceClock::new(&[
                "2025-01-01T00:00:00.000000000Z",
                "2025-01-01T00:00:00.100000000Z",
                "2025-01-01T00:00:00.200000000Z",
            ])),
        );
        let request_id = RequestId::new();
        let (_guard, _) = service.register_request(request_id);
        service.enqueue_openai_artifact_body(request_id, "request", br#"{"model":"safe"}"#, None);
        service.enqueue_openai_artifact_body(
            request_id,
            "response",
            br#"{"id":"safe"}"#,
            Some("Application/JSON; Charset=UTF-8"),
        );

        assert_eq!(service.pump_sync().await, 4);
        let records = sink.records();
        assert!(matches!(records[0], Recorded::Summary(_)));
        assert!(matches!(records[1], Recorded::Event(_)));
        assert!(matches!(records[2], Recorded::Summary(_)));
        let (summary_at, admitted_at, request_at, response_at, response_media_kind) =
            match (&records[0], &records[1], &records[3], &records[5]) {
                (
                    Recorded::Summary(summary_at),
                    Recorded::Event(admitted_at),
                    Recorded::Artifact {
                        occurred_at: request_at,
                        ..
                    },
                    Recorded::Artifact {
                        occurred_at: response_at,
                        media_kind,
                        ..
                    },
                ) => (summary_at, admitted_at, request_at, response_at, media_kind),
                records => panic!("unexpected persistence order: {records:?}"),
            };
        assert_eq!(summary_at, admitted_at);
        assert_eq!(request_at, "2025-01-01T00:00:00.100000000Z");
        assert_eq!(response_at, "2025-01-01T00:00:00.200000000Z");
        assert_ne!(summary_at, response_at);
        assert!(response_at > request_at);
        assert_eq!(response_media_kind, "application/json");
    }

    #[tokio::test]
    async fn oversized_body_becomes_metadata_only_before_queue_copy() {
        let sink = Arc::new(RecordingSink::default());
        let service = LoggingService::new(
            ServiceConfig {
                artifact_command_max_bytes: 4,
                ..ServiceConfig::default()
            },
            Arc::clone(&sink) as Arc<dyn PersistSink>,
            Box::new(SystemClock),
        );
        let request_id = RequestId::new();
        let (_guard, _) = service.register_request(request_id);
        service.enqueue_openai_artifact_body(request_id, "response", b"12345", None);
        assert_eq!(service.pump_sync().await, 3);
        assert!(sink.records().iter().any(|record| matches!(
            record,
            Recorded::Artifact {
                content: "limit_unavailable",
                ..
            }
        )));
    }

    #[test]
    fn invalid_or_binary_json_content_is_labeled_opaque() {
        assert_eq!(
            validated_media_kind(Some("application/json"), Some(&[0xff, 0x00])),
            "application/octet-stream"
        );
        assert_eq!(
            validated_media_kind(Some("application/json"), Some(br#"{"ok":true}"#)),
            "application/json"
        );
        assert_eq!(
            validated_media_kind(Some("not a type"), Some(b"safe")),
            "application/octet-stream"
        );
    }

    #[tokio::test]
    async fn aggregate_memory_budget_saturates_fail_open_and_releases_after_persist() {
        let sink = Arc::new(RecordingSink::default());
        let service = LoggingService::new(
            ServiceConfig {
                artifact_command_max_bytes: 8,
                ..ServiceConfig::default()
            },
            Arc::clone(&sink) as Arc<dyn PersistSink>,
            Box::new(SystemClock),
        )
        .with_artifact_memory_budget_for_test(4);
        let request_id = RequestId::new();
        let (_guard, _) = service.register_request(request_id);
        service.enqueue_openai_artifact_body(request_id, "request", b"1234", None);
        service.enqueue_openai_artifact_body(request_id, "response", b"5", None);
        assert_eq!(service.artifact_memory_in_flight.load(Ordering::Acquire), 4);

        assert_eq!(service.pump_sync().await, 4);
        assert_eq!(service.artifact_memory_in_flight.load(Ordering::Acquire), 0);
        assert!(sink.records().iter().any(|record| matches!(
            record,
            Recorded::Artifact {
                content: "unavailable",
                ..
            }
        )));
    }

    #[test]
    fn queue_eviction_releases_artifact_memory_permit() {
        let sink = Arc::new(RecordingSink::default());
        let service = LoggingService::new(
            ServiceConfig {
                queue_capacity: 1,
                artifact_command_max_bytes: 8,
                ..ServiceConfig::default()
            },
            Arc::clone(&sink) as Arc<dyn PersistSink>,
            Box::new(SystemClock),
        )
        .with_artifact_memory_budget_for_test(4);
        let request_id = RequestId::new();
        let (_guard, _) = service.register_request(request_id);

        service.enqueue_openai_artifact_body(request_id, "request", b"1234", None);
        assert_eq!(service.artifact_memory_in_flight.load(Ordering::Acquire), 4);

        // The second artifact cannot acquire memory, so it is admitted as
        // metadata-only and evicts the first body from the one-entry queue.
        service.enqueue_openai_artifact_body(request_id, "response", b"5", None);
        assert_eq!(service.artifact_memory_in_flight.load(Ordering::Acquire), 0);
        assert!(service.persistence_queue_drops() >= 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ingress_enqueue_returns_while_artifact_worker_is_blocked() {
        let root = tempfile::tempdir().expect("temporary log root");
        let store = Arc::new(
            mesh_llm_log_store::LogStore::open(
                root.path(),
                Arc::new(mesh_llm_log_store::RealClock),
            )
            .expect("open log store"),
        );
        let capture = Arc::new(
            mesh_llm_log_store::FailOpenArtifactCapture::open(
                root.path().join("artifacts"),
                Arc::new(mesh_llm_log_store::RealClock),
                Arc::clone(&store),
                Arc::new(crate::logging::policy::redact_artifact_bytes),
            )
            .expect("open artifact capture"),
        );
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let first = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let hook = Arc::new(move || {
            if first.swap(false, std::sync::atomic::Ordering::AcqRel) {
                started_tx.send(()).expect("report blocked worker");
                release_rx
                    .lock()
                    .expect("release mutex")
                    .recv()
                    .expect("release blocked worker");
            }
        });
        let sink = LogStoreSink::new(Arc::clone(&store))
            .with_artifact_blocking_hook_for_test(hook)
            .with_artifact_capture(capture, 4_096, 8_192);
        let service = LoggingService::new(
            ServiceConfig {
                registry_config: RegistryConfig::default(),
                ..ServiceConfig::default()
            },
            Arc::new(sink),
            Box::new(SystemClock),
        );
        assert!(service.spawn());
        let request_id = RequestId::new();
        let (_guard, _) = service.register_request(request_id);
        service.enqueue_openai_artifact_body(request_id, "request", br#"{"model":"safe"}"#, None);
        tokio::task::spawn_blocking(move || started_rx.recv_timeout(Duration::from_secs(2)))
            .await
            .expect("start observer joins")
            .expect("worker reached blocking hook");

        // The artifact-specific worker hook remains blocked, but the same
        // current-thread ingress producer can immediately enqueue a response.
        service.enqueue_openai_artifact_body(
            request_id,
            "response",
            br#"{"id":"safe"}"#,
            Some("application/json"),
        );
        assert!(service.persistence_outstanding() >= 2);
        release_tx.send(()).expect("release worker");
        assert!(service.shutdown().await);

        let page = store
            .query_artifacts(
                &request_id.as_uuid().to_string(),
                &mesh_llm_log_store::PageQuery {
                    limit: 10,
                    cursor: None,
                    sort: mesh_llm_log_store::QuerySort::Ascending,
                },
            )
            .expect("query captured request");
        assert_eq!(page.items.len(), 2);
    }

    #[tokio::test]
    async fn artifact_write_failure_persists_truthful_unavailable_metadata() {
        let root = tempfile::tempdir().expect("temporary log root");
        let store = Arc::new(
            mesh_llm_log_store::LogStore::open(
                root.path(),
                Arc::new(mesh_llm_log_store::RealClock),
            )
            .expect("open log store"),
        );
        let artifact_root = root.path().join("artifacts-failure");
        let capture = Arc::new(
            mesh_llm_log_store::FailOpenArtifactCapture::open(
                artifact_root.clone(),
                Arc::new(mesh_llm_log_store::RealClock),
                Arc::clone(&store),
                Arc::new(crate::logging::policy::redact_artifact_bytes),
            )
            .expect("open artifact capture"),
        );
        std::fs::remove_dir(&artifact_root).expect("remove empty artifact root");
        let request_id = RequestId::new().as_uuid().to_string();
        let occurred_at = "2025-01-01T00:00:00.000000000Z";
        store
            .upsert_summary_metadata(&request_id, None, None, None, None, occurred_at)
            .expect("summary");
        let sink =
            LogStoreSink::new(Arc::clone(&store)).with_artifact_capture(capture, 4_096, 8_192);
        let status = sink
            .persist_artifact_capture(ArtifactCaptureEntry {
                artifact_id: ArtifactId::new().as_uuid().to_string(),
                request_id: request_id.clone(),
                kind: "request",
                occurred_at: occurred_at.to_string(),
                content: ArtifactCaptureContent::Body(Arc::from(br#"{"model":"safe"}"#.as_slice())),
                media_kind: "application/octet-stream".to_string(),
                version: 1,
                _memory_permit: None,
            })
            .await
            .expect("failure metadata remains durable");
        assert_eq!(status, ArtifactPersistenceStatus::FailedUnavailable);
        let page = store
            .query_artifacts(
                &request_id,
                &mesh_llm_log_store::PageQuery {
                    limit: 10,
                    cursor: None,
                    sort: mesh_llm_log_store::QuerySort::Ascending,
                },
            )
            .expect("query failed capture metadata");
        assert_eq!(
            page.items[0].unavailable_reason.as_deref(),
            Some("artifact_capture_failed")
        );
        assert!(page.items[0].checksum.is_none());
        assert_eq!(page.items[0].bytes, 0);
    }
}
