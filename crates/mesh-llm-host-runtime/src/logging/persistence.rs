//! Durable `PersistSink` implementation backed by the typed SQLite repositories.

use std::sync::Arc;

use async_trait::async_trait;
use mesh_llm_events::logging::envelope::CanonicalEnvelope;
use mesh_llm_events::logging::events::LifecycleEvent;
use mesh_llm_events::logging::identifiers::EventId;
use mesh_llm_events::logging::proxy::ProxyRecord;
use mesh_llm_events::logging::replay::ReplayChannel;
use mesh_llm_log_store::{
    ArtifactCaptureOutcome, FailOpenArtifactCapture, LogStore, LogStoreError,
    UnavailableArtifactPointer,
};

use super::policy::{apply_redaction, sanitize_lifecycle_event, sanitize_paths_in_text};
use super::registry::RequestSummaryEntry;
use super::service::{
    ArtifactCaptureContent, ArtifactCaptureEntry, ArtifactPersistenceStatus,
    ArtifactUnavailableReason, OperationalAuditRecord, PersistSink, validate_proxy_record,
};

#[derive(Clone)]
struct TerminalWebhookEnqueue {
    max_attempts: u32,
}

/// Production persistence adapter for the logging service's typed LogStore.
///
/// It deliberately writes summaries and lifecycle envelopes through their
/// dedicated repositories. Operational audit records use their own sink method
/// and never share the lifecycle tables.
pub struct LogStoreSink {
    store: Arc<LogStore>,
    terminal_webhook_enqueue: Option<TerminalWebhookEnqueue>,
    artifact_capture: Option<Arc<FailOpenArtifactCapture>>,
    artifact_limits: Option<(usize, usize)>,
    #[cfg(test)]
    before_blocking_operation: Option<Arc<dyn Fn() + Send + Sync>>,
    #[cfg(test)]
    before_artifact_capture: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl LogStoreSink {
    pub fn new(store: Arc<LogStore>) -> Self {
        Self {
            store,
            terminal_webhook_enqueue: None,
            artifact_capture: None,
            artifact_limits: None,
            #[cfg(test)]
            before_blocking_operation: None,
            #[cfg(test)]
            before_artifact_capture: None,
        }
    }

    /// Opt into a durable terminal-delivery outbox record committed atomically
    /// with its parent terminal event. This does not start delivery work or
    /// place any HTTP work on the request path.
    pub(crate) fn with_terminal_webhook_enqueue(store: Arc<LogStore>, max_attempts: u32) -> Self {
        Self {
            store,
            terminal_webhook_enqueue: Some(TerminalWebhookEnqueue { max_attempts }),
            artifact_capture: None,
            artifact_limits: None,
            #[cfg(test)]
            before_blocking_operation: None,
            #[cfg(test)]
            before_artifact_capture: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_blocking_hook_for_test(
        store: Arc<LogStore>,
        before_blocking_operation: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        Self {
            store,
            terminal_webhook_enqueue: None,
            artifact_capture: None,
            artifact_limits: None,
            before_blocking_operation: Some(before_blocking_operation),
            before_artifact_capture: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_artifact_blocking_hook_for_test(
        mut self,
        before_artifact_capture: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        self.before_artifact_capture = Some(before_artifact_capture);
        self
    }

    /// Install the artifact writer and policy limits on the production sink.
    /// The same confined capture handle may be cloned by the trusted query
    /// facade for reads, but all writes flow through this serial worker.
    pub(crate) fn with_artifact_capture(
        mut self,
        capture: Arc<FailOpenArtifactCapture>,
        byte_limit: usize,
        aggregate_limit: usize,
    ) -> Self {
        self.artifact_capture = Some(capture);
        self.artifact_limits = Some((byte_limit, aggregate_limit));
        self
    }

    fn map_error(error: LogStoreError) -> String {
        error.to_string()
    }

    /// Run the synchronous rusqlite repository operation on Tokio's bounded
    /// blocking pool. The logging service awaits these operations one at a
    /// time, so this is a serialized hand-off rather than per-entry task
    /// fan-out. In particular, SQLite's 30 second busy timeout can never park
    /// a shared async executor worker.
    async fn run_blocking<T, F>(&self, operation: F) -> Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce(Arc<LogStore>) -> Result<T, LogStoreError> + Send + 'static,
    {
        let store = Arc::clone(&self.store);
        #[cfg(test)]
        let before_blocking_operation = self.before_blocking_operation.clone();
        tokio::task::spawn_blocking(move || {
            #[cfg(test)]
            if let Some(hook) = before_blocking_operation {
                hook();
            }
            operation(store)
        })
        .await
        .map_err(|error| format!("logging sqlite worker failed: {error}"))?
        .map_err(Self::map_error)
    }
}

#[async_trait]
impl PersistSink for LogStoreSink {
    async fn persist_summary(&self, entry: RequestSummaryEntry) -> Result<(), String> {
        self.run_blocking(move |store| {
            store.upsert_summary_metadata_with_caller(
                &entry.request_id,
                entry.metadata.model(),
                entry.metadata.route(),
                entry.metadata.provider(),
                entry.metadata.engine(),
                entry.metadata.caller_endpoint_id(),
                entry.metadata.caller_addr(),
                entry.metadata.caller_path_type(),
                &entry.created_at,
            )
        })
        .await
    }

    async fn persist_event(
        &self,
        request_id: String,
        event_id: String,
        _channel: ReplayChannel,
        _sequence: u64,
        occurred_at: String,
        payload_json: String,
    ) -> Result<(), String> {
        let mut envelope = CanonicalEnvelope::from_json_str(&payload_json)
            .map_err(|error| format!("invalid canonical lifecycle envelope: {error}"))?;
        if envelope.request_id.as_uuid().to_string() != request_id
            || envelope.event_id.as_uuid().to_string() != event_id
            || envelope.occurred_at != occurred_at
        {
            return Err("canonical lifecycle envelope does not match persistence key".to_string());
        }

        envelope.event = sanitize_lifecycle_event(envelope.event);
        let payload_json = serde_json::to_string(&envelope)
            .map_err(|error| format!("serialize sanitized lifecycle envelope: {error}"))?;

        match terminal_intent(&envelope.event) {
            Some((status, terminal_status_code)) => {
                let terminal_webhook_enqueue = self.terminal_webhook_enqueue.clone();
                self.run_blocking(move |store| {
                    if let Some(enqueue) = terminal_webhook_enqueue {
                        let delivery_id = webhook_delivery_id(&event_id);
                        store.write_terminal_event_with_webhook(
                            &request_id,
                            &event_id,
                            &payload_json,
                            status,
                            terminal_status_code,
                            &occurred_at,
                            &delivery_id,
                            enqueue.max_attempts,
                        )?;
                    } else {
                        store.write_terminal_event(
                            &request_id,
                            &event_id,
                            &payload_json,
                            status,
                            terminal_status_code,
                            &occurred_at,
                        )?;
                    }
                    Ok(())
                })
                .await
            }
            None => {
                self.run_blocking(move |store| {
                    store.insert_lifecycle_event(
                        &request_id,
                        &event_id,
                        &payload_json,
                        &occurred_at,
                    )
                })
                .await
            }
        }
    }

    async fn persist_artifact_pointer(
        &self,
        _request_id: String,
        _artifact_data: serde_json::Value,
    ) -> Result<(), String> {
        Err("artifact persistence is not wired by the lifecycle service".to_string())
    }

    async fn persist_artifact_capture(
        &self,
        entry: ArtifactCaptureEntry,
    ) -> Result<ArtifactPersistenceStatus, String> {
        let capture = self.artifact_capture.clone();
        let limits = self.artifact_limits;
        #[cfg(test)]
        let before_artifact_capture = self.before_artifact_capture.clone();
        self.run_blocking(move |store| {
            #[cfg(test)]
            if let Some(hook) = before_artifact_capture {
                hook();
            }
            let ArtifactCaptureEntry {
                artifact_id,
                request_id,
                kind,
                occurred_at,
                content,
                media_kind,
                version,
                _memory_permit: _,
            } = entry;
            let unavailable = |reason: ArtifactUnavailableReason| {
                store.insert_unavailable_artifact_pointer(UnavailableArtifactPointer {
                    artifact_id: &artifact_id,
                    request_id: &request_id,
                    occurred_at: &occurred_at,
                    kind,
                    media_kind: Some(&media_kind),
                    version,
                    reason: reason.code(),
                })?;
                Ok(ArtifactPersistenceStatus::Unavailable)
            };

            let content = match content {
                ArtifactCaptureContent::Body(content) => content,
                ArtifactCaptureContent::Unavailable(reason) => return unavailable(reason),
            };
            let (Some(capture), Some((byte_limit, aggregate_limit))) = (capture, limits) else {
                return unavailable(ArtifactUnavailableReason::ArtifactCaptureDisabled);
            };
            match capture.write_artifact(
                &artifact_id,
                &request_id,
                kind,
                &occurred_at,
                &content,
                Some(&media_kind),
                version,
                true,
                false,
                byte_limit,
                aggregate_limit,
            ) {
                Ok(ArtifactCaptureOutcome::Written(_)) => Ok(ArtifactPersistenceStatus::Written),
                Ok(ArtifactCaptureOutcome::Disabled(_)) => {
                    unavailable(ArtifactUnavailableReason::ArtifactCaptureDisabled)
                }
                Err(LogStoreError::ArtifactLimitExceeded { .. }) => {
                    unavailable(ArtifactUnavailableReason::CaptureContentLimitExceeded)
                }
                Err(_) => {
                    unavailable(ArtifactUnavailableReason::ArtifactCaptureFailed)?;
                    Ok(ArtifactPersistenceStatus::FailedUnavailable)
                }
            }
        })
        .await
    }

    async fn persist_proxy_record(&self, proxy_json: String) -> Result<(), String> {
        let record = serde_json::from_str::<ProxyRecord>(&proxy_json)
            .map_err(|_| "invalid proxy record".to_string())?;
        if !validate_proxy_record(&record) {
            return Err("invalid proxy record".to_string());
        }
        let attempt_id = record.attempt_id.as_uuid().to_string();
        let request_id = record.request_id.as_uuid().to_string();
        let occurred_at = record.started_at.clone();
        self.run_blocking(move |store| {
            store.insert_proxy_record(
                &attempt_id,
                &request_id,
                &occurred_at,
                &record.target,
                record.provider.as_deref(),
                record.engine.as_deref(),
                Some(&record.started_at),
                record.completed_at.as_deref(),
                record.status_code.map(i64::from),
                record.error.as_deref(),
            )
        })
        .await
    }

    async fn persist_audit_entry(&self, record: OperationalAuditRecord) -> Result<(), String> {
        let entry_id = record
            .entry_id()
            .map(str::to_owned)
            .unwrap_or_else(|| EventId::new().as_uuid().to_string());
        let occurred_at = record
            .occurred_at()
            .map(str::to_owned)
            .unwrap_or_else(|| self.store.now());
        let detail_json = if let Some(detail_json) = record.detail_json() {
            Some(apply_redaction(&sanitize_paths_in_text(detail_json)).0)
        } else {
            let mut detail = record.context().map_or_else(
                serde_json::Map::new,
                super::service::OperationalAuditContext::fields,
            );
            if let Some(severity) = record.severity() {
                detail.insert("severity".into(), serde_json::json!(severity.as_str()));
            }
            (!detail.is_empty()).then(|| serde_json::Value::Object(detail).to_string())
        };
        self.run_blocking(move |store| {
            store.insert_audit_entry(
                &entry_id,
                None,
                &occurred_at,
                record.source(),
                record.code(),
                detail_json.as_deref(),
            )
        })
        .await
    }

    async fn persist_webhook_delivery(
        &self,
        _request_id: Option<String>,
        _status_code: u16,
        _error: Option<String>,
    ) -> Result<(), String> {
        Err("webhook persistence is not wired by the lifecycle service".to_string())
    }

    async fn persist_cleanup_run(&self, _deleted_count: u64) -> Result<(), String> {
        Err("cleanup persistence is not wired by the lifecycle service".to_string())
    }
}

fn webhook_delivery_id(event_id: &str) -> String {
    format!("webhook:{event_id}")
}

fn terminal_intent(event: &LifecycleEvent) -> Option<(&'static str, Option<u16>)> {
    match event {
        LifecycleEvent::Completed { status_code, .. } => Some(("completed", *status_code)),
        LifecycleEvent::Failed { status_code, .. } => Some(("failed", *status_code)),
        LifecycleEvent::Rejected { status_code, .. } => Some(("rejected", *status_code)),
        LifecycleEvent::Cancelled { .. } => Some(("cancelled", None)),
        LifecycleEvent::Dropped { .. } => Some(("dropped", None)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use mesh_llm_events::logging::envelope::CanonicalEnvelope;
    use mesh_llm_events::logging::identifiers::{AttemptId, EventId, RequestId};
    use mesh_llm_events::logging::proxy::ProxyRecord;

    use super::*;
    use crate::logging::{
        LoggingService, PersistSink, ServiceConfig, SystemClock, TerminalOutcome,
    };

    const OCCURRED_AT: &str = "2026-08-04T12:00:00Z";

    fn open_store() -> (Arc<LogStore>, tempfile::TempDir) {
        let root = tempfile::tempdir().expect("temporary log store");
        let store = LogStore::open(root.path(), Arc::new(mesh_llm_log_store::RealClock))
            .expect("open log store");
        (Arc::new(store), root)
    }

    fn webhook_delivery_count(store: &LogStore) -> i64 {
        store
            .conn()
            .query_row("SELECT COUNT(*) FROM webhook_deliveries", [], |row| {
                row.get(0)
            })
            .expect("count webhook deliveries")
    }

    fn envelope(
        request_id: RequestId,
        event_id: EventId,
        event: LifecycleEvent,
    ) -> CanonicalEnvelope {
        CanonicalEnvelope::new(
            event_id,
            request_id,
            ReplayChannel::Requests,
            1,
            OCCURRED_AT.to_string(),
            event,
        )
    }

    async fn persist_summary(sink: &LogStoreSink, request_id: &RequestId) {
        sink.persist_summary(RequestSummaryEntry {
            request_id: request_id.as_uuid().to_string(),
            state: "active".to_string(),
            created_at: OCCURRED_AT.to_string(),
            terminal_at: None,
            metadata: Default::default(),
        })
        .await
        .expect("persist summary");
    }

    async fn persist_envelope(
        sink: &LogStoreSink,
        envelope: &CanonicalEnvelope,
    ) -> Result<(), String> {
        sink.persist_event(
            envelope.request_id.as_uuid().to_string(),
            envelope.event_id.as_uuid().to_string(),
            envelope.channel,
            envelope.sequence,
            envelope.occurred_at.clone(),
            serde_json::to_string(envelope).expect("serialize envelope"),
        )
        .await
    }

    #[tokio::test]
    async fn disabled_or_nonterminal_persistence_never_enqueues_a_webhook_delivery() {
        let (store, _root) = open_store();
        let disabled = LogStoreSink::new(Arc::clone(&store));
        let terminal_request_id = RequestId::new();
        persist_summary(&disabled, &terminal_request_id).await;
        let terminal = envelope(
            terminal_request_id,
            EventId::new(),
            LifecycleEvent::Completed {
                status_code: Some(200),
                duration_ms: None,
                usage: None,
            },
        );
        persist_envelope(&disabled, &terminal)
            .await
            .expect("disabled terminal persists");
        assert_eq!(
            store
                .get_summary(&terminal_request_id.as_uuid().to_string())
                .expect("standalone summary query")
                .expect("standalone terminal summary")
                .status_code,
            Some(200)
        );

        let enabled = LogStoreSink::with_terminal_webhook_enqueue(Arc::clone(&store), 3);
        let nonterminal_request_id = RequestId::new();
        persist_summary(&enabled, &nonterminal_request_id).await;
        let nonterminal = envelope(
            nonterminal_request_id,
            EventId::new(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        );
        persist_envelope(&enabled, &nonterminal)
            .await
            .expect("nonterminal persists");

        assert_eq!(webhook_delivery_count(&store), 0);
    }

    #[tokio::test]
    async fn enabled_terminal_commit_enqueues_one_stable_delivery_id() {
        let (store, _root) = open_store();
        let sink = LogStoreSink::with_terminal_webhook_enqueue(Arc::clone(&store), 4);
        let request_id = RequestId::new();
        let event_id = EventId::new();
        persist_summary(&sink, &request_id).await;
        let terminal = envelope(
            request_id,
            event_id,
            LifecycleEvent::Completed {
                status_code: Some(201),
                duration_ms: Some(12),
                usage: None,
            },
        );

        persist_envelope(&sink, &terminal)
            .await
            .expect("terminal commit and enqueue");
        assert!(persist_envelope(&sink, &terminal).await.is_err());

        let delivery_id = webhook_delivery_id(&event_id.as_uuid().to_string());
        let delivery = store
            .webhook_delivery(&delivery_id)
            .expect("load delivery")
            .expect("stable delivery record");
        let request_key = request_id.as_uuid().to_string();
        assert_eq!(delivery.request_id.as_deref(), Some(request_key.as_str()));
        assert_eq!(
            delivery.state,
            mesh_llm_log_store::WebhookDeliveryState::Pending
        );
        assert_eq!(delivery.attempt_number, 0);
        assert_eq!(delivery.max_attempts, 4);
        assert_eq!(delivery.terminal_status_code, Some(201));
        assert_eq!(delivery.response_status_code, None);
        assert_eq!(
            store
                .get_summary(&request_key)
                .expect("summary query")
                .expect("terminal summary")
                .status_code,
            Some(201)
        );
        assert_eq!(webhook_delivery_count(&store), 1);
    }

    #[tokio::test]
    async fn webhook_enqueue_failure_rolls_back_terminal_event_and_summary() {
        let (store, _root) = open_store();
        let sink = LogStoreSink::with_terminal_webhook_enqueue(Arc::clone(&store), 3);
        let request_id = RequestId::new();
        persist_summary(&sink, &request_id).await;
        store
            .conn()
            .execute_batch(
                "CREATE TRIGGER fail_terminal_webhook_enqueue \
                 BEFORE INSERT ON webhook_deliveries \
                 BEGIN SELECT RAISE(ABORT, 'injected webhook enqueue failure'); END;",
            )
            .expect("install deterministic enqueue failure");
        let terminal = envelope(
            request_id,
            EventId::new(),
            LifecycleEvent::Completed {
                status_code: Some(200),
                duration_ms: None,
                usage: None,
            },
        );

        assert!(persist_envelope(&sink, &terminal).await.is_err());
        let request_key = request_id.as_uuid().to_string();
        let summary = store
            .get_summary(&request_key)
            .expect("load summary")
            .expect("summary remains durable");
        assert_eq!(summary.state, "active");
        assert_eq!(summary.status_code, None);
        assert!(
            !store
                .has_terminal_event(&request_key)
                .expect("terminal state")
        );
        assert_eq!(webhook_delivery_count(&store), 0);
    }

    #[tokio::test]
    async fn webhook_enqueue_failure_cannot_change_the_terminal_request_result() {
        let (store, _root) = open_store();
        let sink = Arc::new(LogStoreSink::with_terminal_webhook_enqueue(
            Arc::clone(&store),
            3,
        ));
        store
            .conn()
            .execute_batch(
                "CREATE TRIGGER fail_terminal_webhook_enqueue \
                 BEFORE INSERT ON webhook_deliveries \
                 BEGIN SELECT RAISE(ABORT, 'injected webhook enqueue failure'); END;",
            )
            .expect("install deterministic enqueue failure");
        let service = LoggingService::new(ServiceConfig::default(), sink, Box::new(SystemClock));
        let request_id = RequestId::new();
        let (guard, _) = service.register_request(request_id);

        assert!(
            service
                .transition_terminal(request_id, &guard, TerminalOutcome::Completed)
                .is_ok()
        );
        assert_eq!(service.pump_sync().await, 3);
        assert_eq!(service.persistence_failures(), 1);
        let summary = store
            .get_summary(&request_id.as_uuid().to_string())
            .expect("load summary")
            .expect("durable active summary");
        assert_eq!(summary.state, "active");
        assert!(
            !store
                .has_terminal_event(&request_id.as_uuid().to_string())
                .expect("terminal state")
        );
        assert_eq!(webhook_delivery_count(&store), 0);
    }

    #[tokio::test]
    async fn proxy_record_persistence_preserves_keys_timestamps_and_bounded_fields() {
        let (store, _root) = open_store();
        let sink = LogStoreSink::new(Arc::clone(&store));
        let request_id = RequestId::new();
        let attempt_id = AttemptId::new();
        persist_summary(&sink, &request_id).await;
        let record = ProxyRecord {
            attempt_id,
            request_id,
            target: "remote".to_string(),
            provider: Some("openai_frontend".to_string()),
            engine: Some("responses".to_string()),
            started_at: OCCURRED_AT.to_string(),
            completed_at: Some("2026-08-04T12:00:01Z".to_string()),
            status_code: Some(502),
            error: Some("timeout".to_string()),
        };
        let payload = serde_json::to_string(&record).expect("serialize proxy record");

        sink.persist_proxy_record(payload.clone())
            .await
            .expect("persist proxy record");
        assert!(sink.persist_proxy_record(payload).await.is_err());

        let request_key = request_id.as_uuid().to_string();
        let attempt_key = attempt_id.as_uuid().to_string();
        let row = store
            .conn()
            .query_row(
                "SELECT attempt_id, request_id, occurred_at, target, provider, engine, started_at, completed_at, status_code, error_msg FROM proxy_records",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, Option<String>>(6)?,
                        row.get::<_, Option<String>>(7)?,
                        row.get::<_, Option<i64>>(8)?,
                        row.get::<_, Option<String>>(9)?,
                    ))
                },
            )
            .expect("read persisted proxy record");
        assert_eq!(row.0, attempt_key);
        assert_eq!(row.1, request_key);
        assert_eq!(row.2, "2026-08-04T12:00:00.000000000Z");
        assert_eq!(row.3, "remote");
        assert_eq!(row.4.as_deref(), Some("openai_frontend"));
        assert_eq!(row.5.as_deref(), Some("responses"));
        assert_eq!(row.6.as_deref(), Some("2026-08-04T12:00:00.000000000Z"));
        assert_eq!(row.7.as_deref(), Some("2026-08-04T12:00:01.000000000Z"));
        assert_eq!(row.8, Some(502));
        assert_eq!(row.9.as_deref(), Some("timeout"));

        let unowned = ProxyRecord {
            attempt_id: AttemptId::new(),
            request_id: RequestId::new(),
            ..record.clone()
        };
        assert!(
            sink.persist_proxy_record(
                serde_json::to_string(&unowned).expect("serialize unowned proxy")
            )
            .await
            .is_err()
        );
        let invalid_target = ProxyRecord {
            target: "https://host.invalid/path".to_string(),
            ..record.clone()
        };
        assert!(
            sink.persist_proxy_record(
                serde_json::to_string(&invalid_target).expect("serialize invalid proxy target")
            )
            .await
            .is_err()
        );
        let invalid_error = ProxyRecord {
            error: Some("untrusted_error_text".to_string()),
            ..record
        };
        assert!(
            sink.persist_proxy_record(
                serde_json::to_string(&invalid_error).expect("serialize invalid proxy error")
            )
            .await
            .is_err()
        );
        let proxy_record_count: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM proxy_records", [], |row| row.get(0))
            .expect("count proxy records");
        assert_eq!(proxy_record_count, 1);
        assert!(sink.persist_proxy_record("{}".to_string()).await.is_err());
    }

    #[tokio::test]
    async fn persistence_defensively_sanitizes_untrusted_lifecycle_strings() {
        let (store, _root) = open_store();
        let sink = LogStoreSink::new(Arc::clone(&store));
        let request_id = RequestId::new();
        let event_id = EventId::new();
        persist_summary(&sink, &request_id).await;
        let home = dirs::home_dir().expect("home directory");
        let event = envelope(
            request_id,
            event_id,
            LifecycleEvent::Failed {
                error: format!(
                    "Bearer secret at {} {}",
                    home.join("private/file").display(),
                    "x".repeat(super::super::policy::MAX_LOG_STRING_LEN + 100)
                ),
                status_code: None,
            },
        );

        persist_envelope(&sink, &event)
            .await
            .expect("persist sanitized envelope");

        let payload: String = store
            .conn()
            .query_row(
                "SELECT payload_json FROM lifecycle_events WHERE event_id = ?",
                [event_id.as_uuid().to_string()],
                |row| row.get(0),
            )
            .expect("read lifecycle payload");
        assert!(!payload.contains("secret"));
        assert!(!payload.contains(&home.display().to_string()));
        assert!(payload.len() < 2_000);
    }
}
