use super::*;

#[derive(Default)]
pub(super) struct RecordingOutputSink {
    events: std::sync::Mutex<Vec<OutputEvent>>,
}

impl RecordingOutputSink {
    pub(super) fn take_events(&self) -> Vec<OutputEvent> {
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

pub(super) struct OutputSinkResetGuard;

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
pub(super) enum TestRecord {
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
pub(super) struct TestSink {
    records: std::sync::Mutex<Vec<TestRecord>>,
    fail_flag: Arc<AtomicU64>, // if > 0, all operations return Err
    audit_notifications: Option<mpsc::UnboundedSender<(String, String)>>,
    audit_attempt_notifications: Option<mpsc::UnboundedSender<()>>,
}

impl TestSink {
    pub(super) fn new() -> Self {
        Self {
            records: std::sync::Mutex::new(Vec::new()),
            fail_flag: Arc::new(AtomicU64::new(0)),
            audit_notifications: None,
            audit_attempt_notifications: None,
        }
    }

    pub(super) fn recording() -> (Self, mpsc::UnboundedReceiver<(String, String)>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut sink = Self::new();
        sink.audit_notifications = Some(tx);
        (sink, rx)
    }

    pub(super) fn failing_with_attempt_notifications() -> (Self, mpsc::UnboundedReceiver<()>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut sink = Self::new();
        sink.set_failing();
        sink.audit_attempt_notifications = Some(tx);
        (sink, rx)
    }

    /// Set the sink to return Err on all subsequent operations (simulates store failure).
    pub(super) fn set_failing(&self) {
        self.fail_flag.store(1, AtomicOrdering::Release);
    }

    /// Clear the failing flag.
    #[allow(dead_code)]
    pub(super) fn clear_fail(&self) {
        self.fail_flag.store(0, AtomicOrdering::Release);
    }

    /// Get all records captured so far (for test assertions).
    #[allow(dead_code)]
    pub(super) fn records(&self) -> Vec<TestRecord> {
        self.records.lock().unwrap().clone()
    }

    /// Count of audit entries with a specific level.
    #[allow(dead_code)]
    pub(super) fn audit_count_by_level(&self, level: &str) -> usize {
        self.records()
            .iter()
            .filter(|r| matches!(r, TestRecord::AuditEntry { level: lvl, .. } if lvl == level))
            .count()
    }

    /// Reset records to empty (for multi-phase tests).
    #[allow(dead_code)]
    pub(super) fn clear(&self) {
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
pub(super) struct BlockingAuditSink {
    first_write: AtomicBool,
    started: mpsc::UnboundedSender<()>,
    completed: mpsc::UnboundedSender<String>,
    release: Arc<Notify>,
}

impl BlockingAuditSink {
    pub(super) fn new() -> (
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
pub(super) struct TestClock {
    counter: AtomicU64,
}

impl TestClock {
    pub(super) fn new() -> Self {
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
