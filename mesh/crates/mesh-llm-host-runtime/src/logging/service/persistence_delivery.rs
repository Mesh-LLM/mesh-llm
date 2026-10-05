//! Persistence delivery ownership for LoggingService.
//!
//! This module owns the bounded hand-off queues, worker lifecycle, synchronous
//! pump, and shutdown drain. The parent service retains the public facade and
//! event projection while this module keeps the one-time persistence path
//! isolated.

use super::{
    ArtifactCaptureEntry, ArtifactPersistenceStatus, BusEntry, CanonicalEnvelope, LifecycleEvent,
    LoggingArtifactCaptureStatus, LoggingMetric, LoggingMetrics, LoggingService,
    OperationalAuditRecord, OperationalAuditSeverity, PersistSink, RequestSummaryEntry,
    record_persistence_failure,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::{AbortHandle, JoinHandle};

/// Internal message sent from the service to the persistence worker via mpsc channel.
#[derive(Debug)]
pub(super) enum WorkerMessage {
    /// Persist one accepted entry through the service-owned delivery path.
    Persist(Box<PersistenceEntry>),
    /// Drain every preceding entry, acknowledge the drain, then exit. The
    /// control message is queued behind normal work so the acknowledgement is
    /// a precise durability boundary for the bounded worker channel.
    Shutdown(oneshot::Sender<()>),
}

/// One durable record accepted by the logging service. Proxy attempts are
/// observational records, so they do not enter the lifecycle replay bus.
#[derive(Debug)]
pub(super) enum PersistenceEntry {
    Bus(BusEntry),
    /// A terminal event carries its terminal summary on the priority lane so
    /// a saturated normal queue can never strand a durable request as active.
    Terminal {
        entry: BusEntry,
        summary: RequestSummaryEntry,
    },
    Audit(OperationalAuditRecord),
    ProxyRecord(String),
    Artifact {
        entry: ArtifactCaptureEntry,
        summary: RequestSummaryEntry,
    },
}

impl PersistenceEntry {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal { .. })
    }
}

#[derive(Clone)]
pub(super) struct WorkerSenders {
    normal: mpsc::Sender<WorkerMessage>,
    terminal: mpsc::Sender<PersistenceEntry>,
}

/// The one owner of an accepted entry's persistence hand-off.
///
/// The replay bus intentionally remains independent of this state: replay is a
/// bounded read window, while this queue is a one-time delivery path. Keeping
/// them separate prevents a synchronous persistence pass from consuming replay
/// history or causing a second persistence attempt.
pub(super) enum DeliveryMode {
    /// No worker is running. Entries are retained for an explicit
    /// [`LoggingService::pump_sync`] call or handed to the first worker.
    Manual {
        pending: VecDeque<PersistenceEntry>,
        capacity: usize,
    },
    /// A `pump_sync` task owns entries that were atomically removed from the
    /// manual queue. The completion signal lets shutdown wait or abort without
    /// allowing another pump to duplicate those entries.
    ManualPumping(Arc<ManualPumpCompletion>),
    /// A dedicated worker owns delivery through this bounded channel.
    Worker(WorkerSenders),
    /// Shutdown has frozen new persistence hand-offs while the worker drains
    /// entries already accepted before the transition.
    Stopping,
    /// The previous worker is joined. New events still reach replay, but their
    /// persistence hand-off is counted as unavailable until a later spawn.
    Stopped,
}

/// Path-free local state of the persistence delivery owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PersistenceWorkerState {
    NotStarted,
    Running,
    Stopping,
    Stopped,
}

impl PersistenceWorkerState {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::NotStarted => "not_started",
            Self::Running => "running",
            Self::Stopping => "stopping",
            Self::Stopped => "stopped",
        }
    }
}

pub(super) struct ManualPumpCompletion {
    done: AtomicBool,
    notify: Notify,
}

impl ManualPumpCompletion {
    fn new() -> Self {
        Self {
            done: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    fn finish(&self) {
        self.done.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            if self.done.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }
}

/// Handle to the persistence worker task for controlled shutdown.
pub struct WorkerHandle {
    tx: mpsc::Sender<WorkerMessage>,
    task: JoinHandle<()>,
}

pub(super) fn offer_summary_persistence(
    delivery: &Mutex<DeliveryMode>,
    persistence_queue_drops: &AtomicU64,
    persistence_outstanding: &AtomicU64,
    metrics: &LoggingMetrics,
    summary: RequestSummaryEntry,
) {
    let payload = match serde_json::to_string(&serde_json::json!({
        "kind": "summary",
        "summary": summary,
    })) {
        Ok(payload) => payload,
        Err(_) => {
            record_persistence_queue_drop(persistence_queue_drops, metrics);
            return;
        }
    };
    offer_persistence_to(
        delivery,
        persistence_queue_drops,
        persistence_outstanding,
        metrics,
        PersistenceEntry::Bus(BusEntry {
            payload,
            channel_hint: 0,
        }),
    );
}

pub(super) fn offer_persistence_to(
    delivery: &Mutex<DeliveryMode>,
    persistence_queue_drops: &AtomicU64,
    persistence_outstanding: &AtomicU64,
    metrics: &LoggingMetrics,
    entry: PersistenceEntry,
) {
    let mut delivery = match delivery.lock() {
        Ok(delivery) => delivery,
        Err(poisoned) => poisoned.into_inner(),
    };
    match &mut *delivery {
        DeliveryMode::Manual { pending, capacity } => {
            if pending.len() >= *capacity {
                pending.pop_front();
                record_persistence_queue_drop(persistence_queue_drops, metrics);
                decrement_outstanding(persistence_outstanding, metrics);
            }
            pending.push_back(entry);
            increment_outstanding(persistence_outstanding, metrics);
        }
        DeliveryMode::Worker(senders) => offer_worker_persistence(
            senders,
            persistence_queue_drops,
            persistence_outstanding,
            metrics,
            entry,
        ),
        DeliveryMode::ManualPumping(_) | DeliveryMode::Stopping | DeliveryMode::Stopped => {
            record_persistence_queue_drop(persistence_queue_drops, metrics);
        }
    }
}

fn offer_worker_persistence(
    senders: &WorkerSenders,
    persistence_queue_drops: &AtomicU64,
    persistence_outstanding: &AtomicU64,
    metrics: &LoggingMetrics,
    entry: PersistenceEntry,
) {
    let accepted = if entry.is_terminal() {
        senders.terminal.try_send(entry).is_ok()
    } else {
        senders
            .normal
            .try_send(WorkerMessage::Persist(Box::new(entry)))
            .is_ok()
    };
    if accepted {
        increment_outstanding(persistence_outstanding, metrics);
    } else {
        record_persistence_queue_drop(persistence_queue_drops, metrics);
    }
}

pub(super) fn record_persistence_queue_drop(counter: &AtomicU64, metrics: &LoggingMetrics) {
    counter.fetch_add(1, Ordering::Relaxed);
    metrics.record(LoggingMetric::PersistenceQueueDropped { count: 1 });
}

fn increment_persistence_failure(counter: &AtomicU64, metrics: &LoggingMetrics) {
    counter.fetch_add(1, Ordering::Relaxed);
    metrics.record(LoggingMetric::PersistenceFailure { count: 1 });
}

fn increment_outstanding(outstanding: &AtomicU64, metrics: &LoggingMetrics) {
    let current = outstanding
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1);
    metrics.record(LoggingMetric::PersistenceOutstanding { current });
}

fn decrement_outstanding(outstanding: &AtomicU64, metrics: &LoggingMetrics) {
    let _ = outstanding.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        current.checked_sub(1)
    });
    metrics.record(LoggingMetric::PersistenceOutstanding {
        current: outstanding.load(Ordering::Acquire),
    });
}

impl LoggingService {
    /// Start the persistence worker task. Entries accepted before startup are
    /// transferred from the bounded manual delivery queue. Idempotent: calling
    /// twice is a no-op (second call returns false). Returns true for a new
    /// worker.
    pub fn spawn(&self) -> bool {
        let _transition = self
            .transition_lock
            .lock()
            .expect("transition mutex poisoned");
        if !self.startable.load(Ordering::Acquire) {
            return false;
        }
        // Prevent double-spawn.
        if self
            .spawned
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }

        let sink_opt = self.sink.clone();
        let persistence_failures = Arc::clone(&self.persistence_failures);
        let persistence_outstanding = Arc::clone(&self.persistence_outstanding);
        let metrics = self.metrics.clone();
        let writer = Arc::clone(&self.writer);
        let failure_delivery = self.event_delivery();
        // Tokio rejects a zero-capacity bounded channel. Direct library users
        // can construct ServiceConfig without going through host validation,
        // so retain fail-open behavior with the smallest valid worker queue.
        let persistence_queue_capacity = self.config.queue_capacity.max(1);
        // At most the active and recent registry windows can contribute a
        // distinct terminal transition before a healthy worker catches up.
        // Keep that durable boundary separate from normal observational
        // traffic so stream/proxy volume cannot evict terminal state.
        let terminal_queue_capacity = self
            .config
            .registry_config
            .max_active
            .saturating_add(self.config.registry_config.max_recent)
            .max(1);

        let (tx, mut rx) = mpsc::channel::<WorkerMessage>(persistence_queue_capacity);
        let (terminal_tx, mut terminal_rx) =
            mpsc::channel::<PersistenceEntry>(terminal_queue_capacity);
        let senders = WorkerSenders {
            normal: tx.clone(),
            terminal: terminal_tx,
        };

        // Switch delivery modes while holding one lock. An enqueue can therefore
        // hand an entry to either the manual queue or worker, never both.
        let pending = {
            let mut delivery = self.delivery.lock().expect("delivery mutex poisoned");
            match std::mem::replace(&mut *delivery, DeliveryMode::Worker(senders.clone())) {
                DeliveryMode::Manual { pending, .. } => pending,
                DeliveryMode::Stopped => VecDeque::new(),
                DeliveryMode::ManualPumping(_) | DeliveryMode::Stopping => {
                    *delivery = DeliveryMode::Stopping;
                    self.spawned.store(false, Ordering::Release);
                    return false;
                }
                DeliveryMode::Worker(existing_senders) => {
                    *delivery = DeliveryMode::Worker(existing_senders);
                    self.spawned.store(false, Ordering::Release);
                    return false;
                }
            }
        };

        for entry in pending {
            // Entries were already counted when accepted into the manual
            // queue, so transfer them without adjusting outstanding totals.
            let accepted = if entry.is_terminal() {
                senders.terminal.try_send(entry).is_ok()
            } else {
                senders
                    .normal
                    .try_send(WorkerMessage::Persist(Box::new(entry)))
                    .is_ok()
            };
            if !accepted {
                record_persistence_queue_drop(&self.persistence_queue_drops, &self.metrics);
                decrement_outstanding(&self.persistence_outstanding, &self.metrics);
            }
        }

        let task = tokio::spawn(async move {
            loop {
                let msg = tokio::select! {
                    biased;
                    Some(entry) = terminal_rx.recv() => Some(WorkerMessage::Persist(Box::new(entry))),
                    message = rx.recv() => message,
                };
                let Some(msg) = msg else {
                    break;
                };
                match msg {
                    WorkerMessage::Persist(entry) => {
                        // Parse the bus entry and persist via sink.
                        if let Some(sink) = &sink_opt {
                            // Best-effort: failures are absorbed by fail-open writer.
                            if Self::process_persistence_entry(sink.as_ref(), &entry, &metrics)
                                .await
                                .is_err()
                            {
                                increment_persistence_failure(&persistence_failures, &metrics);
                                record_persistence_failure(
                                    writer.as_ref(),
                                    &failure_delivery,
                                    &entry,
                                );
                            }
                        }
                        decrement_outstanding(&persistence_outstanding, &metrics);
                    }
                    WorkerMessage::Shutdown(ack) => {
                        let _ = ack.send(());
                        break;
                    }
                }
            }
        });

        // Store both the control sender and join handle. Retaining the handle
        // makes shutdown an observable drain boundary rather than a detached
        // best-effort task drop.
        *self
            .worker_handle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(WorkerHandle { tx, task });

        true
    }

    async fn process_persistence_entry(
        sink: &dyn PersistSink,
        entry: &PersistenceEntry,
        metrics: &LoggingMetrics,
    ) -> Result<(), String> {
        match entry {
            PersistenceEntry::Bus(entry) => Self::process_bus_entry(sink, entry).await,
            PersistenceEntry::Terminal { entry, summary } => {
                // The priority lane also carries the current terminal summary.
                // Persist it before the event so a dropped admitted/metadata
                // entry cannot violate the lifecycle foreign key or leave a
                // durable active row behind.
                sink.persist_summary(summary.clone()).await?;
                Self::process_bus_entry(sink, entry).await
            }
            PersistenceEntry::Audit(record) => sink.persist_audit_entry(record.clone()).await,
            PersistenceEntry::ProxyRecord(proxy_json) => {
                sink.persist_proxy_record(proxy_json.clone()).await
            }
            PersistenceEntry::Artifact { entry, summary } => {
                // Carrying the canonical summary makes the artifact FK robust
                // even if earlier observational queue entries were evicted.
                sink.persist_summary(summary.clone()).await?;
                match sink.persist_artifact_capture(entry.clone()).await {
                    Ok(ArtifactPersistenceStatus::Written) => {
                        metrics.record(LoggingMetric::ArtifactCapture {
                            status: LoggingArtifactCaptureStatus::Written,
                        });
                        Ok(())
                    }
                    Ok(ArtifactPersistenceStatus::Unavailable) => {
                        metrics.record(LoggingMetric::ArtifactCapture {
                            status: LoggingArtifactCaptureStatus::Disabled,
                        });
                        Ok(())
                    }
                    Ok(ArtifactPersistenceStatus::FailedUnavailable) => {
                        metrics.record(LoggingMetric::ArtifactCapture {
                            status: LoggingArtifactCaptureStatus::Failed,
                        });
                        Ok(())
                    }
                    Err(error) => {
                        metrics.record(LoggingMetric::ArtifactCapture {
                            status: LoggingArtifactCaptureStatus::Failed,
                        });
                        Err(error)
                    }
                }
            }
        }
    }

    async fn process_bus_entry(sink: &dyn PersistSink, entry: &BusEntry) -> Result<(), String> {
        let record: serde_json::Value = serde_json::from_str(&entry.payload)
            .map_err(|e| format!("invalid bus entry JSON: {}", e))?;

        if record.get("kind").and_then(serde_json::Value::as_str) == Some("summary") {
            let summary = serde_json::from_value(
                record
                    .get("summary")
                    .cloned()
                    .ok_or_else(|| "summary bus record has no summary".to_string())?,
            )
            .map_err(|error| format!("invalid summary bus record: {error}"))?;
            return sink.persist_summary(summary).await;
        }

        if record.get("kind").and_then(serde_json::Value::as_str) == Some("audit") {
            return Err("audit bus record reached lifecycle persistence path".to_string());
        }

        if let Some(envelope_value) = record
            .get("canonical_envelope")
            .filter(|value| !value.is_null())
        {
            let envelope = CanonicalEnvelope::from_json_str(&envelope_value.to_string())
                .map_err(|error| format!("invalid canonical bus envelope: {error}"))?;
            if let LifecycleEvent::AuditError { .. } = envelope.event {
                return sink
                    .persist_audit_entry(
                        OperationalAuditRecord::builder("logging_service", "audit_error")
                            .severity(OperationalAuditSeverity::Error)
                            .build()
                            .with_internal_detail(serde_json::to_string(&envelope).map_err(
                                |error| format!("serialize canonical audit envelope: {error}"),
                            )?),
                    )
                    .await;
            }
            return sink
                .persist_event(
                    envelope.request_id.as_uuid().to_string(),
                    envelope.event_id.as_uuid().to_string(),
                    envelope.channel,
                    envelope.sequence,
                    envelope.occurred_at.clone(),
                    serde_json::to_string(&envelope)
                        .map_err(|error| format!("serialize canonical bus envelope: {error}"))?,
                )
                .await;
        }

        // Entries which are not canonical lifecycle records are operational
        // audit records. They stay out of the lifecycle repositories.
        sink.persist_audit_entry(
            OperationalAuditRecord::builder("logging_service", "uncategorized_bus_record")
                .severity(OperationalAuditSeverity::Info)
                .build()
                .with_internal_detail(entry.payload.clone()),
        )
        .await
    }

    /// Explicit no-worker pump for deterministic tests. It never drains the
    /// replay bus, and it is a no-op while a worker owns delivery. Returns the
    /// number of entries offered to the sink.
    #[allow(dead_code)]
    pub async fn pump_sync(&self) -> usize {
        let (task, completion) = {
            let _transition = self
                .transition_lock
                .lock()
                .expect("transition mutex poisoned");
            let mut abort = self
                .manual_abort_handle
                .lock()
                .expect("manual abort mutex poisoned");
            let mut delivery = self.delivery.lock().expect("delivery mutex poisoned");
            let entries = match &mut *delivery {
                DeliveryMode::Manual { pending, .. } => std::mem::take(pending),
                DeliveryMode::ManualPumping(_)
                | DeliveryMode::Worker(_)
                | DeliveryMode::Stopping
                | DeliveryMode::Stopped => return 0,
            };
            let Some(sink) = self.sink.clone() else {
                return 0;
            };
            if entries.is_empty() {
                return 0;
            }
            let completion = Arc::new(ManualPumpCompletion::new());
            *delivery = DeliveryMode::ManualPumping(Arc::clone(&completion));
            let delivery_state = Arc::clone(&self.delivery);
            let abort_state = Arc::clone(&self.manual_abort_handle);
            let failures = Arc::clone(&self.persistence_failures);
            let outstanding = Arc::clone(&self.persistence_outstanding);
            let metrics = self.metrics.clone();
            let writer = Arc::clone(&self.writer);
            let failure_delivery = self.event_delivery();
            let task_completion = Arc::clone(&completion);
            let persistence_queue_capacity = self.config.queue_capacity.max(1);
            let task = tokio::spawn(async move {
                let count = entries.len();
                for entry in entries {
                    if Self::process_persistence_entry(sink.as_ref(), &entry, &metrics)
                        .await
                        .is_err()
                    {
                        increment_persistence_failure(&failures, &metrics);
                        record_persistence_failure(writer.as_ref(), &failure_delivery, &entry);
                    }
                    decrement_outstanding(&outstanding, &metrics);
                }
                task_completion.finish();
                let mut abort = abort_state.lock().expect("manual abort mutex poisoned");
                *abort = None;
                let mut delivery = delivery_state.lock().expect("delivery mutex poisoned");
                if matches!(&*delivery, DeliveryMode::ManualPumping(_)) {
                    *delivery = DeliveryMode::Manual {
                        pending: VecDeque::new(),
                        capacity: persistence_queue_capacity,
                    };
                }
                count
            });
            *abort = Some(task.abort_handle());
            (task, completion)
        };
        match task.await {
            Ok(count) => count,
            Err(_) => {
                completion.finish();
                0
            }
        }
    }

    /// Snapshot the delivery owner's state for trusted-local diagnostics.
    #[allow(dead_code)]
    pub(crate) fn persistence_worker_state(&self) -> PersistenceWorkerState {
        let delivery = self.delivery.lock().expect("delivery mutex poisoned");
        match &*delivery {
            DeliveryMode::Manual { .. } => PersistenceWorkerState::NotStarted,
            DeliveryMode::ManualPumping(_) | DeliveryMode::Worker(_) => {
                PersistenceWorkerState::Running
            }
            DeliveryMode::Stopping => PersistenceWorkerState::Stopping,
            DeliveryMode::Stopped => PersistenceWorkerState::Stopped,
        }
    }

    /// Stop accepting persistence hand-offs, drain the worker within a fixed
    /// bound, and join it. Replay remains readable throughout. If the worker
    /// is stalled beyond the bound, it is aborted and every still-owned entry
    /// is recorded as a bounded shutdown loss; serving remains fail-open.
    ///
    /// A completed shutdown leaves the service in a stopped state. Calling
    /// [`Self::spawn`] later starts a fresh worker safely; a second shutdown is
    /// a no-op.
    #[allow(dead_code)]
    pub async fn shutdown(&self) -> bool {
        enum ShutdownOwner {
            Worker(WorkerHandle),
            Manual(VecDeque<PersistenceEntry>),
            ManualPump(Arc<ManualPumpCompletion>, AbortHandle),
            Unavailable,
        }

        // Freeze hand-off and claim exactly one delivery owner under the same
        // transition lock used by spawn/pump. No accepted entry can move into
        // a new owner after this boundary.
        let owner = {
            let _transition = self
                .transition_lock
                .lock()
                .expect("transition mutex poisoned");
            let mut delivery = self.delivery.lock().expect("delivery mutex poisoned");
            let previous = std::mem::replace(&mut *delivery, DeliveryMode::Stopping);
            match previous {
                DeliveryMode::Stopped | DeliveryMode::Stopping => {
                    *delivery = previous;
                    return false;
                }
                DeliveryMode::Worker(_) => {
                    self.spawned.store(false, Ordering::Release);
                    let handle = self
                        .worker_handle
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take();
                    match handle {
                        Some(handle) => ShutdownOwner::Worker(handle),
                        None => ShutdownOwner::Unavailable,
                    }
                }
                DeliveryMode::Manual { pending, .. } => ShutdownOwner::Manual(pending),
                DeliveryMode::ManualPumping(completion) => {
                    let abort = self
                        .manual_abort_handle
                        .lock()
                        .expect("manual abort mutex poisoned")
                        .take();
                    match abort {
                        Some(abort) => ShutdownOwner::ManualPump(completion, abort),
                        None => ShutdownOwner::Manual(VecDeque::new()),
                    }
                }
            }
        };

        let drained = match owner {
            ShutdownOwner::Worker(WorkerHandle { tx, mut task }) => {
                let (drained_tx, drained_rx) = oneshot::channel();
                let result = tokio::time::timeout(self.shutdown_drain_timeout, async {
                    tx.send(WorkerMessage::Shutdown(drained_tx))
                        .await
                        .map_err(|_| ())?;
                    drained_rx.await.map_err(|_| ())?;
                    (&mut task).await.map_err(|_| ())
                })
                .await;
                if result.is_err() || !matches!(result, Ok(Ok(()))) {
                    if !task.is_finished() {
                        task.abort();
                        let _ = task.await;
                    }
                    false
                } else {
                    true
                }
            }
            ShutdownOwner::Manual(entries) => match self.sink.clone() {
                None => true,
                Some(sink) => {
                    let result = tokio::time::timeout(self.shutdown_drain_timeout, async {
                        for entry in entries {
                            if Self::process_persistence_entry(sink.as_ref(), &entry, &self.metrics)
                                .await
                                .is_err()
                            {
                                increment_persistence_failure(
                                    &self.persistence_failures,
                                    &self.metrics,
                                );
                                record_persistence_failure(
                                    self.writer.as_ref(),
                                    &self.event_delivery(),
                                    &entry,
                                );
                            }
                            decrement_outstanding(&self.persistence_outstanding, &self.metrics);
                        }
                    })
                    .await;
                    result.is_ok()
                }
            },
            ShutdownOwner::ManualPump(completion, abort) => {
                let result =
                    tokio::time::timeout(self.shutdown_drain_timeout, completion.wait()).await;
                if result.is_err() {
                    abort.abort();
                    false
                } else {
                    true
                }
            }
            ShutdownOwner::Unavailable => false,
        };

        if !drained {
            // Once a bounded owner is cancelled, no later decrement may
            // underflow this total: delivery uses saturating decrement above.
            let lost = self.persistence_outstanding.swap(0, Ordering::AcqRel);
            self.persistence_shutdown_losses
                .fetch_add(lost, Ordering::Relaxed);
            if lost != 0 {
                self.metrics
                    .record(LoggingMetric::PersistenceShutdownLoss { count: lost });
            }
            self.metrics
                .record(LoggingMetric::PersistenceOutstanding { current: 0 });
        }
        self.set_delivery_stopped();
        true
    }

    fn set_delivery_stopping(&self) {
        let mut delivery = self.delivery.lock().expect("delivery mutex poisoned");
        *delivery = DeliveryMode::Stopping;
    }

    fn set_delivery_stopped(&self) {
        let mut delivery = self.delivery.lock().expect("delivery mutex poisoned");
        *delivery = DeliveryMode::Stopped;
    }
}
