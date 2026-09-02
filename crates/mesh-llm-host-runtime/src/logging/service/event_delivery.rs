//! Canonical event projection owned by LoggingService.
//!
//! Request lifecycle callbacks build replay and persistence envelopes here;
//! the parent service remains responsible for configuration and API methods.

use super::{
    BusEntry, Clock, EventId, LifecycleEvent, LifecycleRecorder, LoggingMetric, LoggingMetrics,
    LoggingTerminalOutcome, OperationalAuditRecord, ReplayBus, ReplayChannel, RequestId,
    RequestRegistry, RequestSummaryEntry, RequestSummaryEventSnapshots, RequestSummaryMetadata,
    SequenceGenerators, TerminalOutcome, canonical_clock_timestamp, emit_accepted_canonical_event,
    sanitize_lifecycle_event,
};
use super::{DeliveryMode, PersistenceEntry, offer_persistence_to};
use mesh_llm_events::logging::envelope::{CanonicalEnvelope, CanonicalPresentationContext};
use mesh_llm_events::logging::replay::ReplaySequence;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

/// The service-owned terminal callback installed in request lifecycle guards.
/// It holds only service components, never a guard, so request ownership cannot
/// form a reference cycle with the logging runtime.
#[derive(Clone)]
pub(super) struct EventDelivery {
    pub(super) bus: Arc<ReplayBus>,
    pub(super) registry: Arc<RequestRegistry>,
    pub(super) metrics: LoggingMetrics,
    pub(super) sequences: SequenceGenerators,
    pub(super) summary_line_limit: usize,
    pub(super) sink_enabled: bool,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) delivery: Arc<Mutex<DeliveryMode>>,
    pub(super) persistence_queue_drops: Arc<AtomicU64>,
    pub(super) persistence_outstanding: Arc<AtomicU64>,
}

impl EventDelivery {
    pub(super) fn enqueue(
        &self,
        request_id: RequestId,
        channel: ReplayChannel,
        payload_json: String,
    ) {
        self.enqueue_with_summary_snapshots(request_id, channel, payload_json, None, None);
    }

    fn enqueue_with_summary_snapshots(
        &self,
        request_id: RequestId,
        channel: ReplayChannel,
        payload_json: String,
        summary_snapshots: Option<RequestSummaryEventSnapshots>,
        terminal_summary: Option<RequestSummaryEntry>,
    ) {
        let _ = enqueue_event_with_delivery(
            self,
            request_id,
            channel,
            payload_json,
            None,
            summary_snapshots,
            terminal_summary,
        );
    }

    pub(super) fn enqueue_audit(&self, record: OperationalAuditRecord) {
        let entry_id = EventId::new().as_uuid().to_string();
        let occurred_at = canonical_clock_timestamp(self.clock.as_ref());
        let record = record.with_identity(entry_id.clone(), occurred_at.clone());
        let mut payload = serde_json::json!({
            "kind": "audit",
            "entry_id": entry_id,
            "occurred_at": occurred_at,
            "source": record.source(),
            "code": record.code(),
        });
        if let Some(severity) = record.severity() {
            payload
                .as_object_mut()
                .expect("audit payload is always an object")
                .insert("severity".into(), serde_json::json!(severity.as_str()));
        }
        if let Some(context) = record.context() {
            payload
                .as_object_mut()
                .expect("audit payload is always an object")
                .extend(context.fields());
        }
        let entry = BusEntry {
            payload: payload.to_string(),
            channel_hint: 2,
        };
        let outcome = self
            .bus
            .push_audit_replay(entry.payload.clone(), entry.channel_hint);
        if self.sink_enabled && !matches!(outcome, super::super::bus::PushOutcome::Rejected) {
            offer_persistence_to(
                &self.delivery,
                &self.persistence_queue_drops,
                &self.persistence_outstanding,
                &self.metrics,
                PersistenceEntry::Audit(record),
            );
        }
    }
}

pub(super) struct ServiceLifecycleRecorder {
    pub(super) registry: Arc<RequestRegistry>,
    pub(super) event_delivery: EventDelivery,
}

impl LifecycleRecorder for ServiceLifecycleRecorder {
    fn record_terminal(&self, request_id: RequestId, outcome: TerminalOutcome) {
        self.event_delivery
            .metrics
            .record(LoggingMetric::LifecycleTerminal {
                outcome: logging_terminal_outcome(&outcome),
            });
        let request_id_string = request_id.as_uuid().to_string();
        let terminal = self.registry.terminalize(
            &request_id_string,
            outcome.as_str(),
            canonical_clock_timestamp(self.event_delivery.clock.as_ref()),
        );

        if let Ok(payload) = serde_json::to_string(&terminal_lifecycle_event(&outcome)) {
            let (summary_snapshots, terminal_summary) = terminal
                .map(|(snapshots, summary)| (Some(snapshots), Some(summary)))
                .unwrap_or((None, None));
            self.event_delivery.enqueue_with_summary_snapshots(
                request_id,
                ReplayChannel::Requests,
                payload,
                summary_snapshots,
                terminal_summary,
            );
        }
    }
}

fn logging_terminal_outcome(outcome: &TerminalOutcome) -> LoggingTerminalOutcome {
    match outcome {
        TerminalOutcome::Completed
        | TerminalOutcome::CompletedWithStatus(_)
        | TerminalOutcome::CompletedWithUsage { .. } => LoggingTerminalOutcome::Completed,
        TerminalOutcome::Failed(_) | TerminalOutcome::FailedWithStatus { .. } => {
            LoggingTerminalOutcome::Failed
        }
        TerminalOutcome::Rejected(_) | TerminalOutcome::RejectedWithStatus { .. } => {
            LoggingTerminalOutcome::Rejected
        }
        TerminalOutcome::Cancelled(_) => LoggingTerminalOutcome::Cancelled,
        TerminalOutcome::Dropped(_) => LoggingTerminalOutcome::Dropped,
    }
}

fn terminal_lifecycle_event(outcome: &TerminalOutcome) -> LifecycleEvent {
    match outcome {
        TerminalOutcome::Completed
        | TerminalOutcome::CompletedWithStatus(_)
        | TerminalOutcome::CompletedWithUsage { .. } => LifecycleEvent::Completed {
            status_code: match outcome {
                TerminalOutcome::CompletedWithStatus(status) => Some(*status),
                TerminalOutcome::CompletedWithUsage { status_code, .. } => Some(*status_code),
                _ => None,
            },
            duration_ms: None,
            usage: match outcome {
                TerminalOutcome::CompletedWithUsage { usage, .. } => Some(*usage),
                _ => None,
            },
        },
        TerminalOutcome::Failed(error) => LifecycleEvent::Failed {
            error: error.clone(),
            status_code: None,
        },
        TerminalOutcome::FailedWithStatus { error, status_code } => LifecycleEvent::Failed {
            error: error.clone(),
            status_code: Some(*status_code),
        },
        TerminalOutcome::Rejected(reason) => LifecycleEvent::Rejected {
            reason: reason.clone(),
            status_code: None,
        },
        TerminalOutcome::RejectedWithStatus {
            reason,
            status_code,
        } => LifecycleEvent::Rejected {
            reason: reason.clone(),
            status_code: Some(*status_code),
        },
        TerminalOutcome::Cancelled(reason) => LifecycleEvent::Cancelled {
            reason: reason.clone(),
        },
        TerminalOutcome::Dropped(reason) => LifecycleEvent::Dropped {
            reason: reason.clone(),
        },
    }
}

fn sanitize_noncanonical_payload(payload_json: String) -> String {
    match sanitize_lifecycle_event(LifecycleEvent::AuditError {
        message: payload_json,
    }) {
        LifecycleEvent::AuditError { message } => message,
        _ => unreachable!("sanitizing an audit event preserves its variant"),
    }
}

fn presentation_context_for(
    metadata: &RequestSummaryMetadata,
    event: &LifecycleEvent,
) -> CanonicalPresentationContext {
    let (event_model, event_method) = match event {
        LifecycleEvent::Admitted { model, method } => (model.as_deref(), method.as_deref()),
        LifecycleEvent::RouteSelected {
            model,
            provider: _,
            engine: _,
        }
        | LifecycleEvent::StreamStarted { model } => (model.as_deref(), None),
        _ => (None, None),
    };
    CanonicalPresentationContext::from_parts(
        metadata.route(),
        metadata.source(),
        metadata.model().or(event_model),
        metadata.provider(),
        metadata.engine(),
        metadata.method().or(event_method),
    )
}

pub(super) fn enqueue_event_with_delivery(
    event_delivery: &EventDelivery,
    request_id: RequestId,
    channel: ReplayChannel,
    payload_json: String,
    occurred_at: Option<String>,
    summary_snapshots: Option<RequestSummaryEventSnapshots>,
    terminal_summary: Option<RequestSummaryEntry>,
) -> EventId {
    let sequence = event_delivery.sequences.next(channel);
    let occurred_at =
        occurred_at.unwrap_or_else(|| canonical_clock_timestamp(event_delivery.clock.as_ref()));
    let event_id = EventId::new();
    let request_id_string = request_id.as_uuid().to_string();
    let registry_entry = event_delivery
        .registry
        .get_active(&request_id_string)
        .or_else(|| event_delivery.registry.get_recent(&request_id_string));
    let summary_snapshots = summary_snapshots.or_else(|| {
        registry_entry
            .as_ref()
            .map(RequestSummaryEventSnapshots::current)
    });
    let metadata = terminal_summary
        .as_ref()
        .map(RequestSummaryEntry::metadata)
        .filter(|metadata| !metadata.is_empty())
        .cloned()
        .or_else(|| {
            summary_snapshots.as_ref().and_then(|snapshots| {
                snapshots
                    .iter()
                    .filter(|snapshot| !snapshot.metadata().is_empty())
                    .map(|snapshot| snapshot.metadata().clone())
                    .last()
            })
        })
        .or_else(|| {
            // Match the two fallbacks above and skip empty metadata: an empty
            // `RequestSummaryMetadata` carries no route/source/kind, so building
            // a presentation context from it stamps `kind=unknown` onto every
            // message for the request. Leaving it `None` keeps the envelope
            // free of a misleading context instead.
            registry_entry
                .as_ref()
                .map(RequestSummaryEntry::metadata)
                .filter(|metadata| !metadata.is_empty())
                .cloned()
        });
    let canonical_envelope = serde_json::from_str::<LifecycleEvent>(&payload_json)
        .ok()
        .map(sanitize_lifecycle_event)
        .map(|event| {
            let context = metadata
                .as_ref()
                .map(|metadata| presentation_context_for(metadata, &event));
            let envelope = CanonicalEnvelope::new(
                event_id,
                request_id,
                channel,
                sequence,
                occurred_at.clone(),
                event,
            );
            if let Some(context) = context {
                envelope.with_presentation_context(context)
            } else {
                envelope
            }
        });
    let payload_json = canonical_envelope
        .as_ref()
        .and_then(|envelope| serde_json::to_string(&envelope.event).ok())
        .unwrap_or_else(|| sanitize_noncanonical_payload(payload_json));
    let mut entry = serde_json::json!({
        "request_id": request_id.as_uuid(),
        "channel": channel,
        "sequence": sequence,
        "occurred_at": occurred_at,
        "payload": payload_json,
    });
    if let Some(ref envelope) = canonical_envelope {
        let entry_object = entry
            .as_object_mut()
            .expect("logging bus entry is always a JSON object");
        entry_object.insert(
            "event_id".into(),
            serde_json::json!(envelope.event_id.as_uuid()),
        );
        entry_object.insert("canonical_envelope".into(), serde_json::json!(envelope));
        entry_object.insert(
            "presentation_summary".into(),
            serde_json::json!(
                envelope.presentation_local_summary_with_limit(event_delivery.summary_line_limit,)
            ),
        );
    }
    if let Some(summary_snapshots) = summary_snapshots {
        entry
            .as_object_mut()
            .expect("logging bus entry is always a JSON object")
            .insert(
                "request_summary_snapshots".into(),
                serde_json::json!(summary_snapshots),
            );
    }
    let entry_payload = entry.to_string();
    let channel_hint = match channel {
        ReplayChannel::Requests => 0,
        ReplayChannel::Operations => 1,
        ReplayChannel::System => 2,
    };
    let entry = BusEntry {
        payload: entry_payload,
        channel_hint,
    };
    let outcome = event_delivery.bus.push_replay(
        entry.payload.clone(),
        entry.channel_hint,
        ReplaySequence::next(channel, sequence),
    );
    emit_accepted_canonical_event(outcome, canonical_envelope.as_ref());
    if event_delivery.sink_enabled && !matches!(outcome, super::super::bus::PushOutcome::Rejected) {
        offer_persistence_to(
            &event_delivery.delivery,
            &event_delivery.persistence_queue_drops,
            &event_delivery.persistence_outstanding,
            &event_delivery.metrics,
            match terminal_summary {
                Some(summary) => PersistenceEntry::Terminal { entry, summary },
                None => PersistenceEntry::Bus(entry),
            },
        );
    }
    event_id
}
