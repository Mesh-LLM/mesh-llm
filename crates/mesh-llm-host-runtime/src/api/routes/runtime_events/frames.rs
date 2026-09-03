//! Wire payload types and SSE byte encoding for the v1 stream.
//!
//! Projection types are explicit `Serialize` structs — no `RuntimeFact`
//! ever derives `Serialize` for the wire (see `event_projection`, which
//! hand-projects only fields drawn from `EVENT_PROJECTION_ALLOWLIST`).
//!
//! KNOWN SCOPE LIMIT (documented, not silently papered over): the host
//! engine's `ReducerSnapshot` (task 4) does not yet retain per-category
//! domain state (which operations are models vs. stages vs. sessions), so
//! `models`/`stages`/`sessions`/`requests`/`devices`/`cache` are present as
//! empty arrays — structurally required, not yet domain-populated. `node`
//! is populated from genuinely available reducer/health data. Likewise the
//! engine's replay pipeline does not carry `producer`/`severity` through
//! from `RuntimeEventEnvelope` (only the bare `RuntimeFact` reaches
//! replay), so those two allowlisted keys are never emitted; every key this
//! module DOES emit is real, reducer-backed data.

use mesh_llm_runtime_event_contracts::{
    NumericValue, Outcome, ProgressUnit, ReasonCode, RuntimeFact,
};
use serde::Serialize;

use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::health::EngineHealthSnapshot;
use crate::runtime_events::replay::ReplayFrame;

use super::cursor::Cursor;
use super::recovery::{Gap, GapReason};
use super::state_projection;

/// The full per-event-kind projected JSON key allowlist (deny-by-default):
/// every key `event_projection` may ever emit, drawn from the task-1/3
/// inventory's `projected_event_keys`. `EventProjection`'s own `Serialize`
/// output is asserted a subset of this set by
/// `runtime_event_api_tests::event_projection_keys_are_a_subset_of_the_allowlist_for_every_submitted_kind`.
#[cfg(test)]
pub(super) const EVENT_PROJECTION_ALLOWLIST: &[&str] = &[
    "category",
    "kind",
    "producer",
    "severity",
    "summary",
    "scope",
    "state",
    "progress",
    "outcome",
    "reason_code",
    "duration_ms",
    "numeric_summaries",
];

#[cfg(test)]
pub(super) const REQUIRED_ENVELOPE_KEYS: &[&str] = &[
    "version",
    "cursor",
    "process_instance_id",
    "sequence",
    "rebuild_generation",
];

#[cfg(test)]
pub(super) const STATE_TOP_LEVEL_KEYS: &[&str] = &[
    "node", "models", "stages", "sessions", "requests", "devices", "cache",
];

#[derive(Debug, Serialize)]
pub(super) struct Envelope<T> {
    pub(super) version: u8,
    pub(super) cursor: String,
    pub(super) process_instance_id: String,
    pub(super) sequence: u64,
    pub(super) rebuild_generation: u64,
    #[serde(flatten)]
    pub(super) body: T,
}

fn envelope<T>(cursor: Cursor, rebuild_generation: u64, body: T) -> Envelope<T> {
    Envelope {
        version: 1,
        cursor: cursor.encode(),
        process_instance_id: cursor.process_instance.as_uuid().to_string(),
        sequence: cursor.sequence,
        rebuild_generation,
        body,
    }
}

fn encode<T: Serialize>(event: &'static str, cursor: Cursor, payload: &Envelope<T>) -> String {
    let json = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
    format!("id: {}\nevent: {event}\ndata: {json}\n\n", cursor.encode())
}

pub(super) const KEEPALIVE_FRAME: &str = ": keepalive\n\n";

// ─── runtime_state ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(super) struct StateBody {
    pub(super) state: state_projection::StateProjection,
}

pub(super) fn state_frame(engine: &RuntimeEventEngine, cursor: Cursor) -> String {
    let rebuild_generation = engine.health().snapshot().rebuild_generation;
    let payload = envelope(
        cursor,
        rebuild_generation,
        StateBody {
            state: state_projection::build(engine),
        },
    );
    encode("runtime_state", cursor, &payload)
}

// ─── runtime_health ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(super) struct HealthProjection {
    pub(super) rebuild_generation: u64,
    pub(super) reservation_exhausted: u64,
    pub(super) terminal_delivery_failed: u64,
    pub(super) dropped_progress: u64,
    pub(super) dropped_diagnostic: u64,
    pub(super) replay_evicted: u64,
    pub(super) subscriber_disconnected: u64,
    pub(super) shutdown_degraded: u64,
    pub(super) reducer_rejected: u64,
}

impl From<EngineHealthSnapshot> for HealthProjection {
    fn from(snapshot: EngineHealthSnapshot) -> Self {
        Self {
            rebuild_generation: snapshot.rebuild_generation,
            reservation_exhausted: snapshot.reservation_exhausted,
            terminal_delivery_failed: snapshot.terminal_delivery_failed,
            dropped_progress: snapshot.dropped_progress,
            dropped_diagnostic: snapshot.dropped_diagnostic,
            replay_evicted: snapshot.replay_evicted,
            subscriber_disconnected: snapshot.subscriber_disconnected,
            shutdown_degraded: snapshot.shutdown_degraded,
            reducer_rejected: snapshot.reducer_rejected,
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct HealthBody {
    pub(super) health: HealthProjection,
}

pub(super) fn health_frame(engine: &RuntimeEventEngine, cursor: Cursor) -> String {
    let snapshot = engine.health().snapshot();
    let payload = envelope(
        cursor,
        snapshot.rebuild_generation,
        HealthBody {
            health: snapshot.into(),
        },
    );
    encode("runtime_health", cursor, &payload)
}

// ─── runtime_event ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(super) struct ScopeProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) topology_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stage_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stage_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) device_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct StateTransitionProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) previous: Option<String>,
    pub(super) current: String,
}

#[derive(Debug, Serialize)]
pub(super) struct ProgressProjection {
    pub(super) current: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) total: Option<u64>,
    pub(super) unit: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct NumericSummaryProjection {
    pub(super) key: String,
    pub(super) value: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub(super) struct EventProjection {
    pub(super) category: &'static str,
    pub(super) kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) scope: Option<ScopeProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) state: Option<StateTransitionProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) progress: Option<ProgressProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) outcome: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(super) numeric_summaries: Vec<NumericSummaryProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) summary: Option<String>,
}

pub(super) fn category(fact: &RuntimeFact) -> &'static str {
    match fact {
        RuntimeFact::NativeRuntime(_) => "native_runtime",
        RuntimeFact::ModelPreparation(_) => "model_preparation",
        RuntimeFact::ModelLoading(_) => "model_loading",
        RuntimeFact::ModelAvailability(_) => "model_availability",
        RuntimeFact::ModelUnloading(_) => "model_unloading",
        RuntimeFact::StageTopology(_) => "stage_topology",
        RuntimeFact::Session(_) => "session",
        RuntimeFact::Request(_) => "request",
        RuntimeFact::Prefill(_) => "prefill",
        RuntimeFact::Generation(_) => "generation",
        RuntimeFact::KvRuntimeState(_) => "kv_runtime_state",
        RuntimeFact::ResourceHealth(_) => "resource_health",
        RuntimeFact::Diagnostic(_) => "diagnostic",
        RuntimeFact::NodeAvailability(_) => "node_availability",
        RuntimeFact::EventSystemHealth(_) => "event_system_health",
    }
}

fn outcome_str(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Success => "success",
        Outcome::Failure => "failure",
        Outcome::Rejected => "rejected",
        Outcome::Cancelled => "cancelled",
        Outcome::Unknown => "unknown",
    }
}

fn reason_code_str(reason: &ReasonCode) -> String {
    match reason {
        ReasonCode::InvalidConfiguration => "invalid_configuration".to_string(),
        ReasonCode::UnsupportedCapability => "unsupported_capability".to_string(),
        ReasonCode::MissingArtifact => "missing_artifact".to_string(),
        ReasonCode::ArtifactIoFailure => "artifact_io_failure".to_string(),
        ReasonCode::ModelFormatOrLoadFailure => "model_format_or_load_failure".to_string(),
        ReasonCode::BackendInitializationFailure => "backend_initialization_failure".to_string(),
        ReasonCode::DeviceUnavailable => "device_unavailable".to_string(),
        ReasonCode::ResourceAllocationFailure => "resource_allocation_failure".to_string(),
        ReasonCode::OutOfMemory => "out_of_memory".to_string(),
        ReasonCode::ContextExhausted => "context_exhausted".to_string(),
        ReasonCode::StageUnavailable => "stage_unavailable".to_string(),
        ReasonCode::Timeout => "timeout".to_string(),
        ReasonCode::Cancellation => "cancellation".to_string(),
        ReasonCode::ProcessCrash => "process_crash".to_string(),
        ReasonCode::IncompatibleAbiOrFeatureSet => "incompatible_abi_or_feature_set".to_string(),
        ReasonCode::InternalRuntimeFailure => "internal_runtime_failure".to_string(),
        ReasonCode::UnknownFailure => "unknown_failure".to_string(),
        ReasonCode::TerminalNotDelivered => "terminal_not_delivered".to_string(),
        ReasonCode::ReservationExhausted => "reservation_exhausted".to_string(),
        ReasonCode::Unknown(code) => code.as_str().to_string(),
    }
}

fn progress_unit_str(unit: ProgressUnit) -> &'static str {
    match unit {
        ProgressUnit::None => "none",
        ProgressUnit::Bytes => "bytes",
        ProgressUnit::Items => "items",
        ProgressUnit::Tensors => "tensors",
        ProgressUnit::Steps => "steps",
        ProgressUnit::Tokens => "tokens",
    }
}

fn numeric_value_json(value: NumericValue) -> serde_json::Value {
    match value {
        NumericValue::Unsigned(value) => serde_json::Value::from(value),
        NumericValue::Signed(value) => serde_json::Value::from(value),
        NumericValue::Floating(value) => serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
    }
}

pub(super) fn event_projection(fact: &RuntimeFact) -> EventProjection {
    let data = fact.data();
    let scope = &data.scope;
    let scope_projection = if scope.model_id.is_some()
        || scope.topology_id.is_some()
        || scope.stage.is_some()
        || scope.session_id.is_some()
        || scope.request_id.is_some()
        || scope.device_id.is_some()
    {
        Some(ScopeProjection {
            model_id: scope.model_id.as_ref().map(|id| id.as_str().to_string()),
            topology_id: scope.topology_id.as_ref().map(|id| id.as_str().to_string()),
            stage_id: scope
                .stage
                .as_ref()
                .map(|stage| stage.id.as_str().to_string()),
            stage_index: scope.stage.as_ref().map(|stage| stage.index),
            session_id: scope.session_id.as_ref().map(|id| id.as_str().to_string()),
            request_id: scope.request_id.as_ref().map(|id| id.as_str().to_string()),
            device_id: scope.device_id.as_ref().map(|id| id.as_str().to_string()),
        })
    } else {
        None
    };

    EventProjection {
        category: category(fact),
        kind: fact.kind_id(),
        scope: scope_projection,
        state: data
            .state
            .as_ref()
            .map(|transition| StateTransitionProjection {
                previous: transition
                    .previous
                    .as_ref()
                    .map(|state| state.as_str().to_string()),
                current: transition.current.as_str().to_string(),
            }),
        progress: data.progress.map(|progress| ProgressProjection {
            current: progress.current,
            total: progress.total,
            unit: progress_unit_str(progress.unit),
        }),
        outcome: data.outcome.map(outcome_str),
        reason_code: data.reason.as_ref().map(reason_code_str),
        duration_ms: data
            .duration
            .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)),
        numeric_summaries: data
            .numeric_summaries
            .as_slice()
            .iter()
            .map(|summary| NumericSummaryProjection {
                key: summary.key.as_str().to_string(),
                value: numeric_value_json(summary.value),
            })
            .collect(),
        summary: data
            .summary
            .as_ref()
            .map(|summary| summary.as_str().to_string()),
    }
}

#[derive(Debug, Serialize)]
pub(super) struct EventBody {
    pub(super) event: EventProjection,
}

pub(super) fn event_frame(engine: &RuntimeEventEngine, frame: &ReplayFrame) -> String {
    let cursor = Cursor::new(engine.process_instance(), frame.sequence.get());
    let payload = envelope(
        cursor,
        frame.rebuild_generation,
        EventBody {
            event: event_projection(&frame.fact),
        },
    );
    encode("runtime_event", cursor, &payload)
}

// ─── runtime_replay_gap ─────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(super) struct ReplayGapBody {
    pub(super) requested_cursor: String,
    pub(super) reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) oldest_available_cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) latest_cursor: Option<String>,
}

impl GapReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::StaleInstance => "stale_instance",
            Self::Evicted => "evicted",
        }
    }
}

pub(super) fn replay_gap_frame(engine: &RuntimeEventEngine, gap: &Gap) -> String {
    let instance = engine.process_instance();
    let current_cursor = Cursor::new(instance, gap.latest.or(gap.oldest_available).unwrap_or(0));
    let payload = envelope(
        current_cursor,
        engine.health().snapshot().rebuild_generation,
        ReplayGapBody {
            requested_cursor: gap.requested.encode(),
            reason: gap.reason.as_str(),
            oldest_available_cursor: gap
                .oldest_available
                .map(|sequence| Cursor::new(instance, sequence).encode()),
            latest_cursor: gap
                .latest
                .map(|sequence| Cursor::new(instance, sequence).encode()),
        },
    );
    encode("runtime_replay_gap", current_cursor, &payload)
}
