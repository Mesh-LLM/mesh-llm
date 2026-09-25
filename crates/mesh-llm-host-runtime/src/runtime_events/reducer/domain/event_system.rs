//! `node.event_system`: counts and latest signals from the event system's
//! own health facts.
//!
//! The producer is `runtime_events::health_facts`, driven from the engine
//! driver: it reports growth in engine health counters, rate-limited per
//! kind, with the growth itself in the `delta` numeric summary. One fact
//! can therefore stand for many occurrences, so counts add the fact's
//! `delta` rather than one per fact. A fact without a `delta` (any other
//! producer) counts as a single occurrence.

use std::collections::BTreeMap;

use mesh_llm_runtime_event_contracts::{EventSystemHealthEventKind, FactData, NumericValue};

use crate::runtime_events::health_facts::DELTA_KEY;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EventSystemHealthDomainState {
    /// Wire kind id -> occurrences. Bounded by the closed kind enum.
    pub counts_by_kind: BTreeMap<&'static str, u64>,
    /// Latest ingress-queue pressure: the producer's state name when it
    /// supplies one, else `"pressure"`.
    pub pressure: Option<String>,
    /// `SubscriberLagging` increments, `SubscriberDisconnected` decrements
    /// (saturating at zero): a lagging subscriber is either disconnected by
    /// the lag policy or keeps lagging, so a disconnect retires one lagger.
    pub lagging_subscribers: u64,
    pub telemetry_exporter: Option<&'static str>,
    pub last_kind: Option<&'static str>,
}

impl EventSystemHealthDomainState {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// Occurrences one fact stands for: its unsigned `delta` summary, else 1.
fn occurrences(data: &FactData) -> u64 {
    data.numeric_summaries
        .as_slice()
        .iter()
        .find(|summary| summary.key.as_str() == DELTA_KEY)
        .and_then(|summary| match summary.value {
            NumericValue::Unsigned(value) => Some(value),
            NumericValue::Signed(_) | NumericValue::Floating(_) => None,
        })
        .unwrap_or(1)
}

pub(super) fn apply_event_system_health(
    state: &mut EventSystemHealthDomainState,
    kind: EventSystemHealthEventKind,
    data: &FactData,
) {
    let id = kind.as_str();
    let occurrences = occurrences(data);
    let count = state.counts_by_kind.entry(id).or_insert(0);
    *count = count.saturating_add(occurrences);
    state.last_kind = Some(id);
    match kind {
        EventSystemHealthEventKind::IngressQueuePressure => {
            state.pressure = Some(data.state.as_ref().map_or_else(
                || "pressure".to_string(),
                |transition| transition.current.as_str().to_string(),
            ));
        }
        EventSystemHealthEventKind::SubscriberLagging => {
            state.lagging_subscribers = state.lagging_subscribers.saturating_add(occurrences);
        }
        EventSystemHealthEventKind::SubscriberDisconnected => {
            state.lagging_subscribers = state.lagging_subscribers.saturating_sub(occurrences);
        }
        EventSystemHealthEventKind::TelemetryExporterDegraded => {
            state.telemetry_exporter = Some("degraded");
        }
        EventSystemHealthEventKind::TelemetryExporterRecovered => {
            state.telemetry_exporter = Some("recovered");
        }
        EventSystemHealthEventKind::EventsCoalesced
        | EventSystemHealthEventKind::EventsSampled
        | EventSystemHealthEventKind::EventsDroppedByClass
        | EventSystemHealthEventKind::ReducerError
        | EventSystemHealthEventKind::EventSchemaIncompatibility
        | EventSystemHealthEventKind::UnknownNativeEventReceived => {}
    }
}
