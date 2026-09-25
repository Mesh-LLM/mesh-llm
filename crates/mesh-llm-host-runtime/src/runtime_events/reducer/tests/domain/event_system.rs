//! `node.event_system`: counts by wire kind and latest signals.

use mesh_llm_runtime_event_contracts::{EventSystemHealthEventKind, FamilyFact, RuntimeFact};

use super::super::fixtures::apply_each_on_fresh_root;
use crate::runtime_events::reducer::ReducerSnapshot;

fn health(kind: EventSystemHealthEventKind) -> RuntimeFact {
    RuntimeFact::EventSystemHealth(FamilyFact::new(kind))
}

#[test]
fn a_fresh_reducer_reports_no_event_system_health() {
    assert!(ReducerSnapshot::empty().domain().event_system().is_empty());
}

#[test]
fn counts_are_kept_per_wire_kind_id() {
    let snapshot = apply_each_on_fresh_root(vec![
        health(EventSystemHealthEventKind::EventsCoalesced),
        health(EventSystemHealthEventKind::EventsCoalesced),
        health(EventSystemHealthEventKind::ReducerError),
    ]);

    let event_system = snapshot.domain().event_system();
    assert_eq!(
        event_system.counts_by_kind.get("events_coalesced"),
        Some(&2)
    );
    assert_eq!(event_system.counts_by_kind.get("reducer_error"), Some(&1));
    assert_eq!(event_system.last_kind, Some("reducer_error"));
}

#[test]
fn lagging_subscribers_rise_on_lag_and_fall_saturating_on_disconnect() {
    let snapshot = apply_each_on_fresh_root(vec![
        health(EventSystemHealthEventKind::SubscriberLagging),
        health(EventSystemHealthEventKind::SubscriberDisconnected),
        health(EventSystemHealthEventKind::SubscriberDisconnected),
        health(EventSystemHealthEventKind::SubscriberLagging),
    ]);

    assert_eq!(snapshot.domain().event_system().lagging_subscribers, 1);
}

#[test]
fn exporter_state_and_pressure_are_latest_wins() {
    let snapshot = apply_each_on_fresh_root(vec![
        health(EventSystemHealthEventKind::TelemetryExporterDegraded),
        health(EventSystemHealthEventKind::IngressQueuePressure),
        health(EventSystemHealthEventKind::TelemetryExporterRecovered),
    ]);

    let event_system = snapshot.domain().event_system();
    assert_eq!(event_system.telemetry_exporter, Some("recovered"));
    assert_eq!(event_system.pressure.as_deref(), Some("pressure"));
}

fn health_with_delta(kind: EventSystemHealthEventKind, delta: u64) -> RuntimeFact {
    use mesh_llm_runtime_event_contracts::{
        BoundedNumericSummaries, FactData, NumericSummary, NumericSummaryKey, NumericValue,
    };
    let key =
        NumericSummaryKey::new(crate::runtime_events::health_facts::DELTA_KEY).expect("static key");
    RuntimeFact::EventSystemHealth(FamilyFact::with_data(
        kind,
        FactData {
            numeric_summaries: BoundedNumericSummaries::new(vec![NumericSummary::new(
                key,
                NumericValue::Unsigned(delta),
            )])
            .expect("one summary"),
            ..FactData::default()
        },
    ))
}

#[test]
fn a_fact_carrying_a_delta_counts_that_many_occurrences() {
    let snapshot = apply_each_on_fresh_root(vec![
        health_with_delta(EventSystemHealthEventKind::EventsCoalesced, 52),
        health_with_delta(EventSystemHealthEventKind::EventsCoalesced, 8),
        health(EventSystemHealthEventKind::EventsCoalesced),
    ]);

    assert_eq!(
        snapshot
            .domain()
            .event_system()
            .counts_by_kind
            .get("events_coalesced"),
        Some(&61)
    );
}

#[test]
fn a_disconnect_delta_retires_that_many_laggers() {
    let snapshot = apply_each_on_fresh_root(vec![
        health_with_delta(EventSystemHealthEventKind::SubscriberLagging, 3),
        health_with_delta(EventSystemHealthEventKind::SubscriberDisconnected, 2),
    ]);

    assert_eq!(snapshot.domain().event_system().lagging_subscribers, 1);
}
