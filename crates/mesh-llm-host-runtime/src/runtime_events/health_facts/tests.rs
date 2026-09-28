use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{
    EventSystemHealthEventKind, NumericValue, OperationId, OperationScope, RuntimeEventIngress,
    RuntimeFact,
};

use super::{DELTA_KEY, HealthFactProducer, TOTAL_KEY};
use crate::runtime_event_api::state_projection;
use crate::runtime_events::config::HEALTH_PUBLISH_MIN_INTERVAL;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::ingress::NON_TERMINAL_CREDITS;

fn health_facts(engine: &RuntimeEventEngine) -> Vec<(EventSystemHealthEventKind, u64, u64)> {
    engine
        .replay()
        .snapshot()
        .iter()
        .filter_map(|frame| match frame.fact.as_ref() {
            RuntimeFact::EventSystemHealth(fact) => {
                let summary = |key: &str| {
                    fact.data()
                        .numeric_summaries
                        .as_slice()
                        .iter()
                        .find(|summary| summary.key.as_str() == key)
                        .map(|summary| match summary.value {
                            NumericValue::Unsigned(value) => value,
                            NumericValue::Signed(_) | NumericValue::Floating(_) => 0,
                        })
                        .unwrap_or_default()
                };
                Some((*fact.kind(), summary(DELTA_KEY), summary(TOTAL_KEY)))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn counter_growth_emits_one_fact_of_the_matching_kind() {
    let engine = RuntimeEventEngine::new();
    let mut producer = HealthFactProducer::new();
    engine.health().bump_coalesced_progress_by(3);
    engine.health().bump_coalesced_progress_by(2);

    assert_eq!(producer.observe(&engine, Instant::now()), 1);
    engine.drain();

    assert_eq!(
        health_facts(&engine),
        vec![(EventSystemHealthEventKind::EventsCoalesced, 5, 5)]
    );
}

#[test]
fn each_watched_counter_maps_to_its_kind() {
    let engine = RuntimeEventEngine::new();
    let mut producer = HealthFactProducer::new();
    let health = engine.health();
    health.bump_dropped_diagnostic();
    health.bump_dropped_native_by(2);
    health.bump_subscriber_disconnected();
    health.bump_reducer_rejected();
    health.bump_rejected_native_by(4);

    assert_eq!(producer.observe(&engine, Instant::now()), 4);
    engine.drain();

    let mut facts = health_facts(&engine);
    facts.sort_by_key(|(kind, _, _)| kind.as_str());
    assert_eq!(
        facts,
        vec![
            (EventSystemHealthEventKind::EventsDroppedByClass, 3, 3),
            (EventSystemHealthEventKind::ReducerError, 1, 1),
            (EventSystemHealthEventKind::SubscriberDisconnected, 1, 1),
            (EventSystemHealthEventKind::UnknownNativeEventReceived, 4, 4),
        ]
    );
}

#[test]
fn no_growth_emits_nothing() {
    let engine = RuntimeEventEngine::new();
    let mut producer = HealthFactProducer::new();
    let start = Instant::now();

    assert_eq!(producer.observe(&engine, start), 0);
    engine.health().bump_reservation_exhausted();
    engine.health().bump_replay_evicted();
    assert_eq!(
        producer.observe(&engine, start + HEALTH_PUBLISH_MIN_INTERVAL),
        0,
        "counters with no mapped kind never produce a fact"
    );
    engine.drain();
    assert!(health_facts(&engine).is_empty());
}

#[test]
fn a_kind_emits_at_most_once_per_interval_and_reports_the_accumulated_growth() {
    let engine = RuntimeEventEngine::new();
    let mut producer = HealthFactProducer::new();
    let start = Instant::now();

    engine.health().bump_coalesced_progress_by(1);
    assert_eq!(producer.observe(&engine, start), 1);
    engine.health().bump_coalesced_progress_by(2);
    assert_eq!(
        producer.observe(&engine, start + Duration::from_millis(500)),
        0,
        "inside the interval the growth waits"
    );
    engine.health().bump_coalesced_progress_by(4);
    assert_eq!(
        producer.observe(&engine, start + HEALTH_PUBLISH_MIN_INTERVAL),
        1
    );
    engine.drain();

    assert_eq!(
        health_facts(&engine),
        vec![
            (EventSystemHealthEventKind::EventsCoalesced, 1, 1),
            (EventSystemHealthEventKind::EventsCoalesced, 6, 7),
        ]
    );
}

#[test]
fn the_interval_is_per_kind() {
    let engine = RuntimeEventEngine::new();
    let mut producer = HealthFactProducer::new();
    let start = Instant::now();

    engine.health().bump_coalesced_progress_by(1);
    assert_eq!(producer.observe(&engine, start), 1);
    engine.health().bump_reducer_rejected();
    assert_eq!(
        producer.observe(&engine, start + Duration::from_millis(10)),
        1,
        "a different kind is not held back by another kind's emission"
    );
}

/// With the ring full, the producer's own submission is refused and bumps
/// `state_transition_rejected`. That is the counter `reducer_error`
/// watches, so without crediting it the producer would report its own
/// refusal every interval for as long as the ring stays full.
#[test]
fn a_refused_health_fact_never_reports_itself() {
    let engine = RuntimeEventEngine::new();
    for _ in 0..NON_TERMINAL_CREDITS {
        let scope = OperationScope::root_only(OperationId::new());
        let _ = engine
            .unreserved_ingress(scope)
            .try_submit(super::health_fact(&super::WATCHED[0], &[0], &[1]));
    }
    let mut producer = HealthFactProducer::new();
    let start = Instant::now();
    engine.health().bump_coalesced_progress_by(1);

    assert_eq!(producer.observe(&engine, start), 1);
    let rejected_after_first = engine.health().snapshot().state_transition_rejected;
    assert_eq!(rejected_after_first, 1, "the ring was full");

    for step in 1..=5 {
        assert_eq!(
            producer.observe(&engine, start + HEALTH_PUBLISH_MIN_INTERVAL * step),
            0,
            "the producer's own refusal must not become new growth"
        );
    }
    assert_eq!(
        engine.health().snapshot().state_transition_rejected,
        rejected_after_first
    );
}

#[test]
fn health_growth_reaches_runtime_state_node_event_system() {
    let engine = RuntimeEventEngine::new();
    let mut producer = HealthFactProducer::new();
    let start = Instant::now();

    engine.health().bump_coalesced_progress_by(52);
    engine.health().bump_dropped_diagnostic();
    producer.observe(&engine, start);
    engine.drain();
    engine.health().bump_coalesced_progress_by(8);
    producer.observe(&engine, start + HEALTH_PUBLISH_MIN_INTERVAL);
    engine.drain();

    let value = serde_json::to_value(&state_projection::build(&engine).node).expect("serializable");
    let counts = &value["event_system"]["counts_by_kind"];
    assert_eq!(counts["events_coalesced"], 60);
    assert_eq!(counts["events_dropped_by_class"], 1);
    assert_eq!(value["event_system"]["last_kind"], "events_coalesced");
}
