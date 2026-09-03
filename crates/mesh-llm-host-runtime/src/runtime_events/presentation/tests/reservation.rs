//! Terminal and health events are reserved: forwarded immediately, every
//! single one, never dropped, coalesced, or starved behind a progress
//! flood on unrelated operations.

use std::sync::Arc;
use std::time::Instant;

use mesh_llm_runtime_event_contracts::{EventSequence, OperationScope, RuntimeFact};

use super::super::coalescer::ProgressCoalescer;
use super::super::subscriber::{flush_tick, route_fact};
use super::{RecordingSink, progress_fact, root_scope, terminal_fact};
use crate::runtime_events::health::EngineHealth;
use crate::runtime_events::replay::ReplayFrame;

fn frame_for(scope: OperationScope, fact: RuntimeFact) -> ReplayFrame {
    ReplayFrame {
        sequence: EventSequence::new(1),
        rebuild_generation: 0,
        scope,
        fact: Arc::new(fact),
        recorded_at: Instant::now(),
    }
}

#[test]
fn a_terminal_frame_is_emitted_immediately_and_never_buffered_in_the_progress_coalescer() {
    let coalescer = ProgressCoalescer::new();
    let sink = RecordingSink::default();
    let frame = frame_for(root_scope(), terminal_fact());

    route_fact(&coalescer, &sink, &frame);

    assert_eq!(
        coalescer.pending_len(),
        0,
        "a terminal fact must never enter the progress coalescer"
    );
    let emitted = sink.drain();
    assert_eq!(
        emitted.len(),
        1,
        "the terminal fact reaches the sink on its own submission, with no tick required"
    );
}

#[test]
fn a_terminal_frame_is_forwarded_even_while_a_progress_flood_is_pending_for_other_operations() {
    let coalescer = ProgressCoalescer::new();
    let sink = RecordingSink::default();
    for _ in 0..500 {
        let (scope, fact) = progress_fact(root_scope(), 1, Some(2));
        coalescer.submit(scope, fact);
    }

    let terminal_frame = frame_for(root_scope(), terminal_fact());
    route_fact(&coalescer, &sink, &terminal_frame);

    let emitted = sink.drain();
    assert_eq!(
        emitted.len(),
        1,
        "a terminal event is never starved behind a pending progress flood on unrelated operations"
    );
}

#[test]
fn health_snapshot_is_forwarded_on_flush_without_touching_the_progress_coalescer() {
    let coalescer = ProgressCoalescer::new();
    let health = EngineHealth::default();
    let sink = RecordingSink::default();
    health.bump_reservation_exhausted();

    let now = Instant::now();
    flush_tick(&coalescer, &health, &sink, now);

    let emitted = sink.drain();
    assert!(
        emitted.iter().any(|event| matches!(
            event,
            mesh_llm_events::OutputEvent::Info { context, .. }
                if context.as_deref() == Some("event_system_health")
        )),
        "the coalesced health snapshot must reach the sink on the same tick that flushes progress"
    );
    assert_eq!(
        coalescer.pending_len(),
        0,
        "flushing health must not leave anything behind in the progress coalescer"
    );
}

#[test]
fn health_snapshot_cadence_gate_is_never_bypassed_by_repeated_flush_ticks() {
    let coalescer = ProgressCoalescer::new();
    let health = EngineHealth::default();
    let sink = RecordingSink::default();
    health.bump_reservation_exhausted();

    let now = Instant::now();
    flush_tick(&coalescer, &health, &sink, now);
    flush_tick(&coalescer, &health, &sink, now);

    let health_events = sink
        .drain()
        .into_iter()
        .filter(|event| {
            matches!(
                event,
                mesh_llm_events::OutputEvent::Info { context, .. }
                    if context.as_deref() == Some("event_system_health")
            )
        })
        .count();
    assert_eq!(
        health_events, 1,
        "EngineHealth::publish_at's cadence gate must not be bypassed by repeated flush ticks"
    );
}
