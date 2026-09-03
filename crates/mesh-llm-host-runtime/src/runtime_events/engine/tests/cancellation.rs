use mesh_llm_runtime_event_contracts::{OperationId, Outcome, RuntimeEventIngress};

use super::fixtures::{synthetic_unknown, terminal_success};
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn explicit_cancellation_releases_without_a_terminal_or_wake_entry() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");

    reservation.cancel();

    assert_eq!(engine.drain().applied, 0);
    assert_eq!(engine.health().snapshot().terminal_delivery_failed, 0);
    assert_eq!(engine.health().snapshot().shutdown_degraded, 0);
}

#[test]
fn dropped_guard_with_no_terminal_synthesizes_terminal_not_delivered_unknown() {
    let engine = RuntimeEventEngine::with_capacity(4);
    {
        let _reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        // Guard drops here without ever submitting a terminal: this is the
        // "forgotten terminal" / crash-equivalent path.
    }

    let report = engine.drain();
    assert_eq!(report.applied, 1);

    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].fact.data().outcome,
        Some(Outcome::Unknown),
        "a forgotten terminal on an otherwise-successful path must degrade, never manufacture a failure outcome"
    );
}

#[test]
fn dropped_guard_after_a_real_terminal_does_not_double_synthesize() {
    let engine = RuntimeEventEngine::with_capacity(4);
    {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        assert_eq!(
            reservation.ingress().try_submit(terminal_success()),
            mesh_llm_runtime_event_contracts::SubmitOutcome::Accepted
        );
        // Guard drops after a real terminal was already written.
    }

    let report = engine.drain();
    assert_eq!(report.applied, 1);
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].fact.data().outcome, Some(Outcome::Success));
}
