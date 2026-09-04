//! Proves the presentation subscriber is actually attached to a live
//! engine, synchronously and with no scheduler race, and that a fact the
//! engine-owned driver (`runtime_events::driver`, spawned right alongside
//! this subscriber in `run_auto.rs`) applies and publishes reaches the
//! sink through the subscriber's own subscription -- with no drain() call
//! anywhere in this subscriber or this test: draining is exclusively the
//! driver's job as of task 3.

use std::sync::Arc;
use std::time::Duration;

use mesh_llm_runtime_event_contracts::{OperationId, RuntimeEventIngress};

use super::super::subscriber::{
    attach, drive_presentation_subscriber, spawn_presentation_subscriber,
};
use super::{RecordingSink, terminal_fact};
use crate::runtime_events::engine::RuntimeEventEngine;

#[tokio::test]
async fn spawn_presentation_subscriber_registers_a_live_subscription_before_returning() {
    let engine = RuntimeEventEngine::new();
    assert_eq!(engine.subscribers().active_count(), 0);

    let handle = spawn_presentation_subscriber(&engine).expect("attach succeeds");

    assert_eq!(
        engine.subscribers().active_count(),
        1,
        "spawn_presentation_subscriber must register its subscription on the engine \
         synchronously, before it returns -- with no scheduler race for a caller to observe"
    );

    handle.abort();
}

#[tokio::test(start_paused = true)]
async fn a_fact_the_engine_owned_driver_applies_reaches_the_sink_via_the_subscription() {
    let engine = RuntimeEventEngine::new();
    let sink = Arc::new(RecordingSink::default());
    let subscription = attach(&engine).expect("attach");

    // Submit a terminal fact and spawn the SAME `runtime_events::driver`
    // task `run_auto.rs` spawns right alongside this subscriber -- proving
    // the subscriber needs nothing of its own to observe it, only the
    // driver applying and publishing it upstream of this subscription.
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reservation");
    let outcome = reservation.ingress().try_submit(terminal_fact());
    assert!(matches!(
        outcome,
        mesh_llm_runtime_event_contracts::SubmitOutcome::Accepted
    ));

    let driver = crate::runtime_events::driver::spawn_engine_driver(Arc::clone(&engine));
    let drive_engine = Arc::clone(&engine);
    let drive_sink: Arc<dyn super::super::subscriber::PresentationSink> = sink.clone();
    let drive = tokio::spawn(drive_presentation_subscriber(
        subscription,
        drive_engine,
        drive_sink,
    ));

    tokio::time::advance(Duration::from_millis(50)).await;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    drive.abort();
    driver.abort();
    let emitted = sink.drain();
    assert!(
        !emitted.is_empty(),
        "the engine-owned driver must apply and publish a queued terminal fact so the \
         presentation subscriber's own subscription receives and routes it"
    );
}
