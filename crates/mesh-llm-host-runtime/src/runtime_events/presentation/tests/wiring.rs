//! Proves the presentation subscriber is actually attached to a live
//! engine, synchronously and with no scheduler race, and that its drive
//! loop pumps the engine (`RuntimeEventEngine::drain`) on every tick so
//! queued terminal facts reach the sink without any external explicit
//! `engine.drain()` call -- the two things `run_auto.rs`'s wiring depends
//! on being true.

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
async fn the_drive_loop_drains_the_engine_on_every_tick_so_a_queued_terminal_fact_reaches_the_sink()
{
    let engine = RuntimeEventEngine::new();
    let sink = Arc::new(RecordingSink::default());
    let subscription = attach(&engine).expect("attach");

    // Submit a terminal fact but deliberately never call `engine.drain()`
    // ourselves -- production has no other periodic drain call site (see
    // the module doc on `drive_presentation_subscriber`), so the drive
    // loop's own tick must be the thing that applies and publishes it.
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reservation");
    let outcome = reservation.ingress().try_submit(terminal_fact());
    assert!(matches!(
        outcome,
        mesh_llm_runtime_event_contracts::SubmitOutcome::Accepted
    ));

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
    let emitted = sink.drain();
    assert!(
        !emitted.is_empty(),
        "the drive loop's own tick must call engine.drain() so a queued terminal fact \
         reaches the sink without any external explicit drain() call"
    );
}
