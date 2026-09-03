//! Wires the progress coalescer and privacy-safe projection to the host
//! runtime-event engine's subscriber registry, producing at most one
//! progress `OutputEvent` per operation per render tick plus every terminal
//! and coalesced-health event, all BEFORE they ever reach
//! `mesh_llm_events::emit_event` -- the TUI's own unbounded
//! `OutputCommand::Event` channel (`mesh-llm-tui`) is never touched by this
//! module and never coalesces anything itself.
//!
//! Not yet spawned from process startup: `runtime/run_auto.rs` and
//! `runtime/local_model_only.rs` install the engine (task 9) but attach no
//! persistent subscriber themselves, matching the plan's existing
//! precedent that a subscriber attaches only where a consuming task
//! explicitly wires it (task 13's SSE route subscribes per HTTP
//! connection, not at startup). Spawning [`run_presentation_subscriber`]
//! into the shared runtime-mode startup path is left to that wiring task so
//! `local_model_only.rs`'s documented "zero management subscribers"
//! invariant is not disturbed by this change.

use std::sync::Arc;
use std::time::Instant;

use mesh_llm_events::OutputEvent;
use mesh_llm_runtime_event_contracts::DeliveryClass;
use tokio::sync::broadcast::error::RecvError;
use tokio::time::MissedTickBehavior;

use crate::runtime_events::config::TUI_RENDER_TICK;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::health::EngineHealth;
use crate::runtime_events::replay::ReplayFrame;
use crate::runtime_events::subscribers::SubscribeError;

use super::coalescer::ProgressCoalescer;
use super::projection::{fact_projection_event, health_projection_event};

/// Where a presentation projection lands. Production wiring targets
/// [`EmitEventSink`]; tests inject a bounded recorder so assertions never
/// depend on the global `mesh_llm_events::OutputManager`.
pub trait PresentationSink: Send + Sync {
    fn emit(&self, event: OutputEvent);
}

/// Production sink: forwards to `mesh_llm_events::emit_event`, the same
/// entry point every hand-written `OutputEvent` call site in this crate
/// uses. A write failure is logged, never propagated -- presentation is
/// never domain authority and must not affect primary work.
pub struct EmitEventSink;

impl PresentationSink for EmitEventSink {
    fn emit(&self, event: OutputEvent) {
        if let Err(error) = mesh_llm_events::emit_event(event) {
            tracing::warn!("presentation projection emit failed: {error}");
        }
    }
}

/// Route one accepted `ReplayFrame`: a `Progress`-class fact coalesces into
/// `coalescer` (see its own bounded, latest-value-wins contract); every
/// other delivery class -- most importantly `Terminal` -- is projected and
/// forwarded to `sink` immediately, with no buffering step of any kind.
pub(super) fn route_fact(
    coalescer: &ProgressCoalescer,
    sink: &dyn PresentationSink,
    frame: &ReplayFrame,
) {
    match frame.fact.delivery_class() {
        DeliveryClass::Progress => coalescer.submit(frame.scope, (*frame.fact).clone()),
        DeliveryClass::Terminal | DeliveryClass::StateTransition | DeliveryClass::Diagnostic => {
            sink.emit(fact_projection_event(&frame.fact));
        }
    }
}

/// Flush `coalescer`'s per-operation latest progress values (bounded to at
/// most once per its own configured interval) and forward `health`'s
/// cadence-gated snapshot, both immediately -- neither is queued again
/// downstream of this call.
pub(super) fn flush_tick(
    coalescer: &ProgressCoalescer,
    health: &EngineHealth,
    sink: &dyn PresentationSink,
    now: Instant,
) {
    for (_scope, fact) in coalescer.flush_at(now) {
        sink.emit(fact_projection_event(&fact));
    }
    if let Some(snapshot) = health.publish_at(now) {
        sink.emit(health_projection_event(snapshot));
    }
}

/// Subscribe to `engine` and run the presentation projection loop until its
/// subscriber registry closes or this subscription is disconnected for
/// lagging too far behind. A lagged receiver records the disconnect on
/// engine health and returns -- presentation loss degrades observability
/// only, it never blocks or fails primary work.
pub async fn run_presentation_subscriber(
    engine: Arc<RuntimeEventEngine>,
    sink: Arc<dyn PresentationSink>,
) -> Result<(), SubscribeError> {
    let mut subscription = engine.subscribers().subscribe()?;
    let coalescer = ProgressCoalescer::new();
    let mut tick = tokio::time::interval(TUI_RENDER_TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            received = subscription.recv() => {
                match received {
                    Ok(frame) => route_fact(&coalescer, sink.as_ref(), &frame),
                    Err(RecvError::Lagged(_)) => {
                        subscription.record_disconnect(engine.health());
                        return Ok(());
                    }
                    Err(RecvError::Closed) => return Ok(()),
                }
            }
            _ = tick.tick() => flush_tick(&coalescer, engine.health(), sink.as_ref(), Instant::now()),
        }
    }
}
