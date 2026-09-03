//! Wires the progress coalescer and privacy-safe projection to the host
//! runtime-event engine's subscriber registry, producing at most one
//! progress `OutputEvent` per operation per render tick plus every terminal
//! and coalesced-health event, all BEFORE they ever reach
//! `mesh_llm_events::emit_event` -- the TUI's own unbounded
//! `OutputCommand::Event` channel (`mesh-llm-tui`) is never touched by this
//! module and never coalesces anything itself.
//!
//! **Spawned from `runtime/run_auto.rs`**, immediately after
//! `install_runtime_event_engine(...)`, via [`spawn_presentation_subscriber`]
//! -- the full `mesh-llm serve --auto` / TUI-visible path. Deliberately NOT
//! spawned from `runtime/local_model_only.rs`, which keeps its documented,
//! tested "zero management subscribers" invariant (nothing there calls
//! `.subscribers()`).
//!
//! [`spawn_presentation_subscriber`] is split into a synchronous [`attach`]
//! step and an async [`drive_presentation_subscriber`] loop specifically so
//! attachment is observable by the caller with no scheduler race: the
//! subscription is registered on the engine BEFORE the function returns,
//! not at some later point inside a freshly spawned task.
//!
//! The drive loop also calls [`RuntimeEventEngine::drain`] once per render
//! tick. This is deliberate and load-bearing, not incidental: nothing else
//! in the running host calls `drain()` outside test code (verified by grep
//! across `crates/mesh-llm-host-runtime/src` at the time this was written)
//! -- a submitted terminal fact sits in the engine's wake list, reserved
//! but never applied through the reducer or published to any subscriber,
//! until something drains it. This subscriber is the one always-on,
//! life-of-process consumer the plan's own "Authority boundaries" table
//! describes for presentation, so its own tick is the natural place to pump
//! the engine until a dedicated engine-owned drain loop exists. Draining is
//! idempotent and safe to call from multiple sites (task 3's `drain()` is a
//! plain `&self` method over a lock-protected wake list), so this does not
//! preclude another consumer also draining later.

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
use crate::runtime_events::subscribers::{SubscribeError, SubscriptionHandle};

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

/// Subscribe to `engine` synchronously. Split out from the async drive loop
/// so a caller (`spawn_presentation_subscriber`) can prove the subscription
/// is registered before returning, with no race against the spawned task's
/// own scheduling.
pub fn attach(engine: &RuntimeEventEngine) -> Result<SubscriptionHandle, SubscribeError> {
    engine.subscribers().subscribe()
}

/// Drive the presentation projection loop for an already-`attach`ed
/// `subscription` until the engine's subscriber registry closes or this
/// subscription is disconnected for lagging too far behind. A lagged
/// receiver records the disconnect on engine health and returns --
/// presentation loss degrades observability only, it never blocks or fails
/// primary work. See the module doc for why every tick also calls
/// `engine.drain()`.
pub async fn drive_presentation_subscriber(
    mut subscription: SubscriptionHandle,
    engine: Arc<RuntimeEventEngine>,
    sink: Arc<dyn PresentationSink>,
) {
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
                        return;
                    }
                    Err(RecvError::Closed) => return,
                }
            }
            _ = tick.tick() => {
                engine.drain();
                flush_tick(&coalescer, engine.health(), sink.as_ref(), Instant::now());
            }
        }
    }
}

/// Attach to `engine` synchronously and spawn [`drive_presentation_subscriber`]
/// as a background task using the real [`EmitEventSink`]. This is the
/// `run_auto.rs` wiring point; do NOT call it from `local_model_only.rs`
/// (see the module doc). Returns `Err` only when the engine is already at
/// its concurrent-subscriber cap.
pub fn spawn_presentation_subscriber(
    engine: &Arc<RuntimeEventEngine>,
) -> Result<tokio::task::JoinHandle<()>, SubscribeError> {
    let subscription = attach(engine)?;
    let engine = Arc::clone(engine);
    Ok(tokio::spawn(async move {
        drive_presentation_subscriber(subscription, engine, Arc::new(EmitEventSink)).await;
    }))
}
