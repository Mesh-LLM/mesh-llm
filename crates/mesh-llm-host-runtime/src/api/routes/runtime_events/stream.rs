//! The live socket loop: subscribe, headers, connection-shape recovery
//! frames, then keepalive/health/event fan-out until the client disconnects
//! or its lag bound is exceeded.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::broadcast::error::RecvError;

use crate::api::http::respond_error;
use crate::api::management_lifecycle::record_response_status;
use crate::runtime_events::config::KEEPALIVE_INTERVAL;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::health::HealthDeliveryGate;
use crate::runtime_events::subscribers::{SubscribeError, SubscriptionHandle, lag_bound_exceeded};

use super::cursor::Cursor;
use super::frames::{self, KEEPALIVE_FRAME};
use super::recovery::ConnectionShape;

const WRITE_TIMEOUT: Duration = Duration::from_millis(250);
const SSE_HEADERS: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";

pub(super) async fn run(
    stream: &mut TcpStream,
    engine: &Arc<RuntimeEventEngine>,
    shape: ConnectionShape,
) -> anyhow::Result<()> {
    // Subscribe BEFORE any header byte reaches the socket: a frame the
    // reducer publishes between "we decided to serve this connection" and
    // "we started reading from a subscription" must never be lost between
    // the client observing 200 OK and this task actually listening.
    let mut subscription = match engine.subscribers().subscribe() {
        Ok(subscription) => subscription,
        Err(SubscribeError::CapacityReached) => {
            return respond_error(stream, 503, "maximum runtime event subscribers reached").await;
        }
    };

    stream.write_all(SSE_HEADERS).await?;
    record_response_status(200);

    if write_initial_frames(stream, engine, &shape).await {
        // Task 8 (`.omo/plans/event-system-fixes.md`, defect D9): this
        // connection's own independent health-delivery gate, seeded from
        // the version `write_initial_frames` already sent above so the
        // live loop's first eligible check does not immediately re-deliver
        // the same snapshot (see `HealthDeliveryGate::seeded`).
        let health_gate =
            HealthDeliveryGate::seeded(engine.health().snapshot().version, Instant::now());
        live_loop(stream, engine, &mut subscription, health_gate).await;
    }
    Ok(())
}

fn current_cursor(engine: &RuntimeEventEngine) -> Cursor {
    Cursor::new(
        engine.process_instance(),
        engine.highest_known_sequence().unwrap_or(0),
    )
}

/// Emit the frozen frame order for `shape`. Returns `false` the moment a
/// write fails so the caller skips the live loop entirely.
async fn write_initial_frames(
    stream: &mut TcpStream,
    engine: &Arc<RuntimeEventEngine>,
    shape: &ConnectionShape,
) -> bool {
    match shape {
        ConnectionShape::NoCursor => {
            let cursor = current_cursor(engine);
            write_frame(stream, frames::state_frame(engine, cursor)).await
                && write_frame(stream, frames::health_frame(engine, cursor)).await
        }
        ConnectionShape::InWindow { frames: replay } => {
            for frame in replay {
                if !write_frame(stream, frames::event_frame(engine, frame)).await {
                    return false;
                }
            }
            let cursor = current_cursor(engine);
            write_frame(stream, frames::health_frame(engine, cursor)).await
        }
        ConnectionShape::Gap(gap) => {
            let cursor = current_cursor(engine);
            write_frame(stream, frames::replay_gap_frame(engine, gap)).await
                && write_frame(stream, frames::state_frame(engine, cursor)).await
                && write_frame(stream, frames::health_frame(engine, cursor)).await
        }
    }
}

/// Task 8 (`.omo/plans/event-system-fixes.md`, defect D9): `health_gate` is
/// checked on every keepalive tick AND after every delivered event frame —
/// not gated by the removed engine-global `publish_at` cadence, which let a
/// busy OTHER subscriber's own tick consume the one shared publish window
/// and starve this connection for up to ~50 minutes under load. This
/// connection owns its gate independently of every other connection.
async fn live_loop(
    stream: &mut TcpStream,
    engine: &Arc<RuntimeEventEngine>,
    subscription: &mut SubscriptionHandle,
    mut health_gate: HealthDeliveryGate,
) {
    let mut keepalive = tokio::time::interval(KEEPALIVE_INTERVAL);
    keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    keepalive.tick().await;

    loop {
        tokio::select! {
            _ = keepalive.tick() => {
                if !write_frame(stream, KEEPALIVE_FRAME.to_string()).await {
                    break;
                }
                if !maybe_write_health(stream, engine, &mut health_gate, current_cursor(engine)).await {
                    break;
                }
            }
            received = subscription.recv() => match received {
                Ok(frame) => {
                    // Frame-count lag is enforced by the broadcast channel
                    // itself (surfaces as `Lagged` below); age and bytes
                    // have no channel-native equivalent, so check them
                    // explicitly against every received frame — first
                    // limit wins, same disconnect + health-bump contract.
                    if lag_bound_exceeded(&frame, subscription.backlog_len(), Instant::now()) {
                        subscription.record_disconnect(engine.health());
                        break;
                    }
                    if !write_frame(stream, frames::event_frame(engine, &frame)).await {
                        break;
                    }
                    let cursor = Cursor::new(engine.process_instance(), frame.sequence.get());
                    if !maybe_write_health(stream, engine, &mut health_gate, cursor).await {
                        break;
                    }
                }
                Err(RecvError::Lagged(_)) => {
                    subscription.record_disconnect(engine.health());
                    break;
                }
                Err(RecvError::Closed) => break,
            },
        }
    }
}

/// Deliver a `runtime_health` frame only when `health_gate` says this
/// connection's last-delivered version is stale. Returns `false` only on a
/// write failure, matching every other frame writer in this loop so the
/// caller's `break` handling stays uniform; a gate-suppressed check (nothing
/// to deliver) returns `true`.
async fn maybe_write_health(
    stream: &mut TcpStream,
    engine: &Arc<RuntimeEventEngine>,
    health_gate: &mut HealthDeliveryGate,
    cursor: Cursor,
) -> bool {
    let snapshot = engine.health().snapshot();
    if !health_gate.should_deliver(&snapshot, Instant::now()) {
        return true;
    }
    write_frame(stream, frames::health_frame(engine, cursor)).await
}

async fn write_frame(stream: &mut TcpStream, frame: String) -> bool {
    tokio::time::timeout(WRITE_TIMEOUT, stream.write_all(frame.as_bytes()))
        .await
        .is_ok_and(|result| result.is_ok())
}

// Task 8-fix E5 (`.omo/plans/event-system-fixes.md`): `live_loop`'s recv
// arm calls `maybe_write_health` after every delivered event frame
// (`Ok(frame) =>` above) so a busy v1 subscriber's health delivery is
// never starved behind the 15s keepalive tick alone -- but
// `live_loop`/`maybe_write_health` had zero test callers anywhere in the
// repo before this. `live_loop` takes a real `&mut TcpStream`, so this
// drives it over a genuine loopback TCP pair rather than a mock writer.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    use mesh_llm_runtime_event_contracts::{
        FamilyFact, NativeRuntimeEventKind, OperationId, OperationScope, RuntimeEventIngress,
        RuntimeFact, SubmitOutcome,
    };

    use super::*;

    /// A real loopback TCP pair: `server` is what `live_loop` writes SSE
    /// frames into, exactly like the socket `run` hands it; `client` is
    /// read from directly by the test, exactly like a real subscriber.
    async fn loopback_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback listener");
        let addr = listener.local_addr().expect("listener local addr");
        let client = TcpStream::connect(addr)
            .await
            .expect("connect loopback client");
        let (server, _) = listener.accept().await.expect("accept loopback server");
        (server, client)
    }

    fn distinct_state_transition_fact() -> RuntimeFact {
        RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
    }

    /// Fails if `maybe_write_health`'s call in `live_loop`'s recv arm is
    /// removed (mutation M3, `.omo/evidence/event-system-fixes/task-08/mutation-proof.txt`).
    /// A fresh `HealthDeliveryGate` (mirroring `HealthDeliveryGate::new`'s
    /// own documented "never delivered yet" contract, exercised by
    /// `health_delivery_gate_delivers_on_the_first_check_from_new` in
    /// `health.rs`) delivers unconditionally on its first eligible check,
    /// so the FIRST time the recv arm's `maybe_write_health` runs -- right
    /// after this test's one submitted event frame -- it must write a
    /// `runtime_health` frame. The 15s `KEEPALIVE_INTERVAL` never fires
    /// within this test's real-time read-retry budget, so a
    /// `runtime_health` frame on the wire is attributable to the recv arm
    /// alone, never the keepalive branch.
    #[tokio::test]
    async fn live_loop_delivers_health_right_after_an_event_frame_not_only_on_keepalive() {
        let engine = RuntimeEventEngine::new();
        let mut subscription = engine.subscribers().subscribe().expect("subscribe");
        let (mut server, mut client) = loopback_pair().await;
        let health_gate = HealthDeliveryGate::new();

        let drive_engine = Arc::clone(&engine);
        let loop_task = tokio::spawn(async move {
            live_loop(&mut server, &drive_engine, &mut subscription, health_gate).await;
        });

        let scope = OperationScope::root_only(OperationId::new());
        let outcome = engine
            .unreserved_ingress(scope)
            .try_submit(distinct_state_transition_fact());
        assert_eq!(outcome, SubmitOutcome::Accepted);
        engine.drain();

        let mut received = String::new();
        let mut buf = [0u8; 4096];
        for _ in 0..100 {
            match tokio::time::timeout(Duration::from_millis(5), client.read(&mut buf)).await {
                Ok(Ok(n)) if n > 0 => {
                    received.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if received.contains("event: runtime_event")
                        && received.contains("event: runtime_health")
                    {
                        break;
                    }
                }
                _ => {}
            }
        }

        loop_task.abort();

        assert!(
            received.contains("event: runtime_event"),
            "the submitted fact must reach the client as a runtime_event frame; got: {received:?}"
        );
        assert!(
            received.contains("event: runtime_health"),
            "the per-frame maybe_write_health call must deliver a health frame right \
             after the event frame, not only on the 15s keepalive tick (never reached \
             within this test's real-time read budget); got: {received:?}"
        );
    }
}
