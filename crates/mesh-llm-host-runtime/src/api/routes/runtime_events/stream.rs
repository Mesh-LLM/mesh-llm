//! The live socket loop: subscribe, headers, connection-shape recovery
//! frames, then keepalive/health/event fan-out until the client disconnects
//! or its lag bound is exceeded.

use std::sync::Arc;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::broadcast::error::RecvError;

use crate::api::http::respond_error;
use crate::api::management_lifecycle::record_response_status;
use crate::runtime_events::config::KEEPALIVE_INTERVAL;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::subscribers::{SubscribeError, SubscriptionHandle};

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
        live_loop(stream, engine, &mut subscription).await;
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

async fn live_loop(
    stream: &mut TcpStream,
    engine: &Arc<RuntimeEventEngine>,
    subscription: &mut SubscriptionHandle,
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
            }
            received = subscription.recv() => match received {
                Ok(frame) => {
                    if !write_frame(stream, frames::event_frame(engine, &frame)).await {
                        break;
                    }
                    if engine.health().publish_at(std::time::Instant::now()).is_some() {
                        let cursor = Cursor::new(engine.process_instance(), frame.sequence.get());
                        if !write_frame(stream, frames::health_frame(engine, cursor)).await {
                            break;
                        }
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

async fn write_frame(stream: &mut TcpStream, frame: String) -> bool {
    tokio::time::timeout(WRITE_TIMEOUT, stream.write_all(frame.as_bytes()))
        .await
        .is_ok_and(|result| result.is_ok())
}
