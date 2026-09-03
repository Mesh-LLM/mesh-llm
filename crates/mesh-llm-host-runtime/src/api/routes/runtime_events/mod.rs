//! `GET /api/runtime/events/v1` — restart-aware runtime-event SSE stream.
//!
//! Trusted-local, read-only (classified in `api::access`). This module owns
//! wire encoding, cursor transport, and connection-shape recovery ordering
//! only; it never rebuilds replay or reducer logic — those stay owned by
//! `crate::runtime_events` (task 3/4's host engine).

mod cursor;
mod frames;
mod reconnect;
mod recovery;
mod state_projection;
mod stream;

#[cfg(test)]
mod runtime_event_api_tests;

use tokio::net::TcpStream;

use super::super::MeshApi;
use super::super::http::respond_error;
use cursor::CursorError;

pub(super) async fn handle(
    stream: &mut TcpStream,
    _state: &MeshApi,
    path: &str,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    let Some(engine) = crate::runtime_events::runtime_event_engine() else {
        return respond_error(stream, 503, "runtime event engine is not running").await;
    };

    let peer_key = stream.peer_addr().ok().map(|addr| addr.ip());
    if !reconnect::record_attempt(peer_key) {
        return respond_error(stream, 429, "runtime event reconnect rate limit exceeded").await;
    }

    let requested_cursor = match cursor::resolve(path, raw_request) {
        Ok(cursor) => cursor,
        Err(CursorError::Malformed) => {
            return respond_error(stream, 400, "malformed runtime event cursor").await;
        }
    };

    let shape = match recovery::classify(&engine, requested_cursor) {
        Ok(shape) => shape,
        Err(CursorError::Malformed) => {
            return respond_error(stream, 400, "runtime event cursor is out of range").await;
        }
    };

    stream::run(stream, &engine, shape).await
}
