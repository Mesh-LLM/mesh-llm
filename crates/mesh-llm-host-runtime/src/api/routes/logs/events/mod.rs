//! Semantic protocol for the dedicated logs SSE endpoint.
//!
//! The semantic protocol stays in its own modules; `stream` is the narrow raw
//! management adapter that validates its request before taking `TcpStream`
//! ownership. Durable REST queries remain the recovery authority when the
//! bounded replay window has an eviction gap.

mod protocol;
mod query;
mod session;
mod stream;

#[allow(unused_imports)]
pub(super) use protocol::heartbeat_frame;
pub(super) use query::parse_subscription;
pub(super) use stream::stream;
