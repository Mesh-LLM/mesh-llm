use anyhow::{Context, Result};
use skippy_protocol::binary::{StageWireMessage, read_stage_message};
use std::io;
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};

static BINARY_SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_connection_session_id() -> u64 {
    BINARY_SESSION_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub(super) fn receive_next_message(
    upstream: &mut TcpStream,
    activation_width: i32,
    first_message: Option<StageWireMessage>,
    pending_prefill_replies: usize,
    observed_message_count: usize,
) -> Result<Option<StageWireMessage>> {
    if first_message.is_some() {
        return Ok(first_message);
    }
    match read_stage_message(upstream, activation_width) {
        Ok(message) => Ok(Some(message)),
        Err(error)
            if error.kind() == io::ErrorKind::UnexpectedEof
                && pending_prefill_replies == 0
                && observed_message_count == 0 =>
        {
            Ok(None)
        }
        Err(error) => Err(error).context("read binary stage message"),
    }
}
