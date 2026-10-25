//! Limits on the mesh streams a node serves before a peer is admitted.
//!
//! Gossip streams, and route requests under the open trust policies, are
//! accepted from any peer that connects, because gossip is how a peer gets
//! admitted. Each one spawns a handler that reads a frame of up to 8 MiB, and
//! the transport lets one connection open 1024 streams. A peer that opened
//! every stream and stalled each frame could hold that many handlers and
//! their buffers indefinitely.
//!
//! Each connection may run only a few of these handlers at once, and each
//! frame has to arrive within a deadline. Admitted peers use the same
//! handlers, so the limits apply to them too; ordinary gossip needs one or
//! two at a time.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::protocol::{MAX_CONTROL_FRAME_BYTES, STREAM_GOSSIP, STREAM_ROUTE_REQUEST, read_frame};

/// Gossip and route-request handlers one connection may run at once.
pub(crate) const MAX_PRE_ADMISSION_STREAMS_PER_CONNECTION: usize = 8;
/// How long a peer has to send a gossip or route-request frame.
pub(crate) const PRE_ADMISSION_FRAME_TIMEOUT: Duration = Duration::from_secs(30);

/// The handler slots of one connection.
pub(crate) struct PreAdmissionSlots(Arc<Semaphore>);

/// Whether a stream may start its handler.
pub(crate) enum SlotClaim {
    /// This stream type is not limited.
    Unlimited,
    /// The handler holds the slot until it finishes.
    Granted(OwnedSemaphorePermit),
    /// Every slot is taken; the stream should be refused.
    Full,
}

impl PreAdmissionSlots {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Semaphore::new(
            MAX_PRE_ADMISSION_STREAMS_PER_CONNECTION,
        )))
    }

    pub(crate) fn claim(&self, stream_type: u8) -> SlotClaim {
        if !matches!(stream_type, STREAM_GOSSIP | STREAM_ROUTE_REQUEST) {
            return SlotClaim::Unlimited;
        }
        match Arc::clone(&self.0).try_acquire_owned() {
            Ok(permit) => SlotClaim::Granted(permit),
            Err(_) => SlotClaim::Full,
        }
    }
}

/// Reads the frame a gossip or route-request stream opens with, giving up
/// once the deadline passes.
pub(crate) async fn read_pre_admission_frame<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<Vec<u8>> {
    tokio::time::timeout(
        PRE_ADMISSION_FRAME_TIMEOUT,
        read_frame(reader, MAX_CONTROL_FRAME_BYTES),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "frame did not arrive within {} s",
            PRE_ADMISSION_FRAME_TIMEOUT.as_secs()
        )
    })?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::STREAM_TUNNEL_HTTP;

    #[test]
    fn a_connection_runs_a_bounded_number_of_gossip_and_route_handlers() {
        let slots = PreAdmissionSlots::new();
        let mut held = Vec::new();
        for index in 0..MAX_PRE_ADMISSION_STREAMS_PER_CONNECTION {
            let stream_type = if index % 2 == 0 {
                STREAM_GOSSIP
            } else {
                STREAM_ROUTE_REQUEST
            };
            match slots.claim(stream_type) {
                SlotClaim::Granted(permit) => held.push(permit),
                _ => panic!("slot {index} should be free"),
            }
        }
        assert!(matches!(slots.claim(STREAM_GOSSIP), SlotClaim::Full));
        assert!(matches!(
            slots.claim(STREAM_TUNNEL_HTTP),
            SlotClaim::Unlimited
        ));

        held.pop();
        assert!(matches!(slots.claim(STREAM_GOSSIP), SlotClaim::Granted(_)));
    }

    #[tokio::test(start_paused = true)]
    async fn a_stalled_frame_is_abandoned_at_the_deadline() {
        // The peer sends a length prefix and then nothing.
        let (mut peer, mut node) = tokio::io::duplex(64);
        tokio::io::AsyncWriteExt::write_all(&mut peer, &1024u32.to_le_bytes())
            .await
            .unwrap();
        let error = read_pre_admission_frame(&mut node).await.unwrap_err();
        assert!(error.to_string().contains("did not arrive"), "{error}");
        drop(peer);
    }
}
