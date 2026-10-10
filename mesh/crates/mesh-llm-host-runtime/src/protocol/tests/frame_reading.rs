use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncRead, ReadBuf};

use super::*;

/// Sends a length prefix and then `body` bytes, then stalls like a peer that
/// stopped sending. Records the largest buffer the reader asked it to fill.
struct StallingPeer {
    prefix: [u8; 4],
    sent_prefix: usize,
    body: usize,
    largest_read: usize,
}

impl StallingPeer {
    fn new(claimed_len: u32, body: usize) -> Self {
        Self {
            prefix: claimed_len.to_le_bytes(),
            sent_prefix: 0,
            body,
            largest_read: 0,
        }
    }
}

impl AsyncRead for StallingPeer {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        if self.sent_prefix < self.prefix.len() {
            let start = self.sent_prefix;
            let count = buf.remaining().min(self.prefix.len() - start);
            buf.put_slice(&self.prefix[start..start + count]);
            self.sent_prefix += count;
            return Poll::Ready(Ok(()));
        }
        self.largest_read = self.largest_read.max(buf.remaining());
        if self.body == 0 {
            // Never woken: the peer has gone quiet.
            return Poll::Pending;
        }
        let count = buf.remaining().min(self.body);
        buf.put_slice(&vec![0; count]);
        self.body -= count;
        Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn frame_buffer_grows_with_the_bytes_that_arrive() {
    // A peer claims the largest frame, sends a little of it, and stalls.
    let mut peer = StallingPeer::new(MAX_CONTROL_FRAME_BYTES as u32, 100);
    let read = tokio::time::timeout(
        Duration::from_millis(50),
        read_frame(&mut peer, MAX_CONTROL_FRAME_BYTES),
    )
    .await;
    assert!(read.is_err(), "the frame is incomplete, so the read waits");
    assert!(
        peer.largest_read <= 64 * 1024,
        "the reader allocated {} bytes for a frame that sent 100",
        peer.largest_read
    );
}

#[tokio::test]
async fn frames_read_completely_and_reject_early_ends() {
    let mut whole = b"\x03\0\0\0abc".as_slice();
    assert_eq!(read_frame(&mut whole, 16).await.unwrap(), b"abc");

    let mut truncated = b"\x05\0\0\0abc".as_slice();
    assert!(read_frame(&mut truncated, 16).await.is_err());

    let mut oversized = b"\x11\0\0\0".as_slice();
    assert!(read_frame(&mut oversized, 16).await.is_err());
}
