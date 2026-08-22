//! Propagating a downstream client's disconnect back to the upstream.
//!
//! When the client relaying through us goes away mid-response, whatever is
//! generating that response has no idea and keeps working. Local and remote
//! attempts each reach their upstream over a different protocol, so each needs
//! its own way of saying "stop" — but the *decision* to say it is the same for
//! both, and keeping it in one place is what stops the two arms from drifting.
//!
//! # What this does and does not change
//!
//! Both transports already say something on drop. Dropping a `TcpStream`
//! closes the socket and sends FIN; dropping a `noq::RecvStream` that was not
//! read to completion sends STOP_SENDING with error code 0. Every upstream
//! here is a local of the attempt that opened it, so that drop lands a few
//! instructions after the cancel, with no await in between — the same frame,
//! the same code.
//!
//! So this module does not change what goes on the wire. It changes where the
//! decision lives: one place instead of two hand-rolled teardowns, stated at
//! the point the outcome is known rather than inferred from drop order, and
//! not resting on a drop-time detail of whichever QUIC crate we are on.
//!
//! Which means it is *not* an explanation for RV-DEFECT-003 — a worker that
//! logged `request_stream_completed` 22ms after the gateway logged
//! `request_dropped`. A STOP_SENDING was already being sent then. If that
//! signal is arriving and generation continues anyway, the stall is downstream
//! of here, in the worker's relay teardown (`network/tunnel.rs`): the peer's
//! `relay_tcp_to_quic` write has to fail, and `finish_relay_pair` then waits
//! on the *other* direction before the local TCP stream to the API proxy is
//! dropped. That defect stays open pending a worker-side check.

use super::common::RouteAttemptResult;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

/// An upstream that can be told to stop producing a response we no longer want.
pub(in crate::network::openai::response) trait CancelUpstream {
    /// Abandon this upstream. Best effort: the upstream may already be gone,
    /// and there is nothing useful to do about it if the signal does not land.
    async fn cancel(&mut self);
}

impl CancelUpstream for TcpStream {
    async fn cancel(&mut self) {
        let _ = self.shutdown().await;
    }
}

/// QUIC application error code for "the client this response was for is gone".
///
/// The peer only needs to know it should stop; it does not branch on the code,
/// and `mesh::connections` already uses 0 when it abandons an inbound stream.
/// It is also the code `RecvStream::drop` would have used, which is what keeps
/// the explicit cancel and the drop indistinguishable to the peer.
const CLIENT_DISCONNECTED: u32 = 0;

impl CancelUpstream for iroh::endpoint::RecvStream {
    async fn cancel(&mut self) {
        // STOP_SENDING, so the peer's next write on its half fails. `stop()`
        // marks the stream fully read, so the drop that follows is a no-op
        // rather than a second frame.
        let _ = self.stop(CLIENT_DISCONNECTED.into());
    }
}

/// Pass `result` through, cancelling `upstream` first if the client left.
///
/// Only `ClientDisconnected` cancels. Every other outcome either finished
/// normally or is retryable against another target, and a retryable attempt's
/// upstream is dropped rather than cancelled -- which, per the module docs, is
/// a difference in intent rather than in what the peer receives.
pub(in crate::network::openai::response) async fn cancel_upstream_if_client_disconnected<
    U: CancelUpstream + ?Sized,
>(
    result: RouteAttemptResult,
    upstream: &mut U,
) -> RouteAttemptResult {
    if matches!(result, RouteAttemptResult::ClientDisconnected) {
        upstream.cancel().await;
    }
    result
}

/// These cover the routing decision only: which outcomes reach `cancel()` and
/// which do not. Whether a peer actually stops generating on receipt is a
/// property of the peer, not of this dispatch, and no unit test here can
/// observe it -- see the module docs on RV-DEFECT-003.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::openai::response_quality::ResponseQualityFailure;

    #[derive(Default)]
    struct CancelRecorder {
        cancels: usize,
    }

    impl CancelUpstream for CancelRecorder {
        async fn cancel(&mut self) {
            self.cancels += 1;
        }
    }

    #[tokio::test]
    async fn a_disconnected_client_cancels_the_upstream() {
        let mut upstream = CancelRecorder::default();

        let result = cancel_upstream_if_client_disconnected(
            RouteAttemptResult::ClientDisconnected,
            &mut upstream,
        )
        .await;

        assert_eq!(upstream.cancels, 1);
        assert_eq!(result, RouteAttemptResult::ClientDisconnected);
    }

    #[tokio::test]
    async fn every_other_outcome_leaves_the_upstream_alone() {
        for result in [
            RouteAttemptResult::Delivered {
                status_code: 200,
                usage: None,
            },
            RouteAttemptResult::RetryableTimeout,
            RouteAttemptResult::RetryableUnavailable,
            RouteAttemptResult::RetryableContextOverflow,
            RouteAttemptResult::RetryableResponseQuality(
                ResponseQualityFailure::EmptyAssistantOutput,
            ),
            RouteAttemptResult::CommittedStreamFailure { status_code: 200 },
        ] {
            let mut upstream = CancelRecorder::default();

            let passed_through =
                cancel_upstream_if_client_disconnected(result, &mut upstream).await;

            assert_eq!(upstream.cancels, 0, "{result:?} should not cancel");
            assert_eq!(passed_through, result);
        }
    }
}
