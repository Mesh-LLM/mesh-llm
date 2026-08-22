//! Propagating a downstream client's disconnect back to the upstream.
//!
//! When the client relaying through us goes away mid-response, whatever is
//! generating that response has no idea and keeps working. Local and remote
//! attempts each reach their upstream over a different protocol, so each needs
//! its own way of saying "stop" — but the *decision* to say it is the same for
//! both, and keeping it in one place is what stops the two arms from drifting.

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
const CLIENT_DISCONNECTED: u32 = 0;

impl CancelUpstream for iroh::endpoint::RecvStream {
    async fn cancel(&mut self) {
        // STOP_SENDING. The worker peer's next write on its half fails, which
        // is what actually ends its generation loop -- without this it runs to
        // completion producing tokens for a client that already hung up.
        let _ = self.stop(CLIENT_DISCONNECTED.into());
    }
}

/// Pass `result` through, cancelling `upstream` first if the client left.
///
/// Only `ClientDisconnected` cancels. Every other outcome either finished
/// normally or is retryable against another target, and a retryable attempt's
/// upstream is dropped rather than cancelled.
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
