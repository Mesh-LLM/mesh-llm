//! The `openai.exchange.v1` events the paid serving path publishes, as the
//! free serving path does: an effective event when serving begins and a
//! terminal event with what was delivered, replayed through the relay's own
//! adapter so both sides compute the same usage and digests.

use crate::mesh::Node;
use crate::network::openai::paid_exchange::PaidServedExchange;
use crate::network::payments::request::PaidRequest;

/// Publish the terminal exchange event, when an effective one was published.
pub(super) async fn end_exchange(
    node: &Node,
    exchange: Option<(PaidServedExchange, DeliveredCapture)>,
    adapter: crate::network::openai::transport::ResponseAdapter,
    served: bool,
    model: &str,
    request_digest: Option<&str>,
    payer: iroh::EndpointId,
) {
    if let Some((exchange, delivered)) = exchange {
        let outcome = exchange_outcome(served, delivered, adapter).await;
        exchange
            .finish(node, model, &outcome, request_digest, payer)
            .await;
    }
}

/// The terminal exchange event's outcome: what was delivered when serving
/// completed, `Failed` when generation or closing serving failed.
async fn exchange_outcome(
    served: bool,
    delivered: DeliveredCapture,
    adapter: crate::network::openai::transport::ResponseAdapter,
) -> crate::network::openai::transport::RouteDispatchOutcome {
    if served {
        delivered_outcome(delivered, adapter).await
    } else {
        crate::network::openai::transport::RouteDispatchOutcome::Failed(
            "paid serving failed before the response was delivered",
        )
    }
}

/// What was delivered, for the terminal exchange event: the relay's own
/// usage and digests over the delivered bytes. Past the capture limit, only
/// the response status, read off the kept head of the response.
async fn delivered_outcome(
    delivered: DeliveredCapture,
    adapter: crate::network::openai::transport::ResponseAdapter,
) -> crate::network::openai::transport::RouteDispatchOutcome {
    use crate::network::openai::transport::RouteDispatchOutcome;
    let status = delivered.status();
    match delivered.into_bytes() {
        Some(raw) => crate::network::openai::served_outcome_of_raw_response(&raw, adapter).await,
        None => match status {
            Some(status) => RouteDispatchOutcome::Responded(status),
            None => RouteDispatchOutcome::Failed("the served response had no HTTP status line"),
        },
    }
}

/// The adapter the payer's relay applies to this response, so the seller's
/// replay computes the same usage and digests. Mirrors the payer's default
/// for a chat request (`request_parse.rs`): a paid request arrives already
/// normalized to the backend's chat shape.
pub(super) fn paid_response_adapter(
    request: &PaidRequest,
) -> crate::network::openai::transport::ResponseAdapter {
    use crate::network::openai::transport::ResponseAdapter;
    if request.path.split('?').next() != Some("/v1/chat/completions") {
        return ResponseAdapter::None;
    }
    if request
        .body
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        ResponseAdapter::OpenAiChatCompletionsStream
    } else {
        ResponseAdapter::OpenAiChatCompletionsJson
    }
}

/// The bytes delivered to the payer, kept to replay through the relay for
/// the exchange event's digests. Past [`MAX_CAPTURED_RESPONSE_BYTES`] the
/// capture is dropped and the event carries no digests, never partial ones.
#[derive(Default)]
pub(super) struct DeliveredCapture {
    bytes: Vec<u8>,
    overflowed: bool,
    /// The first bytes delivered, kept past the capture limit so the
    /// response status is still known.
    head: Vec<u8>,
}

/// Enough of the response to hold its status line (`HTTP/1.1 200 `).
const RESPONSE_HEAD_BYTES: usize = 64;

const MAX_CAPTURED_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

impl DeliveredCapture {
    pub(super) fn push(&mut self, bytes: &[u8]) {
        let room = RESPONSE_HEAD_BYTES.saturating_sub(self.head.len());
        self.head.extend_from_slice(&bytes[..room.min(bytes.len())]);
        if self.overflowed || self.bytes.len() + bytes.len() > MAX_CAPTURED_RESPONSE_BYTES {
            self.overflowed = true;
            self.bytes = Vec::new();
            return;
        }
        self.bytes.extend_from_slice(bytes);
    }

    fn into_bytes(self) -> Option<Vec<u8>> {
        (!self.overflowed).then_some(self.bytes)
    }

    /// The HTTP status of the delivered response, from its status line.
    fn status(&self) -> Option<u16> {
        let line = self.head.strip_prefix(b"HTTP/1.")?;
        let code = line.get(2..5)?;
        std::str::from_utf8(code).ok()?.parse().ok()
    }
}

#[cfg(test)]
mod delivered_capture_tests {
    use super::*;
    use crate::network::openai::transport::RouteDispatchOutcome;

    const OK: &[u8] =
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";

    #[test]
    fn replay_uses_the_normalized_paid_request_adapter() {
        use crate::network::openai::transport::ResponseAdapter;
        for (path, stream, expected) in [
            (
                "/v1/chat/completions",
                true,
                ResponseAdapter::OpenAiChatCompletionsStream,
            ),
            (
                "/v1/chat/completions",
                false,
                ResponseAdapter::OpenAiChatCompletionsJson,
            ),
            ("/v1/completions", true, ResponseAdapter::None),
            ("/v1/completions", false, ResponseAdapter::None),
        ] {
            let raw =
                format!("POST {path} HTTP/1.1\r\n\r\n{{\"model\":\"test\",\"stream\":{stream}}}");
            let request = PaidRequest::parse(raw.as_bytes()).unwrap();
            assert_eq!(paid_response_adapter(&request), expected);
        }
    }

    #[test]
    fn the_status_is_read_off_the_first_bytes() {
        let mut capture = DeliveredCapture::default();
        capture.push(&OK[..5]);
        capture.push(&OK[5..]);
        assert_eq!(capture.status(), Some(200));
        assert_eq!(DeliveredCapture::default().status(), None);
        let mut not_http = DeliveredCapture::default();
        not_http.push(b"garbage");
        assert_eq!(not_http.status(), None);
    }

    #[tokio::test]
    async fn past_the_capture_limit_the_status_is_kept_and_only_the_digests_dropped() {
        let mut capture = DeliveredCapture::default();
        capture.push(OK);
        capture.push(&vec![b'x'; MAX_CAPTURED_RESPONSE_BYTES]);
        assert!(matches!(
            exchange_outcome(
                true,
                capture,
                crate::network::openai::transport::ResponseAdapter::None
            )
            .await,
            RouteDispatchOutcome::Responded(200)
        ));
    }

    #[tokio::test]
    async fn a_failed_serving_still_ends_the_exchange_with_a_terminal_failure() {
        let mut capture = DeliveredCapture::default();
        capture.push(OK);
        assert!(matches!(
            exchange_outcome(
                false,
                capture,
                crate::network::openai::transport::ResponseAdapter::None
            )
            .await,
            RouteDispatchOutcome::Failed(_)
        ));
    }
}
