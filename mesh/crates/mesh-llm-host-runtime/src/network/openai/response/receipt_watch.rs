//! The receipt audit (`network/receipt_audit`) of one attempt on a peer: the
//! request the peer is sent names this node's receipt key, and the peer's
//! response is collected for the audit.
//!
//! The audit reads the peer's own chat completion. When the client gets it
//! as it came, the bytes the client is written are read, so the audit
//! checks the text the client got. When this node translates it first
//! (`/v1/messages`, `/v1/responses`), the translation keeps no receipt, so
//! the bytes are read as they come from the peer instead.

use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, ReadBuf};

use super::cancellation::CancelUpstream;
use super::common::RouteAttemptResult;
use crate::logging::OpenAiRouteObserver;
use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::client_stream::response_wire_bytes::HttpResponseByteTap;
use crate::network::openai::request_normalize::ResponseAdapter;
use crate::network::openai::trust_policy;
use crate::network::receipt_audit;
use crate::network::request_trust::{RequestTrust, Unsatisfied};

/// The watch on one attempt; empty when this node does not audit the turn.
pub(super) struct ReceiptWatch(Option<receipt_audit::Watch>);

impl ReceiptWatch {
    /// Watches the turn `request` asks `host_id` for, when this node audits
    /// it. A local client's chat completion names this node's receipt key
    /// when this node audits its peers' receipts. Whether it does was
    /// decided with the request's trust policy, so a request that is not
    /// audited pays for no second read of the settings or the head here. An
    /// error for a request whose policy requires receipts and that this
    /// node cannot watch: it must not be forwarded.
    pub(super) async fn begin(
        node: &mesh::Node,
        trust: RequestTrust,
        client: &ClientStream,
        host_id: iroh::EndpointId,
        request: &[u8],
    ) -> Result<Self, String> {
        if !trust.audits_peers() || client.is_mesh_peer() {
            return Ok(Self(None));
        }
        let tier = trust.requires_receipts();
        receipt_audit::watch(node, host_id, request, tier)
            .await
            .map(Self)
    }

    /// The receipt id this node asks the peer to issue its receipt under,
    /// when the turn is watched.
    pub(super) fn receipt_id(&self) -> Option<&str> {
        self.0.as_ref().map(|watch| watch.receipt_id.as_str())
    }

    /// The request to forward: `request`, or the one that names this node's
    /// receipt key when the turn is watched.
    pub(super) fn request<'a>(&'a self, request: &'a [u8]) -> &'a [u8] {
        self.0
            .as_ref()
            .map_or(request, |watch| watch.request.as_slice())
    }

    /// Collects what `client` is written from here on for the audit, when
    /// the client gets the peer's answer as it came (`adapter`).
    pub(super) fn observe(&self, client: &mut ClientStream, adapter: ResponseAdapter) {
        if let Some(watch) = &self.0
            && adapter == ResponseAdapter::None
        {
            client.observe(watch.observer.clone());
        }
    }

    /// Stops collecting: the attempt is over.
    pub(super) fn stop_observing(&self, client: &mut ClientStream) {
        if let Some(watch) = &self.0 {
            client.stop_observing(&watch.observer);
        }
    }

    /// `upstream`, read for the audit too when this node translates the
    /// peer's answer before the client gets it (`adapter`).
    pub(super) fn upstream<'a, R>(
        &self,
        upstream: &'a mut R,
        adapter: ResponseAdapter,
    ) -> WatchedUpstream<'a, R> {
        let tap = self
            .0
            .as_ref()
            .filter(|_| adapter != ResponseAdapter::None)
            .map(|watch| Box::new(HttpResponseByteTap::new(watch.observer.clone())));
        WatchedUpstream {
            upstream,
            tap,
            at_eof: false,
        }
    }
}

/// Answers a request whose policy requires receipts and that this node
/// cannot watch: 409, and the peer is not asked. The attempt is over, and
/// no other peer may serve the request unwatched either.
pub(super) async fn refuse_unwatched(
    client: &mut ClientStream,
    host_id: iroh::EndpointId,
    why: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteAttemptResult {
    tracing::warn!(
        "receipt audit: a request that requires receipts was not sent to {}: {why}",
        host_id.fmt_short()
    );
    let reason = format!(
        "the request's trust policy requires receipts, and this node cannot audit this request: {why}"
    );
    let stream = std::mem::replace(client, ClientStream::Null);
    trust_policy::answer(stream, Unsatisfied::NoCandidate(reason), route_observer).await;
    RouteAttemptResult::Delivered {
        status_code: 409,
        usage: None,
        cache_cost: None,
        output_digests: Default::default(),
    }
}

/// A peer's response, read for the audit as the relay reads it.
pub(super) struct WatchedUpstream<'a, R> {
    upstream: &'a mut R,
    tap: Option<Box<HttpResponseByteTap>>,
    at_eof: bool,
}

impl<R> WatchedUpstream<'_, R> {
    /// The relay is done with the response: what it read is all the audit
    /// gets. A response the relay stopped reading before its end is no
    /// turn to audit.
    pub(super) fn finish(&mut self) {
        if let Some(mut tap) = self.tap.take() {
            tap.finish_read(self.at_eof);
        }
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for WatchedUpstream<'_, R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        let before = buf.filled().len();
        let result = Pin::new(&mut *this.upstream).poll_read(cx, buf);
        if let Poll::Ready(Ok(())) = &result {
            let read = &buf.filled()[before..];
            this.at_eof |= read.is_empty();
            if let Some(tap) = this.tap.as_mut() {
                tap.update(read);
            }
        }
        result
    }
}

impl<R: CancelUpstream> CancelUpstream for WatchedUpstream<'_, R> {
    async fn cancel(&mut self) {
        self.upstream.cancel().await;
    }
}

/// A watch that reports to `observer`, as one of a watched turn does.
#[cfg(test)]
impl ReceiptWatch {
    pub(super) fn observed_by(
        observer: std::sync::Arc<dyn skippy_inference_api::wire_bytes::WireBytesObserver>,
    ) -> Self {
        Self(Some(receipt_audit::Watch {
            request: Vec::new(),
            receipt_id: String::new(),
            observer,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use mesh_llm_events::logging::identifiers::RequestId;
    use tokio::io::AsyncWriteExt;

    use super::super::common::ResponseRetryPolicy;
    use super::super::dispatch::{RelayAttemptContext, relay_attempted_response};
    use super::super::probe::probe_http_response;
    use super::*;
    use crate::logging::OpenAiRouteObserver;
    use crate::network::openai::request_normalize::ResponseAdapter;
    use crate::network::receipt_audit::CollectorProbe;

    const RECEIPT: &str = r#"{"receipt_id":"r","version":7}"#;

    /// The peer's chat completion, whole, with its receipt.
    fn json_answer() -> Vec<u8> {
        let body = format!(
            r#"{{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{{"index":0,"message":{{"role":"assistant","content":"Paris."}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}},"receipt":{RECEIPT}}}"#
        );
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    /// The peer's chat completion as a chunked event stream, the receipt in
    /// its last event before `[DONE]`.
    fn stream_answer() -> Vec<u8> {
        let events = [
            r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Paris."},"finish_reason":null}]}"#.to_owned(),
            r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#.to_owned(),
            format!(r#"{{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"receipt":{RECEIPT}}}"#),
            "[DONE]".to_owned(),
        ];
        let mut wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for event in events {
            let frame = format!("data: {event}\n\n");
            wire.extend_from_slice(format!("{:x}\r\n{frame}\r\n", frame.len()).as_bytes());
        }
        wire.extend_from_slice(b"0\r\n\r\n");
        wire
    }

    /// Relays `answer` to a client as `adapter` shapes it, with the watch
    /// attached as an attempt on a peer attaches it, and returns what the
    /// audit collected.
    async fn relay_watched(
        adapter: ResponseAdapter,
        answer: Vec<u8>,
    ) -> Result<Option<serde_json::Value>, String> {
        let probe = Arc::new(CollectorProbe::default());
        let watch = ReceiptWatch::observed_by(probe.clone());
        let (mut peer, mut upstream) = tokio::io::duplex(1 << 20);
        peer.write_all(&answer).await.unwrap();
        peer.shutdown().await.unwrap();
        let mut client = ClientStream::null();
        watch.observe(&mut client, adapter);
        let mut upstream = watch.upstream(&mut upstream, adapter);
        let probed = probe_http_response(&mut upstream)
            .await
            .expect("a response");
        let context = RelayAttemptContext {
            request_id: RequestId::new(),
            disconnect_message: "test: disconnected",
            commit_message: "test: committed",
            served_by: None,
            route_observer: OpenAiRouteObserver::default(),
        };
        let policy = ResponseRetryPolicy::next_target_available(false);
        relay_attempted_response(&mut client, &mut upstream, probed, context, policy, adapter)
            .await;
        upstream.finish();
        watch.stop_observing(&mut client);
        probe.receipt().expect("the watch saw the response end")
    }

    /// `/v1/messages` and `/v1/responses` are served as chat completions and
    /// translated back for the client. The audit reads the peer's own
    /// answer, before the translation drops its receipt.
    #[tokio::test]
    async fn a_translated_answer_keeps_its_receipt_for_the_audit() {
        for (adapter, answer) in [
            (ResponseAdapter::None, json_answer()),
            (ResponseAdapter::None, stream_answer()),
            (ResponseAdapter::AnthropicMessagesJson, json_answer()),
            (ResponseAdapter::AnthropicMessagesStream, stream_answer()),
            (ResponseAdapter::OpenAiResponsesJson, json_answer()),
            (ResponseAdapter::OpenAiResponsesStream, stream_answer()),
        ] {
            let receipt = relay_watched(adapter, answer).await;
            let receipt = receipt.unwrap_or_else(|why| panic!("{adapter:?}: no turn: {why}"));
            let receipt = receipt.unwrap_or_else(|| panic!("{adapter:?}: no receipt"));
            assert_eq!(receipt["receipt_id"], "r", "{adapter:?}");
        }
        // a peer's answer that broke off is no turn, however it reaches the
        // client: never a turn that owed a receipt
        let mut cut = stream_answer();
        cut.truncate(cut.len() - 40);
        let mut short = json_answer();
        short.truncate(short.len() - 10);
        for (adapter, answer) in [
            (ResponseAdapter::AnthropicMessagesStream, cut.clone()),
            (ResponseAdapter::OpenAiResponsesStream, cut),
            (ResponseAdapter::AnthropicMessagesJson, short.clone()),
            (ResponseAdapter::OpenAiResponsesJson, short),
        ] {
            let collected = relay_watched(adapter, answer).await;
            assert!(collected.is_err(), "{adapter:?}: {collected:?}");
        }
    }
}
