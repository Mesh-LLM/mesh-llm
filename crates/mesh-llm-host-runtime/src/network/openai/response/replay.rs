//! The dispatch outcome the ordinary relay reports for a backend response,
//! computed from that response's raw bytes.
//!
//! The paid seller forwards its backend's raw HTTP response to the payer and
//! never runs the relay itself; the payer runs the relay over those same
//! bytes. Replaying them here through the same relay (into a discard sink)
//! gives the seller the same usage and response digests the payer's relay
//! reports, as the free path's relay reports them for a free exchange.

use super::common::{ResponseRetryPolicy, RouteAttemptResult};
use super::routing::route_local_attempt_after_forward;
use crate::logging::OpenAiRouteObserver;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::request_normalize::ResponseAdapter;
use crate::network::openai::transport::{RouteDispatchOutcome, delivered_outcome};
use mesh_llm_events::logging::identifiers::RequestId;
use tokio::io::AsyncWriteExt;

/// The outcome (status, usage, output digests) the relay reports for `raw`,
/// a complete backend HTTP response. `Failed` when the bytes are not one.
pub(crate) async fn served_outcome_of_raw_response(
    raw: &[u8],
    adapter: ResponseAdapter,
) -> RouteDispatchOutcome {
    // Room for every byte, so the write completes before the relay reads.
    let (mut writer, mut upstream) = tokio::io::duplex(raw.len().max(1));
    if writer.write_all(raw).await.is_err() {
        return RouteDispatchOutcome::Failed("could not replay the served response");
    }
    drop(writer);
    let mut sink = ClientStream::null();
    match route_local_attempt_after_forward(
        &mut sink,
        &mut upstream,
        0,
        RequestId::new(),
        ResponseRetryPolicy::next_target_available(false),
        adapter,
        None,
        None,
        OpenAiRouteObserver::default(),
    )
    .await
    {
        RouteAttemptResult::Delivered {
            status_code,
            usage,
            output_digests,
            ..
        } => delivered_outcome(status_code, usage, output_digests),
        _ => RouteDispatchOutcome::Failed("the served response did not relay"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::openai_exchange::ExchangeOutputDigests;

    fn http(content_type: &str, body: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    /// A JSON response replays to the digests the relay computes on the body.
    #[tokio::test]
    async fn a_json_response_replays_to_the_relay_digests() {
        let body = r#"{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#;
        let outcome = served_outcome_of_raw_response(
            &http("application/json", body),
            ResponseAdapter::OpenAiChatCompletionsJson,
        )
        .await;
        let RouteDispatchOutcome::RespondedWithUsage {
            status_code,
            output_digests,
            ..
        } = outcome
        else {
            panic!("expected a served outcome with usage, got {outcome:?}");
        };
        assert_eq!(status_code, 200);
        assert_eq!(
            output_digests,
            ExchangeOutputDigests::from_response_body(body.as_bytes())
        );
        assert!(output_digests.has_any());
    }

    #[tokio::test]
    async fn streamed_chat_replay_matches_the_payer_relay() {
        use tokio::io::AsyncReadExt;
        let body = concat!(
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n",
            "data: [DONE]\n\n",
        );
        let raw = http("text/event-stream", body);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (socket, _) = listener.accept().await.unwrap();
        let payer_raw = raw.clone();
        let payer = tokio::spawn(async move {
            let (mut writer, mut upstream) = tokio::io::duplex(payer_raw.len());
            writer.write_all(&payer_raw).await.unwrap();
            drop(writer);
            let mut sink = ClientStream::from(socket);
            route_local_attempt_after_forward(
                &mut sink,
                &mut upstream,
                0,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::OpenAiChatCompletionsStream,
                None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });
        let mut received = Vec::new();
        client.read_to_end(&mut received).await.unwrap();
        assert!(String::from_utf8_lossy(&received).contains("data: [DONE]"));
        let RouteAttemptResult::Delivered {
            status_code,
            usage,
            output_digests,
            ..
        } = payer.await.unwrap()
        else {
            panic!("payer did not deliver stream");
        };
        let usage = usage.expect("stream usage");
        assert_eq!(usage.completion_tokens, Some(1));
        assert!(output_digests.has_any());
        let replayed =
            served_outcome_of_raw_response(&raw, ResponseAdapter::OpenAiChatCompletionsStream)
                .await;
        assert_eq!(
            replayed,
            delivered_outcome(status_code, Some(usage), output_digests)
        );
        assert_ne!(
            replayed,
            served_outcome_of_raw_response(&raw, ResponseAdapter::None).await
        );
    }

    #[tokio::test]
    async fn bytes_that_are_not_a_response_never_yield_digests() {
        let outcome =
            served_outcome_of_raw_response(b"not http at all", ResponseAdapter::None).await;
        assert!(
            matches!(outcome, RouteDispatchOutcome::Failed(_)),
            "{outcome:?}"
        );
    }
}
