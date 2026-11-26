use super::chat_stream::relay_translated_messages_stream;
use super::common::{
    CommittedStreamFailureCause, RelayIoError, ResponseRetryPolicy, RouteAttemptResult,
    is_client_disconnect_error,
};
use super::json_adaptation::{
    relay_normalized_chat_completion_json, relay_translated_messages_json,
    relay_translated_responses_json,
};
use super::probe::{ResponseProbe, try_parse_response_headers};
use super::raw_stream_inspector::{ChunkedFramingStatus, ChunkedFramingTracker, RawSseInspector};
use super::relay::{relay_error_response, relay_success_response_with_sse_done};
use super::stream_translation::{
    relay_normalized_chat_completion_stream, relay_translated_responses_stream,
};
use crate::logging::OpenAiRouteObserver;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::request_normalize::ResponseAdapter;
use anyhow::{Result, anyhow};
use mesh_llm_events::logging::events::TokenUsage;
use mesh_llm_events::logging::identifiers::RequestId;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

pub(super) fn is_event_stream_content_type(content_type: Option<&str>) -> bool {
    content_type.is_some_and(|ct| {
        ct.split(';')
            .next()
            .is_some_and(|media| media.trim().eq_ignore_ascii_case("text/event-stream"))
    })
}

#[derive(Debug, Default)]
pub(super) struct UnboundedBodyCopyOutcome {
    pub(super) copied_bytes: u64,
    pub(super) usage: Option<TokenUsage>,
    pub(super) saw_error: bool,
    pub(super) saw_done: bool,
    pub(super) chunked_framing_complete: bool,
}

fn feed_raw_stream_inspectors(
    bytes: &[u8],
    chunk_tracker: &mut Option<ChunkedFramingTracker>,
    sse_inspector: &mut Option<RawSseInspector>,
    route_observer: OpenAiRouteObserver<'_>,
) {
    if let Some(tracker) = chunk_tracker.as_mut() {
        tracker.feed(bytes, |entity| {
            if let Some(inspector) = sse_inspector.as_mut() {
                inspector.feed(entity, || {
                    route_observer.inference_observation_first_output();
                });
            }
        });
    } else if let Some(inspector) = sse_inspector.as_mut() {
        inspector.feed(bytes, || {
            route_observer.inference_observation_first_output();
        });
    }
}

pub(super) async fn copy_unbounded_response_body<R: AsyncRead + Unpin>(
    reader: &mut R,
    tcp_stream: &mut ClientStream,
    content_type: Option<&str>,
    chunked: bool,
    initial_body: &[u8],
    route_observer: OpenAiRouteObserver<'_>,
) -> std::result::Result<UnboundedBodyCopyOutcome, RelayIoError> {
    let mut chunk_tracker = chunked.then(ChunkedFramingTracker::default);
    let mut sse_inspector =
        is_event_stream_content_type(content_type).then(RawSseInspector::default);
    if !initial_body.is_empty() {
        feed_raw_stream_inspectors(
            initial_body,
            &mut chunk_tracker,
            &mut sse_inspector,
            route_observer,
        );
    }
    let mut copied_bytes = 0u64;
    let mut buf = [0u8; 8192];
    loop {
        let n = reader
            .read(&mut buf)
            .await
            .map_err(RelayIoError::UpstreamRead)?;
        if n == 0 {
            break;
        }
        tcp_stream
            .write_all(&buf[..n])
            .await
            .map_err(RelayIoError::DownstreamWrite)?;
        feed_raw_stream_inspectors(
            &buf[..n],
            &mut chunk_tracker,
            &mut sse_inspector,
            route_observer,
        );
        copied_bytes += n as u64;
    }
    let chunked_framing_complete = chunk_tracker
        .as_ref()
        .is_none_or(|tracker| tracker.finish() == ChunkedFramingStatus::Complete);
    let sse_outcome = sse_inspector
        .map(RawSseInspector::finish)
        .unwrap_or_default();
    Ok(UnboundedBodyCopyOutcome {
        copied_bytes,
        usage: sse_outcome.usage,
        saw_error: sse_outcome.saw_error,
        saw_done: sse_outcome.saw_done,
        chunked_framing_complete,
    })
}

pub(in crate::network::openai::response) struct RelayAttemptContext<'a> {
    pub(in crate::network::openai::response) request_id: RequestId,
    pub(in crate::network::openai::response) disconnect_message: &'a str,
    pub(in crate::network::openai::response) commit_message: &'a str,
    pub(in crate::network::openai::response) served_by: Option<&'a str>,
    pub(in crate::network::openai::response) route_observer: OpenAiRouteObserver<'a>,
    pub(in crate::network::openai::response) require_sse_done: bool,
}

/// Relay one probed upstream response through the adapter, error, or success path.
#[allow(clippy::too_many_arguments)]
pub(in crate::network::openai::response) async fn relay_probed_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
    require_sse_done: bool,
) -> Result<RouteAttemptResult> {
    if let Some(result) = relay_adapted_response(
        tcp_stream,
        reader,
        probe.clone(),
        request_id,
        retry_policy,
        response_adapter,
        served_by,
        route_observer,
    )
    .await?
    {
        return Ok(result);
    }

    if let Some(result) = super::common::retry_before_relay(&probe, retry_policy) {
        return Ok(result);
    }
    if !(200..300).contains(&probe.status_code) {
        if tcp_stream.is_anthropic() {
            return relay_translated_messages_json(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
                request_id,
            )
            .await;
        }
        return relay_error_response(
            tcp_stream,
            reader,
            probe,
            served_by,
            route_observer,
            request_id,
        )
        .await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    relay_success_response_with_sse_done(
        tcp_stream,
        reader,
        probe,
        parsed,
        retry_policy,
        served_by,
        route_observer,
        request_id,
        require_sse_done,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn relay_adapted_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<Option<RouteAttemptResult>> {
    match response_adapter {
        ResponseAdapter::AnthropicMessagesJson => Ok(Some(
            relay_translated_messages_json(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
                request_id,
            )
            .await?,
        )),
        ResponseAdapter::AnthropicMessagesStream => Ok(Some(
            relay_translated_messages_stream(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
                request_id,
            )
            .await?,
        )),
        ResponseAdapter::OpenAiChatCompletionsJson => Ok(Some(
            relay_normalized_chat_completion_json(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
                request_id,
            )
            .await?,
        )),
        ResponseAdapter::OpenAiChatCompletionsStream => Ok(Some(
            relay_normalized_chat_completion_stream(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
                request_id,
            )
            .await?,
        )),
        ResponseAdapter::OpenAiResponsesJson => Ok(Some(
            relay_translated_responses_json(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
                request_id,
            )
            .await?,
        )),
        ResponseAdapter::OpenAiResponsesStream => Ok(Some(
            relay_translated_responses_stream(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
                request_id,
            )
            .await?,
        )),
        ResponseAdapter::None => Ok(None),
    }
}

pub(in crate::network::openai::response) async fn relay_attempted_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    context: RelayAttemptContext<'_>,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    let status_code = probe.status_code;
    match relay_probed_response(
        tcp_stream,
        reader,
        probe,
        context.request_id,
        retry_policy,
        response_adapter,
        context.served_by,
        context.route_observer,
        context.require_sse_done,
    )
    .await
    {
        Ok(result) => result,
        Err(err) => map_relay_attempt_error(status_code, &context, response_adapter, err),
    }
}

fn map_relay_attempt_error(
    status_code: u16,
    context: &RelayAttemptContext<'_>,
    response_adapter: ResponseAdapter,
    err: anyhow::Error,
) -> RouteAttemptResult {
    if is_streaming_response_adapter(response_adapter) {
        if is_client_disconnect_error(&err) {
            context.route_observer.stream_cancelled();
        } else {
            context.route_observer.stream_error("stream_relay_failed");
        }
    }
    if is_client_disconnect_error(&err) {
        tracing::info!("{}", context.disconnect_message);
        return RouteAttemptResult::ClientDisconnected;
    }
    tracing::debug!("{}: {err}", context.commit_message);
    RouteAttemptResult::CommittedStreamFailure {
        status_code: if is_streaming_response_adapter(response_adapter) {
            200
        } else {
            status_code
        },
        cause: CommittedStreamFailureCause::Truncated,
    }
}

const fn is_streaming_response_adapter(response_adapter: ResponseAdapter) -> bool {
    matches!(
        response_adapter,
        ResponseAdapter::OpenAiChatCompletionsStream
            | ResponseAdapter::OpenAiResponsesStream
            | ResponseAdapter::AnthropicMessagesStream
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::inference_observation::hub::ObservationHub;
    use crate::plugin::inference_observation::plugin_endpoint::PluginExecutionObservation;
    use mesh_llm_plugin::proto::inference_observation_event::Kind;
    use std::sync::Arc;
    use tokio::net::{TcpListener, TcpStream};

    async fn run_copy_case(
        content_type: Option<&str>,
        initial_body: &[u8],
        remaining: &[u8],
    ) -> (Vec<u8>, bool, Option<TokenUsage>) {
        let hub = Arc::new(ObservationHub::new());
        let (sub, _, _) = hub.subscribe();
        let obs = PluginExecutionObservation::start(
            hub.clone(),
            Some("dev/echo"),
            "/v1/completions",
            true,
        )
        .expect("start streaming observation");
        let observer =
            OpenAiRouteObserver::default().with_optional_inference_observation(Some(&obs));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let reader_task = tokio::spawn(async move {
            let mut client = TcpStream::connect(addr).await.unwrap();
            let mut received = Vec::new();
            client.read_to_end(&mut received).await.unwrap();
            received
        });
        let (server_stream, _) = listener.accept().await.unwrap();
        let mut client_stream = ClientStream::from(server_stream);
        let mut cursor = std::io::Cursor::new(remaining.to_vec());

        let outcome = copy_unbounded_response_body(
            &mut cursor,
            &mut client_stream,
            content_type,
            false,
            initial_body,
            observer,
        )
        .await
        .unwrap();
        assert_eq!(outcome.copied_bytes, remaining.len() as u64);
        let _ = client_stream.shutdown().await;
        drop(client_stream);
        obs.complete(outcome.usage.as_ref());

        let received = reader_task.await.unwrap();
        let drained = hub.drain(sub, usize::MAX).unwrap();
        let saw_first_output = drained
            .events
            .iter()
            .any(|e| matches!(e.kind.as_ref(), Some(Kind::FirstOutput(_))));
        (received, saw_first_output, outcome.usage)
    }

    #[tokio::test]
    async fn copy_unbounded_response_body_signals_first_output_only_for_non_empty_sse() {
        let payload = b"data: {\"text\":\"hi\"}\n\n";
        let (rx, first, _) = run_copy_case(Some("text/event-stream"), b"data: {}", payload).await;
        assert_eq!(rx, payload);
        assert!(first, "SSE with initial_body must signal first_output");

        let (rx, first, _) =
            run_copy_case(Some("text/event-stream; charset=utf-8"), b"", payload).await;
        assert_eq!(rx, payload);
        assert!(first, "SSE first chunk read must signal first_output");

        let (rx, first, _) = run_copy_case(Some("text/event-stream"), b"", b"").await;
        assert!(rx.is_empty());
        assert!(!first, "empty SSE EOF must not signal first_output");

        let (rx, first, _) = run_copy_case(Some("application/octet-stream"), b"", payload).await;
        assert_eq!(rx, payload);
        assert!(!first, "non-SSE body must not signal first_output");

        let (rx, first, _) =
            run_copy_case(Some("text/event-stream"), b": keep-alive\n\n", b"").await;
        assert!(rx.is_empty());
        assert!(
            !first,
            "SSE keep-alive comment must not signal first_output"
        );
    }

    #[tokio::test]
    async fn adapted_stream_upstream_reset_reports_failed_not_dropped() {
        use std::pin::Pin;
        use std::task::{Context, Poll};

        struct ResetAfterPartialReader {
            sent_first: bool,
        }
        impl tokio::io::AsyncRead for ResetAfterPartialReader {
            fn poll_read(
                mut self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                buf: &mut tokio::io::ReadBuf<'_>,
            ) -> Poll<std::io::Result<()>> {
                if !self.sent_first {
                    self.sent_first = true;
                    buf.put_slice(
                        b"data: {\"id\":\"chatcmpl-r\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
                    );
                    return Poll::Ready(Ok(()));
                }
                Poll::Ready(Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "backend connection reset",
                )))
            }
        }

        for adapter in [
            ResponseAdapter::OpenAiChatCompletionsStream,
            ResponseAdapter::OpenAiResponsesStream,
            ResponseAdapter::AnthropicMessagesStream,
        ] {
            let hub = Arc::new(ObservationHub::new());
            let (sub, _, _) = hub.subscribe();
            let obs = PluginExecutionObservation::start(
                hub.clone(),
                Some("dev/echo"),
                "/v1/chat/completions",
                true,
            )
            .expect("start observation");
            let observer =
                OpenAiRouteObserver::default().with_optional_inference_observation(Some(&obs));

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let client_task = tokio::spawn(async move {
                let mut client = TcpStream::connect(addr).await.unwrap();
                let mut buf = Vec::new();
                let _ = client.read_to_end(&mut buf).await;
                buf
            });
            let (server_stream, _) = listener.accept().await.unwrap();
            let mut client_stream = ClientStream::from(server_stream);
            let header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
            let probe = ResponseProbe {
                buffered: header.to_vec(),
                header_end: header.len(),
                status_code: 200,
                retryable_context_overflow: false,
            };
            let mut upstream = ResetAfterPartialReader { sent_first: false };
            let result = relay_attempted_response(
                &mut client_stream,
                &mut upstream,
                probe,
                RelayAttemptContext {
                    request_id: RequestId::new(),
                    disconnect_message: "client disconnected",
                    commit_message: "stream commit failed",
                    served_by: None,
                    route_observer: observer,
                    require_sse_done: false,
                },
                ResponseRetryPolicy::next_target_available(false),
                adapter,
            )
            .await;
            drop(client_stream);
            let _ = client_task.await;

            assert_eq!(
                result,
                RouteAttemptResult::CommittedStreamFailure {
                    status_code: 200,
                    cause: CommittedStreamFailureCause::Truncated,
                },
                "upstream reset on {adapter:?} must report CommittedStreamFailure, not ClientDisconnected"
            );
            match result {
                RouteAttemptResult::ClientDisconnected => obs.dropped(),
                _ => obs.fail(),
            }
            let drained = hub.drain(sub, usize::MAX).unwrap();
            let ended = drained.events.iter().find_map(|e| match e.kind.as_ref() {
                Some(Kind::Ended(end)) => Some(end.outcome()),
                _ => None,
            });
            assert_eq!(
                ended,
                Some(mesh_llm_plugin::proto::InferenceExecutionOutcome::Failed),
                "upstream reset on {adapter:?} must end observation as Failed"
            );
        }
    }
}
