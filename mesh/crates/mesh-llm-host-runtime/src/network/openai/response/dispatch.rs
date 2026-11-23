use super::chat_stream::relay_translated_messages_stream;
use super::common::{
    CommittedStreamFailureCause, ResponseRetryPolicy, RouteAttemptResult,
    is_client_disconnect_error, parse_token_usage_from_json_body,
};
use super::json_adaptation::{
    relay_normalized_chat_completion_json, relay_translated_messages_json,
    relay_translated_responses_json,
};
use super::probe::{ResponseProbe, try_parse_response_headers};
use super::relay::{relay_error_response, relay_success_response};
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

#[derive(Debug)]
pub(super) enum BodyCopyError {
    Upstream(std::io::Error),
    Downstream(std::io::Error),
}

impl std::fmt::Display for BodyCopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upstream(err) => write!(f, "upstream read error: {err}"),
            Self::Downstream(err) => write!(f, "downstream write error: {err}"),
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct UnboundedBodyCopyOutcome {
    pub(super) copied_bytes: u64,
    pub(super) usage: Option<TokenUsage>,
}

const MAX_SSE_USAGE_LINE_BYTES: usize = 64 * 1024;

#[derive(Default)]
struct SseUsageTracker {
    line: Vec<u8>,
    overflowed: bool,
    usage: Option<TokenUsage>,
}

impl SseUsageTracker {
    fn feed(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if byte == b'\n' {
                self.flush_line();
            } else if !self.overflowed {
                if self.line.len() < MAX_SSE_USAGE_LINE_BYTES {
                    self.line.push(byte);
                } else {
                    self.line.clear();
                    self.overflowed = true;
                }
            }
        }
    }

    fn finish(mut self) -> Option<TokenUsage> {
        self.flush_line();
        self.usage
    }

    fn flush_line(&mut self) {
        if self.overflowed {
            self.overflowed = false;
            self.line.clear();
            return;
        }
        let trimmed = self.line.strip_suffix(b"\r").unwrap_or(&self.line);
        if let Some(rest) = trimmed.strip_prefix(b"data:") {
            let payload = rest
                .iter()
                .position(|b| !b.is_ascii_whitespace())
                .map(|start| &rest[start..])
                .unwrap_or(&[]);
            if !payload.is_empty()
                && payload != b"[DONE]"
                && let Some(usage) = parse_token_usage_from_json_body(payload)
            {
                self.usage = Some(usage);
            }
        }
        self.line.clear();
    }
}

async fn copy_reader_to_client<R: AsyncRead + Unpin>(
    reader: &mut R,
    tcp_stream: &mut ClientStream,
    mut usage_tracker: Option<&mut SseUsageTracker>,
) -> std::result::Result<u64, BodyCopyError> {
    let mut total = 0u64;
    let mut buf = [0u8; 8192];
    loop {
        let n = reader
            .read(&mut buf)
            .await
            .map_err(BodyCopyError::Upstream)?;
        if n == 0 {
            return Ok(total);
        }
        if let Some(tracker) = usage_tracker.as_deref_mut() {
            tracker.feed(&buf[..n]);
        }
        tcp_stream
            .write_all(&buf[..n])
            .await
            .map_err(BodyCopyError::Downstream)?;
        total += n as u64;
    }
}

pub(super) async fn copy_unbounded_response_body<R: AsyncRead + Unpin>(
    reader: &mut R,
    tcp_stream: &mut ClientStream,
    content_type: Option<&str>,
    initial_body: &[u8],
    route_observer: OpenAiRouteObserver<'_>,
) -> std::result::Result<UnboundedBodyCopyOutcome, BodyCopyError> {
    if !is_event_stream_content_type(content_type) {
        let copied_bytes = copy_reader_to_client(reader, tcp_stream, None).await?;
        return Ok(UnboundedBodyCopyOutcome {
            copied_bytes,
            usage: None,
        });
    }
    let mut tracker = SseUsageTracker::default();
    if !initial_body.is_empty() {
        tracker.feed(initial_body);
        route_observer.inference_observation_first_output();
        let copied_bytes = copy_reader_to_client(reader, tcp_stream, Some(&mut tracker)).await?;
        return Ok(UnboundedBodyCopyOutcome {
            copied_bytes,
            usage: tracker.finish(),
        });
    }
    let mut first = [0u8; 4096];
    match reader
        .read(&mut first)
        .await
        .map_err(BodyCopyError::Upstream)?
    {
        0 => Ok(UnboundedBodyCopyOutcome::default()),
        n => {
            tracker.feed(&first[..n]);
            route_observer.inference_observation_first_output();
            tcp_stream
                .write_all(&first[..n])
                .await
                .map_err(BodyCopyError::Downstream)?;
            let rest = copy_reader_to_client(reader, tcp_stream, Some(&mut tracker)).await?;
            Ok(UnboundedBodyCopyOutcome {
                copied_bytes: rest + n as u64,
                usage: tracker.finish(),
            })
        }
    }
}

pub(in crate::network::openai::response) struct RelayAttemptContext<'a> {
    pub(in crate::network::openai::response) request_id: RequestId,
    pub(in crate::network::openai::response) disconnect_message: &'a str,
    pub(in crate::network::openai::response) commit_message: &'a str,
    pub(in crate::network::openai::response) served_by: Option<&'a str>,
    pub(in crate::network::openai::response) route_observer: OpenAiRouteObserver<'a>,
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
    relay_success_response(
        tcp_stream,
        reader,
        probe,
        parsed,
        retry_policy,
        served_by,
        route_observer,
        request_id,
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
    }
}
