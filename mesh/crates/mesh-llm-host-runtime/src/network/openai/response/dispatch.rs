use super::chat_stream::relay_translated_messages_stream;
use super::common::{
    CommittedStreamFailureCause, ResponseRetryPolicy, RouteAttemptResult,
    is_client_disconnect_error,
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
use mesh_llm_events::logging::identifiers::RequestId;
use tokio::io::AsyncRead;

pub(in crate::network::openai::response) struct RelayAttemptContext<'a> {
    pub(in crate::network::openai::response) request_id: RequestId,
    pub(in crate::network::openai::response) disconnect_message: &'a str,
    pub(in crate::network::openai::response) commit_message: &'a str,
    pub(in crate::network::openai::response) served_by: Option<&'a str>,
    pub(in crate::network::openai::response) route_observer: OpenAiRouteObserver<'a>,
    pub(in crate::network::openai::response) responses_turn:
        Option<&'a std::sync::Arc<skippy_inference_api::responses::ResponsesTurn>>,
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
    responses_turn: Option<&std::sync::Arc<skippy_inference_api::responses::ResponsesTurn>>,
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
        responses_turn,
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
    responses_turn: Option<&std::sync::Arc<skippy_inference_api::responses::ResponsesTurn>>,
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
                responses_turn.map(std::sync::Arc::as_ref),
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
                responses_turn.cloned(),
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
        context.responses_turn,
    )
    .await
    {
        Ok(result) => result,
        Err(error) => relay_failure_result(error, context, response_adapter, status_code),
    }
}

fn relay_failure_result(
    error: anyhow::Error,
    context: RelayAttemptContext<'_>,
    response_adapter: ResponseAdapter,
    status_code: u16,
) -> RouteAttemptResult {
    let disconnected = is_client_disconnect_error(&error);
    if is_streaming_response_adapter(response_adapter) {
        if disconnected {
            context.route_observer.stream_cancelled();
        } else {
            context.route_observer.stream_error("stream_relay_failed");
        }
    }
    if disconnected {
        tracing::info!("{}", context.disconnect_message);
        return RouteAttemptResult::ClientDisconnected;
    }
    tracing::debug!("{}: {error}", context.commit_message);
    if is_streaming_response_adapter(response_adapter) {
        return RouteAttemptResult::CommittedStreamFailure {
            status_code: 200,
            cause: CommittedStreamFailureCause::Truncated,
        };
    }
    // Buffered adapters validate and translate the whole body before writing
    // a response head. Their failures are safe to retry and cannot be called
    // delivered empty successes.
    if is_buffered_response_adapter(response_adapter) {
        return RouteAttemptResult::RetryableUnavailable;
    }
    RouteAttemptResult::Delivered {
        status_code,
        usage: None,
        cache_cost: None,
        output_digests: Default::default(),
    }
}

const fn is_buffered_response_adapter(response_adapter: ResponseAdapter) -> bool {
    matches!(
        response_adapter,
        ResponseAdapter::OpenAiChatCompletionsJson
            | ResponseAdapter::OpenAiResponsesJson
            | ResponseAdapter::AnthropicMessagesJson
    )
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
    use crate::logging::OpenAiRouteObserver;

    #[tokio::test]
    async fn invalid_buffered_responses_json_is_retryable_not_delivered() {
        let header =
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1\r\n\r\n";
        let mut buffered = header.to_vec();
        buffered.push(b'{');
        let mut client = ClientStream::null();
        let mut reader = std::io::Cursor::new(Vec::<u8>::new());
        let result = relay_attempted_response(
            &mut client,
            &mut reader,
            ResponseProbe {
                buffered,
                header_end: header.len(),
                status_code: 200,
                retryable_context_overflow: false,
            },
            RelayAttemptContext {
                request_id: RequestId::new(),
                disconnect_message: "test disconnect",
                commit_message: "test failure",
                served_by: None,
                route_observer: OpenAiRouteObserver::default(),
                responses_turn: None,
            },
            ResponseRetryPolicy::next_target_available(false),
            ResponseAdapter::OpenAiResponsesJson,
        )
        .await;
        assert!(matches!(result, RouteAttemptResult::RetryableUnavailable));
    }
}
