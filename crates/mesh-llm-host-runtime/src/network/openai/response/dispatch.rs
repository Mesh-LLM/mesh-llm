use super::common::{ResponseRetryPolicy, RouteAttemptResult, is_client_disconnect_error};
use super::json_adaptation::{
    relay_normalized_chat_completion_json, relay_translated_responses_json,
};
use super::probe::{ResponseProbe, try_parse_response_headers};
use super::relay::{relay_error_response, relay_success_response};
use super::stream_translation::{
    relay_normalized_chat_completion_stream, relay_translated_responses_stream,
};
use crate::network::openai::request_normalize::ResponseAdapter;
use anyhow::{Result, anyhow};
use tokio::io::AsyncRead;
use tokio::net::TcpStream;

pub(in crate::network::openai::response) async fn relay_probed_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> Result<RouteAttemptResult> {
    if let Some(result) = relay_adapted_response(
        tcp_stream,
        reader,
        probe.clone(),
        retry_policy,
        response_adapter,
    )
    .await?
    {
        return Ok(result);
    }

    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }
    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe).await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    relay_success_response(tcp_stream, reader, probe, parsed, retry_policy).await
}

async fn relay_adapted_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> Result<Option<RouteAttemptResult>> {
    match response_adapter {
        ResponseAdapter::OpenAiChatCompletionsJson => Ok(Some(
            relay_normalized_chat_completion_json(tcp_stream, reader, probe, retry_policy).await?,
        )),
        ResponseAdapter::OpenAiChatCompletionsStream => Ok(Some(
            relay_normalized_chat_completion_stream(tcp_stream, reader, probe, retry_policy)
                .await?,
        )),
        ResponseAdapter::OpenAiResponsesJson => Ok(Some(
            relay_translated_responses_json(tcp_stream, reader, probe, retry_policy).await?,
        )),
        ResponseAdapter::OpenAiResponsesStream => Ok(Some(
            relay_translated_responses_stream(tcp_stream, reader, probe, retry_policy).await?,
        )),
        ResponseAdapter::None => Ok(None),
    }
}

pub(in crate::network::openai::response) async fn relay_attempted_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    disconnect_message: &str,
    commit_message: &str,
) -> RouteAttemptResult {
    let status_code = probe.status_code;
    match relay_probed_response(tcp_stream, reader, probe, retry_policy, response_adapter).await {
        Ok(result) => result,
        Err(err) => {
            if is_client_disconnect_error(&err) {
                tracing::info!("{disconnect_message}");
                return RouteAttemptResult::ClientDisconnected;
            }
            tracing::debug!("{commit_message}: {err}");
            RouteAttemptResult::Delivered {
                status_code,
                completion_tokens: None,
            }
        }
    }
}
