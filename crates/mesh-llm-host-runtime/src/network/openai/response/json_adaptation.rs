use super::common::{
    ResponseRetryPolicy, RouteAttemptResult, parse_completion_tokens_from_json_body,
    retryable_quality_result,
};
use super::probe::{
    ResponseBodyReadLimits, ResponseProbe, read_transformed_response_body,
    try_parse_response_headers,
};
use super::relay::relay_error_response;
use crate::network::openai::response_adapter;
use crate::network::openai::tool_call_ids::normalize_chat_completion_json_body;
use anyhow::{Result, anyhow};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWriteExt};
use tokio::net::TcpStream;

const MAX_TRANSFORMED_RESPONSE_BODY_BYTES: usize = 8 * 1024 * 1024;
const TRANSFORMED_RESPONSE_BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const TRANSFORMED_RESPONSE_READ_LIMITS: ResponseBodyReadLimits = ResponseBodyReadLimits {
    max_body_bytes: MAX_TRANSFORMED_RESPONSE_BODY_BYTES,
    idle_timeout: TRANSFORMED_RESPONSE_BODY_IDLE_TIMEOUT,
};

pub(in crate::network::openai::response) async fn relay_translated_responses_json<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
) -> Result<RouteAttemptResult> {
    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe).await;
    }
    let mut buffered = probe.buffered;
    let parsed = try_parse_response_headers(&buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    let body_end = read_transformed_response_body(
        reader,
        &mut buffered,
        parsed.header_end,
        parsed.content_length,
        TRANSFORMED_RESPONSE_READ_LIMITS,
    )
    .await?;
    let body = &buffered[parsed.header_end..body_end];
    if let Some(result) = retryable_quality_result(body, retry_policy) {
        return Ok(result);
    }
    let translated_body = response_adapter::translate_chat_completion_to_responses(body)?;
    let completion_tokens = parse_completion_tokens_from_json_body(&translated_body);
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        translated_body.len()
    );
    tcp_stream.write_all(header.as_bytes()).await?;
    tcp_stream.write_all(&translated_body).await?;
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
        completion_tokens,
    })
}

pub(in crate::network::openai::response) async fn relay_normalized_chat_completion_json<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut TcpStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
) -> Result<RouteAttemptResult> {
    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe).await;
    }
    let mut buffered = probe.buffered;
    let parsed = try_parse_response_headers(&buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    let body_end = read_transformed_response_body(
        reader,
        &mut buffered,
        parsed.header_end,
        parsed.content_length,
        TRANSFORMED_RESPONSE_READ_LIMITS,
    )
    .await?;
    let body = &buffered[parsed.header_end..body_end];
    let normalized_body =
        normalize_chat_completion_json_body(body).unwrap_or_else(|| body.to_vec());
    if let Some(result) = retryable_quality_result(&normalized_body, retry_policy) {
        return Ok(result);
    }
    let completion_tokens = parse_completion_tokens_from_json_body(&normalized_body);
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        normalized_body.len()
    );
    tcp_stream.write_all(header.as_bytes()).await?;
    tcp_stream.write_all(&normalized_body).await?;
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
        completion_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[test]
    fn test_translate_chat_completion_to_responses_json() {
        let translated = response_adapter::translate_chat_completion_to_responses(
            serde_json::json!({
                "id": "chatcmpl_123",
                "object": "chat.completion",
                "created": 1234,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello from mesh"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8
                }
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let response: serde_json::Value = serde_json::from_slice(&translated).unwrap();

        assert_eq!(response["object"], "response");
        assert_eq!(response["model"], "test-model");
        assert_eq!(response["output_text"], "hello from mesh");
        assert_eq!(response["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(response["usage"]["input_tokens"], 5);
        assert_eq!(response["usage"]["output_tokens"], 3);
        assert_eq!(response["usage"]["total_tokens"], 8);
    }

    #[tokio::test]
    async fn relay_normalized_chat_completion_json_adds_missing_tool_call_id() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let body = br#"{"id":"chatcmpl-a","object":"chat.completion","created":1,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup_fixture_fact","arguments":"{\"key\":\"codeword\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":4}}"#;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let header_end = header.len();
        let server_task = tokio::spawn(async move {
            let (mut client_socket, _) = listener.accept().await.unwrap();
            let probe = ResponseProbe {
                buffered: header.into_bytes(),
                header_end,
                status_code: 200,
                retryable_context_overflow: false,
            };
            relay_normalized_chat_completion_json(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
            )
            .await
            .expect("relay")
        });

        upstream_writer.write_all(body).await.unwrap();
        let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), client.read_to_end(&mut output))
            .await
            .expect("relay should not wait for upstream keep-alive close")
            .unwrap();
        drop(upstream_writer);
        let route_result = server_task.await.expect("server task");
        let body_start = output
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&output[body_start..]).unwrap();

        assert_eq!(
            route_result,
            RouteAttemptResult::Delivered {
                status_code: 200,
                completion_tokens: Some(4),
            }
        );
        assert_eq!(
            parsed["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_mesh_chatcmpl_a_0_0"
        );
        assert_eq!(
            parsed["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "lookup_fixture_fact"
        );
    }
}
