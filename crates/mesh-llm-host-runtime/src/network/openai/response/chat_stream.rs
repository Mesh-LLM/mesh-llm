use super::cache_cost::parse_cache_cost_from_json_body;
use super::common::{
    ResponseRetryPolicy, RouteAttemptResult, parse_token_usage_from_json_body,
    sse_data_frame_is_openai_error,
};
use super::probe::{
    ResponseProbe, append_capsule_nonce_headers, append_mesh_served_by_header,
    response_is_event_stream, try_parse_response_headers,
};
use super::relay::{relay_error_response, relay_success_response};
use super::stream_translation::write_captured_sse_event;
use crate::logging::{OpenAiRouteObserver, OpenAiStreamArtifactCapture};
use crate::network::openai::{
    client_stream::ClientStream, tool_call_ids::ChatStreamNormalizationState,
};
use anyhow::{Result, anyhow};
use tokio::io::{AsyncRead, AsyncWriteExt};

pub(in crate::network::openai::response) async fn relay_normalized_chat_completion_stream<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    relay_chat_protocol_stream(
        tcp_stream,
        reader,
        probe,
        retry_policy,
        served_by,
        route_observer,
        false,
    )
    .await
}

pub(in crate::network::openai::response) async fn relay_translated_messages_stream<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    relay_chat_protocol_stream(
        tcp_stream,
        reader,
        probe,
        retry_policy,
        served_by,
        route_observer,
        true,
    )
    .await
}

async fn write_chat_protocol_event(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    messages: &mut Option<openai_frontend::anthropic::MessagesWireStream>,
    data: &str,
) -> Result<()> {
    if let Some(messages) = messages {
        for event in messages.push(data)? {
            write_captured_sse_event(
                tcp_stream,
                capture,
                Some(event.event_name()),
                &serde_json::to_string(&event)?,
            )
            .await?;
        }
    } else {
        write_captured_sse_event(tcp_stream, capture, None, data).await?;
    }
    Ok(())
}

pub(in crate::network::openai::response) async fn relay_chat_protocol_stream<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
    anthropic: bool,
) -> Result<RouteAttemptResult> {
    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        route_observer.stream_error("upstream_status");
        if anthropic {
            return super::json_adaptation::relay_translated_messages_json(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
            )
            .await;
        }
        return relay_error_response(tcp_stream, reader, probe, served_by, route_observer).await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    if !response_is_event_stream(&parsed) {
        return relay_non_streaming_reply(
            tcp_stream,
            reader,
            probe,
            retry_policy,
            served_by,
            route_observer,
            anthropic,
        )
        .await;
    }

    let mut body_reader = super::body_reader::BodyReader::new(
        reader,
        probe.buffered[parsed.header_end..].to_vec(),
        parsed.chunked,
        parsed.content_length,
    );
    let mut carry = Vec::new();
    let mut state = ChatStreamNormalizationState::default();
    let mut messages = anthropic.then(openai_frontend::anthropic::MessagesWireStream::new);
    let mut observed_usage = None;
    let mut observed_cache_cost = None;
    let mut header = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n",
    );
    append_capsule_nonce_headers(
        &mut header,
        parsed.client_nonce.as_deref(),
        parsed.nonce_origin.as_deref(),
    );
    append_mesh_served_by_header(&mut header, served_by);
    header.push_str("Connection: close\r\n\r\n");
    tcp_stream.write_all(header.as_bytes()).await?;
    let mut response_capture = route_observer.begin_stream_response_capture();
    route_observer.stream_started(None);

    let mut done_seen = false;
    let mut first_chunk_seen = false;
    let mut upstream_error_seen = false;
    loop {
        let mut processed = 0usize;
        while let Some((frame_end_rel, delimiter)) = carry[processed..]
            .windows(2)
            .position(|bytes| bytes == b"\n\n")
            .map(|offset| (offset, 2))
            .into_iter()
            .chain(
                carry[processed..]
                    .windows(4)
                    .position(|bytes| bytes == b"\r\n\r\n")
                    .map(|offset| (offset, 4)),
            )
            .min_by_key(|(offset, _)| *offset)
        {
            let frame_end = processed + frame_end_rel;
            let frame = std::str::from_utf8(&carry[processed..frame_end])?;
            processed = frame_end + delimiter;
            let data_lines = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim_start)
                .collect::<Vec<_>>();
            if data_lines.is_empty() {
                continue;
            }
            let data = data_lines.join("\n");
            if data == "[DONE]" {
                done_seen = true;
                write_chat_protocol_event(
                    tcp_stream,
                    &mut response_capture,
                    &mut messages,
                    "[DONE]",
                )
                .await?;
                break;
            }

            if !upstream_error_seen && sse_data_frame_is_openai_error(&data) {
                // The upstream backend frames failures as OpenAI error bodies
                // inside a 200 stream. Relay the frame untouched, but do not
                // let it count as stream progress or terminal success.
                upstream_error_seen = true;
            }
            if let Some(usage) = parse_token_usage_from_json_body(data.as_bytes()) {
                observed_usage = Some(usage);
            }
            observed_cache_cost =
                observed_cache_cost.or_else(|| parse_cache_cost_from_json_body(data.as_bytes()));
            let normalized = state.normalize_data(&data);
            write_chat_protocol_event(
                tcp_stream,
                &mut response_capture,
                &mut messages,
                &normalized,
            )
            .await?;
            if upstream_error_seen {
                continue;
            }
            if first_chunk_seen {
                route_observer.stream_chunk();
            } else {
                route_observer.stream_first_token();
                first_chunk_seen = true;
            }
        }
        if processed > 0 {
            carry.drain(..processed);
        }

        if done_seen {
            break;
        }

        let Some(bytes) = body_reader.next().await? else {
            break;
        };
        carry.extend(bytes);
        if carry.len() > 8 * 1024 * 1024 {
            return Err(anyhow!("upstream SSE frame exceeds 8 MiB"));
        }
    }

    write_truncated_message(
        tcp_stream,
        &mut response_capture,
        messages.as_mut(),
        done_seen,
    )
    .await?;
    let _ = tcp_stream.write_all(b"0\r\n\r\n").await;
    let _ = tcp_stream.shutdown().await;
    if upstream_error_seen {
        // An embedded upstream error frame is terminal even when the upstream
        // never sent [DONE]: report the failure reason it carried rather than
        // a generic incomplete-stream truncation.
        route_observer.stream_error("upstream_stream_error");
        return Ok(RouteAttemptResult::Delivered {
            status_code: 200,
            usage: None,
            cache_cost: None,
        });
    }
    if !done_seen {
        route_observer.stream_error("upstream_stream_incomplete");
        return Err(anyhow!("upstream chat stream ended before [DONE]"));
    }
    route_observer.complete_stream_response_capture(response_capture);
    route_observer.stream_completed(observed_usage);
    Ok(RouteAttemptResult::Delivered {
        status_code: 200,
        usage: observed_usage,
        cache_cost: observed_cache_cost,
    })
}

async fn write_truncated_message(
    tcp_stream: &mut ClientStream,
    response_capture: &mut Option<OpenAiStreamArtifactCapture>,
    messages: Option<&mut openai_frontend::anthropic::MessagesWireStream>,
    done_seen: bool,
) -> Result<()> {
    if !done_seen && let Some(messages) = messages {
        for event in messages.truncated() {
            write_captured_sse_event(
                tcp_stream,
                response_capture,
                Some(event.event_name()),
                &serde_json::to_string(&event)?,
            )
            .await?;
        }
    }
    Ok(())
}

async fn relay_non_streaming_reply<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
    anthropic: bool,
) -> Result<RouteAttemptResult> {
    if anthropic {
        super::json_adaptation::relay_translated_messages_json(
            tcp_stream,
            reader,
            probe,
            retry_policy,
            served_by,
            route_observer,
        )
        .await
    } else {
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
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    async fn relay_anthropic_upstream(
        upstream_response: &[u8],
        served_by: Option<&str>,
    ) -> (Vec<u8>, tokio::task::JoinHandle<RouteAttemptResult>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let served_by = served_by.map(str::to_string);
        let upstream_response = upstream_response.to_vec();
        let sse_marker = b"text/event-stream";
        let is_event_stream_response = upstream_response
            .windows(sse_marker.len())
            .any(|window| window == sse_marker);
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let header_end = upstream_response
                .windows(4)
                .position(|bytes| bytes == b"\r\n\r\n")
                .expect("response headers")
                + 4;
            let probe = ResponseProbe {
                buffered: upstream_response,
                header_end,
                status_code: 200,
                retryable_context_overflow: false,
            };
            relay_translated_messages_stream(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                served_by.as_deref(),
                OpenAiRouteObserver::default(),
            )
            .await
            .expect("relay")
        });

        if is_event_stream_response {
            upstream_writer
                .write_all(
                    b"data: {\"id\":\"chatcmpl-a\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"qwen\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
                )
                .await
                .unwrap();
            upstream_writer
                .write_all(b"data: [DONE]\n\n")
                .await
                .unwrap();
        }
        upstream_writer.shutdown().await.unwrap();

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        (output, server_task)
    }

    #[tokio::test]
    async fn anthropic_stream_echoes_served_by_header_only_when_set() {
        let sse_headers =
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";

        let (with_target_output, with_target_task) =
            relay_anthropic_upstream(sse_headers, Some("peer-endpoint-hex")).await;
        let body = String::from_utf8_lossy(&with_target_output);
        assert!(
            body.contains("x-mesh-served-by: peer-endpoint-hex\r\n"),
            "anthropic SSE must echo the resolved peer:\n{body}"
        );
        assert!(
            body.contains("message_start"),
            "missing message_start:\n{body}"
        );
        assert!(
            body.contains("message_stop"),
            "missing message_stop:\n{body}"
        );
        with_target_task.await.expect("relay");

        let (without_target_output, without_target_task) =
            relay_anthropic_upstream(sse_headers, None).await;
        assert!(
            !String::from_utf8_lossy(&without_target_output).contains("x-mesh-served-by"),
            "absent x-mesh-target must not add x-mesh-served-by"
        );
        without_target_task.await.expect("relay");
    }

    #[tokio::test]
    async fn anthropic_non_streaming_reply_echoes_served_by_header_only_when_set() {
        let completion = r#"{"id":"chatcmpl-b","object":"chat.completion","created":1,"model":"qwen","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        let upstream = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{completion}",
            completion.len()
        );

        let (with_target_output, with_target_task) =
            relay_anthropic_upstream(upstream.as_bytes(), Some("peer-endpoint-hex")).await;
        let body = String::from_utf8_lossy(&with_target_output);
        assert!(
            body.contains("x-mesh-served-by: peer-endpoint-hex\r\n"),
            "anthropic JSON translation must echo the resolved peer:\n{body}"
        );
        assert!(
            body.contains("\"type\":\"message\""),
            "expected an Anthropic message envelope:\n{body}"
        );
        with_target_task.await.expect("relay");

        let (without_target_output, without_target_task) =
            relay_anthropic_upstream(upstream.as_bytes(), None).await;
        assert!(
            !String::from_utf8_lossy(&without_target_output).contains("x-mesh-served-by"),
            "absent x-mesh-target must not add x-mesh-served-by"
        );
        without_target_task.await.expect("relay");
    }
}
