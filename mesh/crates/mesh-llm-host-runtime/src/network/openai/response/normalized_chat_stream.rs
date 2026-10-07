//! Client-facing chat SSE normalization. Billing watermarks remain on the peer transport.
use super::normalized_stream_completion::finish_normalized_chat_stream;
use super::*;

fn observe_output_progress(observer: OpenAiRouteObserver<'_>, errored: bool, first: &mut bool) {
    if errored {
        return;
    }
    if *first {
        observer.stream_chunk();
    } else {
        observer.stream_first_token();
        *first = true;
    }
}

fn frame_data(frame: &str) -> Option<String> {
    let lines = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn hold_usage_frame(data: &str, pending: &mut Option<String>) -> bool {
    let usage_only = serde_json::from_str::<serde_json::Value>(data).is_ok_and(|value| {
        value
            .get("choices")
            .and_then(serde_json::Value::as_array)
            .is_some_and(Vec::is_empty)
            && value.get("usage").is_some_and(serde_json::Value::is_object)
            && value.get("error").is_none()
    });
    // A usage-only frame between deltas is a billing watermark, not the
    // terminal usage frame OpenAI clients expect. Keep only a trailing one.
    // Without final backend totals, this remains the last partial watermark.
    if usage_only {
        *pending = Some(data.to_owned());
    } else if !is_finish_frame(data) {
        *pending = None;
    }
    usage_only
}

fn is_finish_frame(data: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(data).is_ok_and(|value| {
        value
            .get("choices")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|choices| {
                !choices.is_empty()
                    && choices.iter().all(|choice| {
                        choice
                            .get("finish_reason")
                            .is_some_and(serde_json::Value::is_string)
                    })
            })
    })
}

async fn flush_pending_usage(
    client: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    pending: &mut Option<String>,
) -> Result<()> {
    if let Some(data) = pending.take() {
        write_captured_sse_event(client, capture, None, &data).await?;
    }
    Ok(())
}

/// Relay a streaming chat-completions upstream response, normalizing tool-call ids.
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
    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        route_observer.stream_error("upstream_status");
        return relay_error_response(tcp_stream, reader, probe, served_by, route_observer).await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    if !response_is_event_stream(&parsed) {
        return relay_success_response(
            tcp_stream,
            reader,
            probe,
            parsed,
            retry_policy,
            served_by,
            route_observer,
        )
        .await;
    }

    let mut carry = String::from_utf8_lossy(&probe.buffered[parsed.header_end..]).to_string();
    let mut state = ChatStreamNormalizationState::default();
    let mut assembly = StreamedChatAssembly::default();
    let mut observed_usage = None;
    let mut pending_usage = None;
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
        while let Some(frame_end_rel) = carry[processed..].find("\n\n") {
            let frame_end = processed + frame_end_rel;
            let frame = &carry[processed..frame_end];
            processed = frame_end + 2;
            let Some(data) = frame_data(frame) else {
                continue;
            };
            if data == "[DONE]" {
                done_seen = true;
                flush_pending_usage(tcp_stream, &mut response_capture, &mut pending_usage).await?;
                write_captured_sse_event(tcp_stream, &mut response_capture, None, "[DONE]").await?;
                break;
            }

            if !upstream_error_seen && sse_data_frame_is_openai_error(&data) {
                // The upstream backend frames failures as OpenAI error bodies
                // inside a 200 stream. Relay the frame untouched, but do not
                // let it count as stream progress or terminal success.
                upstream_error_seen = true;
                flush_pending_usage(tcp_stream, &mut response_capture, &mut pending_usage).await?;
            }
            if let Some(usage) = parse_token_usage_from_json_body(data.as_bytes()) {
                observed_usage = Some(usage);
            }
            observed_cache_cost =
                observed_cache_cost.or_else(|| parse_cache_cost_from_json_body(data.as_bytes()));
            let normalized = state.normalize_data(&data);
            assembly.ingest_chunk(&normalized);
            if hold_usage_frame(&normalized, &mut pending_usage) {
                continue;
            }
            write_captured_sse_event(tcp_stream, &mut response_capture, None, &normalized).await?;
            observe_output_progress(route_observer, upstream_error_seen, &mut first_chunk_seen);
        }
        if processed > 0 {
            carry = carry[processed..].to_string();
        }

        if done_seen {
            break;
        }

        let mut chunk = [0u8; 8192];
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        let new_data = String::from_utf8_lossy(&chunk[..n]);
        carry.push_str(&new_data);
        if carry.contains('\r') {
            carry = carry.replace("\r\n", "\n");
        }
    }

    flush_pending_usage(tcp_stream, &mut response_capture, &mut pending_usage).await?;
    finish_normalized_chat_stream(
        tcp_stream,
        route_observer,
        (done_seen, upstream_error_seen),
        response_capture,
        &assembly,
        observed_usage,
        observed_cache_cost,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn frames(watermarks: bool) -> String {
        let mut events = vec![
            json!({"id":"c1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{"}}]},"finish_reason":null}]}),
        ];
        if watermarks {
            events.push(json!({"choices":[],"usage":{"prompt_tokens":0,"completion_tokens":1,"total_tokens":1}}));
        }
        events.push(json!({"id":"c1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"command\":\"ls\"}"}}]},"finish_reason":null}]}));
        if watermarks {
            events.push(json!({"choices":[],"usage":{"prompt_tokens":0,"completion_tokens":4,"total_tokens":4}}));
        }
        events.push(json!({"id":"c1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}));
        events.push(json!({"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":8}}}));
        events
            .into_iter()
            .map(|v| format!("data: {v}\n\n"))
            .collect::<String>()
            + "data: [DONE]\n\n"
    }

    async fn relay(body: String) -> (String, Result<RouteAttemptResult>) {
        let header =
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut client = ClientStream::from(socket);
            let mut reader = std::io::Cursor::new(body.into_bytes());
            relay_normalized_chat_completion_stream(
                &mut client,
                &mut reader,
                ResponseProbe {
                    buffered: header.to_vec(),
                    header_end: header.len(),
                    status_code: 200,
                    retryable_context_overflow: false,
                },
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });
        let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut bytes = Vec::new();
        socket.read_to_end(&mut bytes).await.unwrap();
        (String::from_utf8(bytes).unwrap(), task.await.unwrap())
    }

    #[tokio::test]
    async fn billing_watermarks_do_not_interrupt_tool_arguments() {
        let (paid, paid_result) = relay(frames(true)).await;
        let (free, free_result) = relay(frames(false)).await;
        assert_eq!(paid, free);
        assert_eq!(paid.matches("\"usage\"").count(), 1);
        let RouteAttemptResult::Delivered {
            usage,
            output_digests,
            ..
        } = paid_result.unwrap()
        else {
            panic!("paid stream not delivered")
        };
        let RouteAttemptResult::Delivered {
            usage: free_usage,
            output_digests: free_digests,
            ..
        } = free_result.unwrap()
        else {
            panic!("free stream not delivered")
        };
        assert_eq!(usage, free_usage);
        assert_eq!(usage.unwrap().completion_tokens, Some(4));
        assert_eq!(output_digests, free_digests);
        assert!(
            paid.find("\"finish_reason\":\"tool_calls\"").unwrap()
                < paid.find("\"usage\"").unwrap()
        );
        assert!(paid.find("\"usage\"").unwrap() < paid.find("[DONE]").unwrap());
    }

    #[tokio::test]
    async fn usage_precedes_upstream_error_without_hiding_error() {
        let usage = json!({"choices":[],"usage":{"completion_tokens":2}});
        let error = json!({"error":{"message":"generation failed","type":"backend_error"}});
        let (wire, _) = relay(format!(
            "data: {usage}\n\ndata: {error}\n\ndata: [DONE]\n\n"
        ))
        .await;
        assert!(
            wire.find("\"completion_tokens\":2").unwrap() < wire.find("generation failed").unwrap()
        );
    }

    #[tokio::test]
    async fn eof_flushes_usage_but_remains_truncated() {
        let body = frames(true).replace("data: [DONE]\n\n", "");
        let (wire, result) = relay(body).await;
        assert!(result.is_err());
        assert_eq!(wire.matches("\"usage\"").count(), 1);
        assert!(!wire.contains("[DONE]"));
    }

    #[cfg(feature = "payments")]
    #[tokio::test]
    async fn seller_replay_matches_payer_usage_and_digests() {
        let body = frames(true);
        let (_, result) = relay(body.clone()).await;
        let RouteAttemptResult::Delivered {
            usage,
            output_digests,
            ..
        } = result.unwrap()
        else {
            panic!("not delivered")
        };
        let raw = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
        );
        let seller = crate::network::openai::response::served_outcome_of_raw_response(
            raw.as_bytes(),
            crate::network::openai::request_normalize::ResponseAdapter::OpenAiChatCompletionsStream,
        )
        .await;
        let payer =
            crate::network::openai::transport::delivered_outcome(200, usage, output_digests);
        assert_eq!(seller, payer);
    }

    #[tokio::test]
    async fn totals_before_finish_are_preserved_after_finish() {
        let body = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16}}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";
        let (wire, result) = relay(body.to_string()).await;
        assert!(result.is_ok());
        assert_eq!(wire.matches("\"usage\"").count(), 1);
        assert!(wire.find("tool_calls").unwrap() < wire.find("\"usage\"").unwrap());
    }

    #[test]
    fn only_usage_only_frames_are_deferred() {
        let mut pending = None;
        assert!(hold_usage_frame(
            r#"{"choices":[],"usage":{"completion_tokens":1}}"#,
            &mut pending
        ));
        assert!(hold_usage_frame(
            r#"{"choices":[],"usage":{"completion_tokens":2}}"#,
            &mut pending
        ));
        assert!(pending.as_ref().unwrap().contains(":2"));
        for data in [
            r#"{"choices":[{"delta":{"content":"x"}}],"usage":{"completion_tokens":2}}"#,
            r#"{"choices":[],"usage":null}"#,
            "not-json",
        ] {
            assert!(!hold_usage_frame(data, &mut pending));
            assert!(pending.is_none());
        }
    }
}
