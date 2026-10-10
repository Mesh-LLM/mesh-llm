use super::*;

#[test]
fn nullable_tool_calls_do_not_discard_generated_content() {
    let mut stream = Stream::default();
    stream
        .consume(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\",\"tool_calls\":null}}]}\n",
            Duration::from_secs(1),
        )
        .unwrap();
    stream.consume(b"data: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n", Duration::from_secs(2)).unwrap();
    let evidence = stream.finish(Duration::from_secs(2), false).unwrap();
    assert_eq!(evidence.content_events, 1);
}

const SUCCESS: &[u8] = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":30}}}\n\ndata: [DONE]\n\n";

#[test]
fn arbitrary_frame_boundaries_preserve_usage_and_terminal_marker() {
    let mut stream = Stream::default();
    for chunk in SUCCESS.chunks(3) {
        stream.consume(chunk, Duration::from_secs(1)).unwrap();
    }
    let evidence = stream.finish(Duration::from_secs(2), false).unwrap();
    assert_eq!(evidence.prompt_tokens, 40);
    assert_eq!(evidence.cached_tokens, 30);
    assert_eq!(evidence.completion_tokens, 2);
    assert_eq!(evidence.content_events, 1);
    assert_eq!(evidence.cache_pct, 75.0);
}

#[test]
fn missing_terminal_marker_and_server_error_fail_closed() {
    let mut stream = Stream::default();
    stream
        .consume(&SUCCESS[..SUCCESS.len() - 14], Duration::from_secs(1))
        .unwrap();
    assert!(stream.finish(Duration::from_secs(2), false).is_err());
    let mut stream = Stream::default();
    stream
        .consume(
            b"data: {\"error\":{\"message\":\"cache timeout\"}}\n\ndata: [DONE]\n",
            Duration::ZERO,
        )
        .unwrap();
    assert!(
        stream
            .finish(Duration::from_secs(1), true)
            .unwrap_err()
            .contains("cache timeout")
    );
}

#[test]
fn first_generated_identity_uses_typed_content_reasoning_and_tool_delta() {
    for delta in [
        serde_json::json!({"content":"first"}),
        serde_json::json!({"reasoning_content":"first"}),
        serde_json::json!({"tool_calls":[{"index":0,"type":"function","function":{"name":"f","arguments":"{}"}}]}),
    ] {
        let decoded: Delta = serde_json::from_value(delta.clone()).unwrap();
        let expected = hex::encode(Sha256::digest(serde_json::to_vec(&decoded).unwrap()));
        let mut stream = Stream::default();
        stream
            .consume(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n",
                Duration::ZERO,
            )
            .unwrap();
        let event = format!(
            "data: {}\n",
            serde_json::json!({"choices":[{"delta":delta}]})
        );
        stream
            .consume(event.as_bytes(), Duration::from_secs(1))
            .unwrap();
        stream.consume(SUCCESS, Duration::from_secs(2)).unwrap();
        let evidence = stream.finish(Duration::from_secs(3), false).unwrap();
        assert_eq!(
            evidence.first_generated_sha256.as_deref(),
            Some(expected.as_str())
        );
        assert_eq!(evidence.ttft_seconds, 1.0);
        assert_eq!(evidence.content_sha256.len(), 64);
    }
}

#[test]
fn first_delta_identity_and_full_output_identity_remain_distinct() {
    let mut first = Stream::default();
    first.consume(SUCCESS, Duration::from_secs(1)).unwrap();
    let first = first.finish(Duration::from_secs(2), false).unwrap();
    let mut split = Stream::default();
    for content in ["h", "i"] {
        split
            .consume(
                format!(
                    "data: {}\n",
                    serde_json::json!({"choices":[{"delta":{"content":content}}]})
                )
                .as_bytes(),
                Duration::from_secs(1),
            )
            .unwrap();
    }
    split.consume(b"data: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":30}}}\n\ndata: [DONE]\n", Duration::from_secs(2)).unwrap();
    let split = split.finish(Duration::from_secs(2), false).unwrap();
    assert_eq!(first.content_sha256, split.content_sha256);
    assert_ne!(first.first_generated_sha256, split.first_generated_sha256);
}
