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
