use super::*;

#[test]
fn historical_and_decode_only_rates_preserve_distinct_measured_intervals() {
    assert_eq!(decode_rate(Some(100), Some(2000.0)), Some(50.0));
    assert_eq!(
        decode_only(Some(100), Some(5000.0), Some(500.0)),
        Some(100.0 / 4.5)
    );
    assert_eq!(decode_only(Some(1), Some(1e-6), Some(0.0)), Some(1e6));
}
#[test]
fn failed_missing_nonfinite_and_nonpositive_intervals_remain_null() {
    for elapsed in [
        None,
        Some(0.0),
        Some(-1.0),
        Some(f64::NAN),
        Some(f64::INFINITY),
    ] {
        assert_eq!(decode_rate(Some(1), elapsed), None);
    }
    for (tokens, elapsed, ttft) in [
        (None, Some(1.0), Some(0.0)),
        (Some(1), None, Some(0.0)),
        (Some(1), Some(1.0), None),
        (Some(1), Some(1.0), Some(1.0)),
        (Some(1), Some(1.0), Some(2.0)),
        (Some(1), Some(1.0), Some(-1.0)),
    ] {
        assert_eq!(decode_only(tokens, elapsed, ttft), None);
    }
}
#[test]
fn fragmented_sse_uses_first_nonempty_content_and_minimal_terminal_usage() {
    let mut stream = Stream::default();
    stream
        .consume(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n",
            Duration::from_millis(10),
        )
        .unwrap();
    stream
        .consume(
            b"data: {\"choices\":[{\"delta\":{\"con",
            Duration::from_millis(20),
        )
        .unwrap();
    stream
        .consume(
            b"tent\":\"hello\"}}]}\ndata: {\"usage\":{\"completion_tokens\":7}}\ndata: [DONE]\n",
            Duration::from_millis(250),
        )
        .unwrap();
    let result = stream.finish(Duration::from_millis(500));
    assert_eq!(result.completion_tokens, Some(7));
    assert_eq!(result.ttft_ms, Some(250.0));
    assert_eq!(result.decode_tok_s, Some(14.0));
    assert_eq!(result.decode_only_tok_s, Some(28.0));
    assert!(!result.malformed);
}
#[test]
fn malformed_chunks_comments_and_empty_deltas_do_not_fabricate_ttft() {
    let mut stream = Stream::default();
    stream.consume(b":keepalive\ndata: {broken\ndata: []\ndata: {\"choices\":[{\"delta\":{}}]}\ndata: {\"usage\":{\"completion_tokens\":2}}\n",Duration::from_millis(10)).unwrap();
    let result = stream.finish(Duration::from_millis(20));
    assert_eq!(result.completion_tokens, Some(2));
    assert_eq!(result.ttft_ms, None);
    assert_eq!(result.decode_only_tok_s, None);
}
#[test]
fn missing_usage_or_server_error_remains_failed_without_zero_metrics() {
    for bytes in [b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\ndata: [DONE]\n".as_slice(),b"data: {\"error\":{\"message\":\"failed\"}}\ndata: {\"usage\":{\"completion_tokens\":2}}\n"] {let mut stream=Stream::default();stream.consume(bytes,Duration::from_millis(10)).unwrap();let result=stream.finish(Duration::from_millis(20));assert!(result.malformed);assert_eq!(result.completion_tokens,None);assert_eq!(result.decode_tok_s,None);assert_eq!(result.ttft_ms,None);}
}
#[test]
fn unterminated_usage_line_is_consumed_at_eof_and_byte_limits_refuse_oversize() {
    let mut stream = Stream::default();
    stream
        .consume(
            b"data: {\"usage\":{\"completion_tokens\":1}}",
            Duration::from_millis(1),
        )
        .unwrap();
    assert_eq!(
        stream.finish(Duration::from_millis(10)).completion_tokens,
        Some(1)
    );
    assert!(
        Stream::default()
            .consume(&vec![b'x'; MAX_LINE + 1], Duration::ZERO)
            .is_err()
    );
    assert!(
        Stream::default()
            .consume(&vec![b'x'; MAX_BYTES + 1], Duration::ZERO)
            .is_err()
    );
}
#[test]
fn model_resolution_and_request_body_use_the_real_advertised_id() {
    let value = json!({"data":[{"id":"local-gguf/sha256-abc"}]});
    let id = first_model(&value).unwrap();
    let body = chat_body("prompt", 16, id).unwrap();
    assert_eq!(body["model"], id);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["temperature"], 0.0);
    for value in [
        json!({}),
        json!({"data":[]}),
        json!({"data":"bad"}),
        json!({"data":{"0":{"id":"fake"}}}),
        json!({"data":[{"id":""}]}),
    ] {
        assert_eq!(first_model(&value), None);
    }
}
