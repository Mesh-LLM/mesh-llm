//! Mesh relay for the shared Responses SSE state machine.
//!
//! The caller ingress owns the [`ResponsesTurn`]. Workers receive only the
//! expanded Chat Completions request, so a retry or a different worker cannot
//! lose conversation state and only this relay can commit the terminal turn.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use mesh_llm_events::logging::events::TokenUsage;
use mesh_llm_events::logging::identifiers::RequestId;
use serde_json::Value;
use skippy_inference_api::responses::{ResponsesStream, ResponsesStreamEvent, ResponsesTurn};
use tokio::io::{AsyncRead, AsyncWriteExt};

use super::body_reader::BodyReader;
use super::cache_cost::{CacheCostObservation, parse_cache_cost_from_json_body};
use super::common::{
    CommittedStreamFailureCause, ResponseRetryPolicy, RouteAttemptResult,
    parse_token_usage_from_json_body,
};
use super::probe::{
    ResponseProbe, append_capsule_nonce_headers, append_mesh_request_id_header,
    append_mesh_served_by_header, try_parse_response_headers,
};
use super::relay::relay_error_response;
use super::stream_translation::{StreamedChatAssembly, write_captured_sse_event};
use crate::logging::{OpenAiRouteObserver, OpenAiStreamArtifactCapture};
use crate::network::openai::client_stream::ClientStream;

pub(super) const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Default)]
struct Progress {
    done_seen: bool,
    first_chunk_seen: bool,
    failure: Option<CommittedStreamFailureCause>,
    observed_usage: Option<TokenUsage>,
    observed_cache_cost: Option<CacheCostObservation>,
    assembly: StreamedChatAssembly,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn relay<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
    request_id: RequestId,
    turn: Arc<ResponsesTurn>,
) -> Result<RouteAttemptResult> {
    if let Some(result) = super::common::retry_before_relay(&probe, retry_policy) {
        return Ok(result);
    }
    if !(200..300).contains(&probe.status_code) {
        route_observer.stream_error("upstream_status");
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
    write_response_head(tcp_stream, &parsed, served_by, request_id).await?;
    let mut capture = route_observer.begin_stream_response_capture();
    route_observer.stream_started(None);
    let mut machine = ResponsesStream::new(turn);
    write_events(tcp_stream, &mut capture, machine.start()).await?;

    let mut body = BodyReader::new(
        reader,
        probe.buffered[parsed.header_end..].to_vec(),
        parsed.chunked,
        parsed.content_length,
    );
    let mut carry = Vec::new();
    let mut progress = Progress::default();
    loop {
        process_available_frames(
            tcp_stream,
            &mut capture,
            &mut machine,
            &mut progress,
            &route_observer,
            &mut carry,
        )
        .await?;
        if progress.done_seen || progress.failure.is_some() {
            break;
        }
        let bytes = match body.next().await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => break,
            Err(_) => {
                write_events(
                    tcp_stream,
                    &mut capture,
                    machine.fail(
                        "upstream_stream_incomplete",
                        "upstream stream failed before its terminal marker",
                    ),
                )
                .await?;
                progress.failure = Some(CommittedStreamFailureCause::Truncated);
                break;
            }
        };
        carry.extend_from_slice(&bytes);
        normalize_crlf(&mut carry);
        if carry.len() > MAX_SSE_FRAME_BYTES && frame_end(&carry, 0).is_none() {
            write_events(
                tcp_stream,
                &mut capture,
                machine.fail(
                    "upstream_stream_frame_too_large",
                    "upstream stream event exceeded the frame limit",
                ),
            )
            .await?;
            progress.failure = Some(CommittedStreamFailureCause::BackendError);
        }
    }

    let terminal = if let Some(cause) = progress.failure {
        Some(cause)
    } else {
        let events = machine.finish();
        let failed = events
            .last()
            .is_some_and(|event| event.event == "response.failed");
        write_events(tcp_stream, &mut capture, events).await?;
        failed.then_some(CommittedStreamFailureCause::Truncated)
    };

    write_captured_sse_event(tcp_stream, &mut capture, Some("done"), "[DONE]").await?;
    let _ = tcp_stream.write_all(b"0\r\n\r\n").await;
    let _ = tcp_stream.shutdown().await;
    route_observer.complete_stream_response_capture(capture);

    if let Some(cause) = terminal {
        tcp_stream.record_exchange_outcome("backend_error");
        route_observer.stream_error(cause.reason());
        return Ok(RouteAttemptResult::CommittedStreamFailure {
            status_code: 200,
            cause,
        });
    }
    route_observer.stream_completed(progress.observed_usage);
    Ok(RouteAttemptResult::Delivered {
        status_code: 200,
        usage: progress.observed_usage,
        cache_cost: progress.observed_cache_cost,
        output_digests: progress.assembly.output_digests(),
    })
}

async fn write_response_head(
    tcp_stream: &mut ClientStream,
    parsed: &super::probe::ParsedResponseHeaders,
    served_by: Option<&str>,
    request_id: RequestId,
) -> Result<()> {
    let mut header = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n",
    );
    append_capsule_nonce_headers(
        &mut header,
        parsed.client_nonce.as_deref(),
        parsed.nonce_origin.as_deref(),
    );
    parsed.served_model.append_to(&mut header);
    append_mesh_served_by_header(&mut header, served_by);
    append_mesh_request_id_header(&mut header, request_id);
    header.push_str("Connection: close\r\n\r\n");
    tcp_stream.write_all(header.as_bytes()).await?;
    Ok(())
}

async fn process_available_frames(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    machine: &mut ResponsesStream,
    progress: &mut Progress,
    route_observer: &OpenAiRouteObserver<'_>,
    carry: &mut Vec<u8>,
) -> Result<()> {
    let mut processed = 0;
    while let Some(end) = frame_end(carry, processed) {
        if end.saturating_sub(processed) > MAX_SSE_FRAME_BYTES {
            write_events(
                tcp_stream,
                capture,
                machine.fail(
                    "upstream_stream_frame_too_large",
                    "upstream stream event exceeded the frame limit",
                ),
            )
            .await?;
            progress.failure = Some(CommittedStreamFailureCause::BackendError);
            processed = end + 2;
            break;
        }
        let frame = match std::str::from_utf8(&carry[processed..end]) {
            Ok(frame) => frame,
            Err(_) => {
                write_events(
                    tcp_stream,
                    capture,
                    machine.fail(
                        "upstream_stream_invalid",
                        "upstream emitted a non-UTF-8 stream event",
                    ),
                )
                .await?;
                progress.failure = Some(CommittedStreamFailureCause::BackendError);
                processed = end + 2;
                break;
            }
        };
        processed = end + 2;
        let data = sse_frame_data(frame);
        if data.is_empty() {
            continue;
        }
        process_frame(
            tcp_stream,
            capture,
            machine,
            progress,
            route_observer,
            &data,
        )
        .await?;
        if progress.done_seen || progress.failure.is_some() {
            break;
        }
    }
    if processed > 0 {
        carry.drain(..processed);
    }
    Ok(())
}

pub(super) fn frame_end(carry: &[u8], start: usize) -> Option<usize> {
    carry[start..]
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|relative| start + relative)
}

pub(super) fn normalize_crlf(bytes: &mut Vec<u8>) {
    if !bytes.contains(&b'\r') {
        return;
    }
    let mut read = 0;
    let mut write = 0;
    while read < bytes.len() {
        if bytes[read] == b'\r' && bytes.get(read + 1) == Some(&b'\n') {
            bytes[write] = b'\n';
            read += 2;
        } else {
            bytes[write] = bytes[read];
            read += 1;
        }
        write += 1;
    }
    bytes.truncate(write);
}

pub(super) fn sse_frame_data(frame: &str) -> String {
    frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n")
}

async fn process_frame(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    machine: &mut ResponsesStream,
    progress: &mut Progress,
    route_observer: &OpenAiRouteObserver<'_>,
    data: &str,
) -> Result<()> {
    if data == "[DONE]" {
        progress.done_seen = true;
        return Ok(());
    }
    let value: Value = match serde_json::from_str(data) {
        Ok(value) => value,
        Err(_) => {
            write_events(
                tcp_stream,
                capture,
                machine.fail(
                    "upstream_stream_invalid",
                    "upstream emitted an invalid stream event",
                ),
            )
            .await?;
            progress.failure = Some(CommittedStreamFailureCause::BackendError);
            return Ok(());
        }
    };
    if let Some(error) = value.get("error") {
        let code = error
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or("server_error");
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| error.as_str())
            .unwrap_or("upstream stream failed");
        write_events(tcp_stream, capture, machine.fail(code, message)).await?;
        progress.failure = Some(CommittedStreamFailureCause::BackendError);
        return Ok(());
    }
    progress.assembly.ingest_chunk(data);
    progress.observed_usage =
        parse_token_usage_from_json_body(data.as_bytes()).or(progress.observed_usage);
    progress.observed_cache_cost = progress
        .observed_cache_cost
        .or_else(|| parse_cache_cost_from_json_body(data.as_bytes()));
    let events = machine.push_chunk(&value);
    let failed = machine.is_failed();
    write_events(tcp_stream, capture, events).await?;
    if failed {
        progress.failure = Some(CommittedStreamFailureCause::BackendError);
        return Ok(());
    }
    if progress.first_chunk_seen {
        route_observer.stream_chunk();
    } else {
        route_observer.stream_first_token();
        progress.first_chunk_seen = true;
    }
    Ok(())
}

async fn write_events(
    tcp_stream: &mut ClientStream,
    capture: &mut Option<OpenAiStreamArtifactCapture>,
    events: Vec<ResponsesStreamEvent>,
) -> Result<()> {
    for event in events {
        let data =
            serde_json::to_string(&event.data).context("serialize Responses stream event")?;
        write_captured_sse_event(tcp_stream, capture, Some(event.event), &data).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_inference_api::responses::{CallerScope, ResponseContinuation, ResponseStore};
    use skippy_inference_api::wire_bytes::{WireBytesCommitment, WireBytesObserver};
    use std::sync::Mutex;

    #[derive(Default)]
    struct WireCapture(Mutex<Vec<u8>>);

    impl WireBytesObserver for WireCapture {
        fn response_status(&self, _status: u16) {}

        fn try_chunk(&self, offset: u64, bytes: &[u8]) -> bool {
            let mut captured = self.0.lock().unwrap();
            assert_eq!(offset, captured.len() as u64);
            captured.extend_from_slice(bytes);
            true
        }

        fn finish(&self, _commitment: WireBytesCommitment) {}
    }

    fn turn(store: &Arc<ResponseStore>) -> Arc<ResponsesTurn> {
        let mut request = serde_json::json!({
            "model": "m",
            "input": "weather",
            "tools": [{"type": "function", "name": "weather"}],
        });
        Arc::new(
            ResponsesTurn::prepare(
                &mut request,
                ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost)),
            )
            .unwrap(),
        )
    }

    async fn run(upstream: &str, turn: Arc<ResponsesTurn>) -> (String, RouteAttemptResult) {
        run_with_framing(upstream, turn, false).await
    }

    async fn run_with_framing(
        upstream: &str,
        turn: Arc<ResponsesTurn>,
        chunked: bool,
    ) -> (String, RouteAttemptResult) {
        run_with_framing_bytes(upstream.as_bytes(), turn, chunked).await
    }

    async fn run_with_framing_bytes(
        upstream: &[u8],
        turn: Arc<ResponsesTurn>,
        chunked: bool,
    ) -> (String, RouteAttemptResult) {
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        upstream_writer.write_all(upstream).await.unwrap();
        upstream_writer.shutdown().await.unwrap();
        let capture = Arc::new(WireCapture::default());
        let mut client = ClientStream::null().with_wire_bytes_observer(capture.clone());
        let header = if chunked {
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".as_slice()
        } else {
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
                .as_slice()
        };
        let result = relay(
            &mut client,
            &mut upstream_reader,
            ResponseProbe {
                buffered: header.to_vec(),
                header_end: header.len(),
                status_code: 200,
                retryable_context_overflow: false,
            },
            ResponseRetryPolicy::next_target_available(false),
            None,
            OpenAiRouteObserver::default(),
            RequestId::new(),
            turn,
        )
        .await
        .unwrap();
        let output = capture.0.lock().unwrap().clone();
        (String::from_utf8(output).unwrap(), result)
    }

    #[tokio::test]
    async fn shared_relay_streams_tools_and_commits_only_after_terminal_done() {
        let store = Arc::new(ResponseStore::default());
        let turn = turn(&store);
        let id = turn.id().to_string();
        let upstream = concat!(
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"reasoning_content\":\"check\"}}]}\n\n",
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n",
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Oslo\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3,\"total_tokens\":7}}\n\n",
            "data: [DONE]\n\n",
        );
        let (wire, result) = run(upstream, turn).await;
        assert!(matches!(result, RouteAttemptResult::Delivered { .. }));
        for expected in [
            "event: response.created",
            "event: response.reasoning.delta",
            "event: response.function_call_arguments.delta",
            "event: response.function_call_arguments.done",
            "event: response.completed",
            "data: [DONE]",
            r#""cached_tokens":0"#,
            r#""reasoning_tokens":0"#,
        ] {
            assert!(wire.contains(expected), "missing {expected}: {wire}");
        }
        let stored = store.get(&CallerScope::LocalHost, &id).unwrap();
        assert_eq!(stored.output[1]["call_id"], "call_a");
        assert_eq!(stored.output[1]["arguments"], "{\"city\":\"Oslo\"}");
    }

    #[tokio::test]
    async fn shared_mesh_relay_quarantines_fragmented_unauthorized_name() {
        let store = Arc::new(ResponseStore::default());
        let mut request = serde_json::json!({
            "model":"m", "input":"safe only",
            "tools":[
                {"type":"function","name":"safe"},
                {"type":"function","name":"safe_admin"}
            ],
            "tool_choice":{"type":"allowed_tools","tools":[
                {"type":"function","name":"safe"}
            ]}
        });
        let turn = Arc::new(
            ResponsesTurn::prepare(
                &mut request,
                ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost)),
            )
            .unwrap(),
        );
        let id = turn.id().to_string();
        let upstream = concat!(
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"safe\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"_admin\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (wire, result) = run(upstream, turn).await;
        assert!(matches!(
            result,
            RouteAttemptResult::CommittedStreamFailure { .. }
        ));
        assert!(wire.contains("event: response.failed"), "{wire}");
        assert!(wire.contains("tool_choice_violation"), "{wire}");
        assert!(
            !wire.contains("event: response.output_item.added"),
            "{wire}"
        );
        assert!(store.get(&CallerScope::LocalHost, &id).is_err());

        let sequence_numbers: Vec<u64> = wire
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter_map(|event| event["sequence_number"].as_u64())
            .collect();
        assert_eq!(
            sequence_numbers,
            (0..sequence_numbers.len() as u64).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn shared_mesh_relay_quarantines_repeated_calls_over_max_tool_calls() {
        let store = Arc::new(ResponseStore::default());
        let mut request = serde_json::json!({
            "model":"m", "input":"one call only", "max_tool_calls":1,
            "tools":[{"type":"function","name":"same"}]
        });
        let turn = Arc::new(
            ResponsesTurn::prepare(
                &mut request,
                ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost)),
            )
            .unwrap(),
        );
        let id = turn.id().to_string();
        let upstream = concat!(
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"same\",\"arguments\":\"{}\"}},{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"same\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (wire, result) = run(upstream, turn).await;
        assert!(matches!(
            result,
            RouteAttemptResult::CommittedStreamFailure { .. }
        ));
        assert!(wire.contains("event: response.failed"), "{wire}");
        assert!(wire.contains("tool_call_limit_exceeded"), "{wire}");
        assert!(
            !wire.contains("event: response.output_item.added"),
            "{wire}"
        );
        assert!(
            !wire.contains("event: response.function_call_arguments"),
            "{wire}"
        );
        assert!(store.get(&CallerScope::LocalHost, &id).is_err());

        let sequence_numbers = wire
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter_map(|event| event["sequence_number"].as_u64())
            .collect::<Vec<_>>();
        assert_eq!(
            sequence_numbers,
            (0..sequence_numbers.len() as u64).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn premature_upstream_eof_emits_failed_terminal_and_retains_nothing() {
        let store = Arc::new(ResponseStore::default());
        let turn = turn(&store);
        let id = turn.id().to_string();
        let upstream =
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
        let (wire, result) = run(upstream, turn).await;
        assert!(matches!(
            result,
            RouteAttemptResult::CommittedStreamFailure {
                cause: CommittedStreamFailureCause::Truncated,
                ..
            }
        ));
        assert!(wire.contains("event: response.failed"), "{wire}");
        assert!(wire.contains("upstream_stream_incomplete"), "{wire}");
        assert!(store.get(&CallerScope::LocalHost, &id).is_err());
    }

    #[tokio::test]
    async fn eof_after_finish_reason_succeeds_without_done_sentinel() {
        let store = Arc::new(ResponseStore::default());
        let turn = turn(&store);
        let id = turn.id().to_string();
        let upstream = concat!(
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        );
        let (wire, result) = run(upstream, turn).await;
        assert!(matches!(result, RouteAttemptResult::Delivered { .. }));
        assert!(wire.contains("event: response.completed"), "{wire}");
        assert!(store.get(&CallerScope::LocalHost, &id).is_ok());
    }

    #[tokio::test]
    async fn chunk_boundaries_inside_json_are_decoded_before_sse_parsing() {
        let store = Arc::new(ResponseStore::default());
        let turn = turn(&store);
        let pieces = [
            "data: {\"model\":\"m\",\"cho",
            "ices\":[{\"delta\":{\"content\":\"ok",
            "\"},\"finish_reason\":\"stop\"}]}\n\n",
        ];
        let mut chunked = String::new();
        for piece in pieces {
            chunked.push_str(&format!("{:x}\r\n{piece}\r\n", piece.len()));
        }
        chunked.push_str("0\r\n\r\n");
        let (wire, result) = run_with_framing(&chunked, turn, true).await;
        assert!(matches!(result, RouteAttemptResult::Delivered { .. }));
        assert!(wire.contains("event: response.completed"), "{wire}");
        assert!(!wire.contains("response.failed"), "{wire}");
    }

    #[tokio::test]
    async fn chunk_boundary_inside_utf8_is_retained_until_the_frame_is_complete() {
        let store = Arc::new(ResponseStore::default());
        let turn = turn(&store);
        let payload = "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"café\"},\"finish_reason\":\"stop\"}]}\n\n";
        let split = payload
            .as_bytes()
            .iter()
            .position(|byte| *byte == 0xc3)
            .unwrap()
            + 1;
        let mut chunked = Vec::new();
        for piece in [&payload.as_bytes()[..split], &payload.as_bytes()[split..]] {
            chunked.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
            chunked.extend_from_slice(piece);
            chunked.extend_from_slice(b"\r\n");
        }
        chunked.extend_from_slice(b"0\r\n\r\n");
        let (wire, result) = run_with_framing_bytes(&chunked, turn, true).await;
        assert!(matches!(result, RouteAttemptResult::Delivered { .. }));
        assert!(wire.contains("café"), "{wire}");
        assert!(!wire.contains("response.failed"), "{wire}");
    }

    #[tokio::test]
    async fn oversized_sse_frame_fails_without_unbounded_carry() {
        let store = Arc::new(ResponseStore::default());
        let mut machine = ResponsesStream::new(turn(&store));
        let capture = Arc::new(WireCapture::default());
        let mut client = ClientStream::null().with_wire_bytes_observer(capture);
        let mut response_capture = None;
        let mut progress = Progress::default();
        let mut carry = format!("data: {}\n\n", "x".repeat(MAX_SSE_FRAME_BYTES + 1)).into_bytes();
        let observer = OpenAiRouteObserver::default();
        process_available_frames(
            &mut client,
            &mut response_capture,
            &mut machine,
            &mut progress,
            &observer,
            &mut carry,
        )
        .await
        .unwrap();
        assert!(machine.is_failed());
        assert!(matches!(
            progress.failure,
            Some(CommittedStreamFailureCause::BackendError)
        ));
    }
}
