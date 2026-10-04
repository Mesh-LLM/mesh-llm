//! Terminal emission and outcome recording for normalized chat SSE.
use super::{
    CacheCostObservation, RouteAttemptResult, StreamedChatAssembly, normalized_stream_is_truncated,
};
use crate::logging::{OpenAiRouteObserver, OpenAiStreamArtifactCapture};
use crate::network::openai::client_stream::ClientStream;
use anyhow::{Result, anyhow};
use mesh_llm_events::logging::events::TokenUsage;
use tokio::io::AsyncWriteExt;

pub(super) async fn finish_normalized_chat_stream(
    tcp_stream: &mut ClientStream,
    route_observer: OpenAiRouteObserver<'_>,
    progress: (bool, bool),
    response_capture: Option<OpenAiStreamArtifactCapture>,
    assembly: &StreamedChatAssembly,
    observed_usage: Option<TokenUsage>,
    observed_cache_cost: Option<CacheCostObservation>,
) -> Result<RouteAttemptResult> {
    let (done_seen, upstream_error_seen) = progress;
    if upstream_error_seen {
        tcp_stream.record_exchange_outcome("backend_error");
    }
    if normalized_stream_is_truncated(done_seen, upstream_error_seen) {
        tcp_stream
            .finish_wire_bytes(openai_frontend::wire_bytes::WireBytesIncomplete::TransportError);
    }
    let _ = tcp_stream.write_all(b"0\r\n\r\n").await;
    let _ = tcp_stream.shutdown().await;
    if upstream_error_seen {
        route_observer.stream_error("upstream_stream_error");
        return Ok(RouteAttemptResult::Delivered {
            status_code: 200,
            usage: None,
            cache_cost: None,
            output_digests: Default::default(),
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
        output_digests: assembly.output_digests(),
    })
}
