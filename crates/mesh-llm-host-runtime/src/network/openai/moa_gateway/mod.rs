//! Mesh-wide MoA orchestration entrypoint.
//!
//! Any node that receives a chat-completion request with `model: "mesh"`
//! runs MoA orchestration here, regardless of whether that node is serving
//! models locally. The worker pool is built from gossip — every model
//! advertised by any peer (or hosted locally) is a candidate.
//!
//! Both the host's `api_proxy` and the passive `handle_mesh_request` path
//! call `try_handle_moa`. On a pure client node, all backends are remote;
//! on a serving host, the local model is wired directly to its skippy port
//! and the rest go over QUIC.

use self::streaming::write_moa_response;
use self::workers::effective_enable_thinking_for_moa;
use crate::inference::election;
use crate::mesh;
use crate::network::openai::transport as proxy;
use mesh_mixture_of_agents as moa;
use tokio::net::TcpStream;

pub use self::workers::build_moa_config;

/// Detect `model: "mesh"`, build a mesh-wide MoA config, run the turn,
/// and write the HTTP response (JSON or SSE) directly to the stream.
///
/// Return value carries the un-consumed `TcpStream` so the caller knows
/// what to do next:
///
/// * `Some(stream)` — the request is *not* MoA-shaped (effective model
///   is not the virtual `"mesh"` name). The stream is returned unused
///   and the caller should fall through to normal routing.
///
/// * `None` — MoA owns the response. The stream has been consumed: a
///   successful MoA response, a 503 (when fewer than 2 models are
///   reachable), or a 400 (when the request body wasn't JSON) was
///   already written. The caller must *not* attempt to respond again.
pub async fn try_handle_moa(
    node: &mesh::Node,
    tcp_stream: TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    effective_model: Option<&str>,
    targets: Option<&election::ModelTargets>,
    required_tokens: Option<u32>,
) -> Option<TcpStream> {
    if effective_model != Some(moa::VIRTUAL_MODEL_NAME) {
        return Some(tcp_stream);
    }

    request.ensure_body_json();
    let Some(body_json) = request.body_json.clone() else {
        let _ = proxy::send_400(tcp_stream, "MoA requires a JSON body").await;
        return None;
    };

    let enable_thinking = effective_enable_thinking_for_moa(&body_json);

    let Some(mut config) = build_moa_config(node, targets, required_tokens).await else {
        let _ = proxy::send_503(tcp_stream, "MoA requires ≥2 models available in the mesh").await;
        return None;
    };
    config.enable_thinking = enable_thinking;

    run_moa_turn(tcp_stream, body_json, &config, request.response_adapter).await;
    None
}

pub(in crate::network::openai) mod context_selection;
mod progress;
mod streaming;
mod workers;

/// Run a turn through the gateway and write the response with x-moa-* headers.
///
/// Streaming MoA turns are handed off to [`progress::run_moa_turn_with_progress`],
/// which sends HTTP headers immediately and drips `reasoning_content` /
/// `response.reasoning_text.delta` heartbeats into the thinking pane
/// while the arbiter waits; non-streaming turns and the synchronous SSE
/// path stay here so the post-hoc `x-moa-*` observability headers can
/// be derived from the finished `TurnResult`.
/// Caller has already validated the request and built the config.
async fn run_moa_turn(
    tcp_stream: TcpStream,
    body_json: serde_json::Value,
    config: &moa::GatewayConfig,
    response_adapter: proxy::ResponseAdapter,
) {
    let was_streaming = body_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mut moa_body = body_json;
    moa_body.as_object_mut().map(|o| o.remove("stream"));

    // Streaming MoA: the arbiter takes ~3s before any content can be
    // emitted. Send response headers immediately and drip progress
    // text into `reasoning_content` so the chat UI's "thinking" pane
    // shows live activity instead of a stalled spinner.
    //
    // Trade-off: HTTP headers must precede the body, so this path
    // loses the post-hoc `x-moa-*` observability headers (the
    // result-derived ones). Worth it for the live feel.
    if was_streaming
        && matches!(
            response_adapter,
            proxy::ResponseAdapter::None
                | proxy::ResponseAdapter::OpenAiChatCompletionsStream
                | proxy::ResponseAdapter::OpenAiResponsesStream
        )
    {
        progress::run_moa_turn_with_progress(tcp_stream, moa_body, config, response_adapter).await;
        return;
    }

    let moa_result = moa::handle_turn(config, &moa_body).await;
    let extra_headers = build_moa_headers(&moa_result);
    write_moa_response(
        tcp_stream,
        &moa_result,
        &extra_headers,
        was_streaming,
        response_adapter,
    )
    .await;
}

/// Build the `x-moa-*` observability headers from a finished turn and log
/// a one-line summary.
fn build_moa_headers(result: &moa::TurnResult) -> Vec<(&'static str, String)> {
    let workers_ok = result
        .worker_summaries
        .iter()
        .filter(|w| w.succeeded)
        .count();
    let workers_total = result.worker_summaries.len();
    tracing::info!(
        "moa: {}ms, {}/{} workers, kind={}, reducer={} (attempts={})",
        result.elapsed_ms,
        workers_ok,
        workers_total,
        result.turn_kind.label(),
        result.reducer_used,
        result.reducer_attempts,
    );

    vec![
        ("x-moa-elapsed-ms", result.elapsed_ms.to_string()),
        ("x-moa-turn", result.turn_kind.label().to_string()),
        ("x-moa-workers", workers_total.to_string()),
        ("x-moa-workers-ok", workers_ok.to_string()),
        ("x-moa-reducer", result.reducer_used.to_string()),
        (
            "x-moa-reducer-attempts",
            result.reducer_attempts.to_string(),
        ),
    ]
}
