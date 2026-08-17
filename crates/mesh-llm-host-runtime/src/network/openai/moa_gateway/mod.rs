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
use crate::logging::OpenAiRouteObserver;
use crate::mesh;
use crate::network::openai::automatic;
use crate::network::openai::transport as proxy;
use mesh_llm_events::logging::events::TokenUsage;
use mesh_mixture_of_agents as moa;
use tokio::net::TcpStream;

pub use self::workers::build_moa_config;

pub(crate) enum MoaDispatchResult {
    Passthrough(TcpStream),
    Responded(u16),
    RespondedWithUsage {
        status_code: u16,
        usage: TokenUsage,
    },
    FailedWithStatus {
        status_code: u16,
        reason: &'static str,
    },
    Dropped(&'static str),
}

/// Fall back to serving a single real model when MoA cannot form a committee.
///
/// Picks any model advertised in the mesh (local or peer), rewrites the
/// request's `model` from the virtual `"mesh"` name to it, and hands the stream
/// back so the caller routes it as an ordinary single-model request. Returns
/// `None` (503) only when the node genuinely has no model at all.
async fn degrade_to_single_model(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    tcp_stream: TcpStream,
    request: &mut proxy::BufferedHttpRequest,
    route_observer: OpenAiRouteObserver<'_>,
) -> MoaDispatchResult {
    // Prefer the same source `/v1/models` and routing use — the local
    // targets table (`callable_models`) — since `models_being_served()` can be
    // empty at request time on a fresh serve node. Fall back to the gossiped
    // served set for a pure client node that has no local targets.
    // Try each source /v1/models draws from, cheapest-first: the local targets
    // table (`callable_models`), the gossiped served set, then the node's own
    // `serving_models` — the last is what a fresh serve node populates first
    // (the others can lag at request time).
    let mut candidates = targets
        .map(super::ingress::callable_models)
        .unwrap_or_default();
    if candidates.is_empty() {
        candidates = node.models_being_served().await;
    }
    if candidates.is_empty() {
        candidates = node.serving_models().await;
    }
    let Some(target) = candidates
        .into_iter()
        .find(|m| m != moa::VIRTUAL_MODEL_NAME)
    else {
        return match proxy::send_503_observed(
            tcp_stream,
            "no models available in the mesh",
            route_observer,
        )
        .await
        {
            Ok(()) => MoaDispatchResult::Responded(503),
            Err(_) => MoaDispatchResult::Dropped("moa_response_write_failed"),
        };
    };

    tracing::info!("MoA: <2 workers, degrading model=mesh to single model {target}");

    // Rewrite every surface the downstream router reads. The forwarded request
    // is driven by `request.raw` (the raw HTTP bytes), so `rewrite_model_field`
    // patches raw + body + Content-Length together — rewriting only
    // `model_name`/`body_json` left `raw` saying "mesh", so the embedded
    // frontend still saw the virtual model and 404'd.
    proxy::rewrite_model_field(request, &target);
    request.model_name = Some(target);

    // Hand the stream back: the caller falls through to normal routing.
    MoaDispatchResult::Passthrough(tcp_stream)
}

/// Whether a request that named the automatic directive may be served by a
/// committee.
enum CommitteeAdmission {
    /// Convene a committee.
    Admitted,
    /// Automatic routing applies, but a committee cannot honour this request.
    ServeFromSingleModel(automatic::SingleModelReason),
    /// The request is chat-shaped but malformed; reject it with this message.
    Reject(&'static str),
}

/// Decide whether to convene a committee for an automatic chat request.
///
/// The three checks are ordered deliberately, and the order is the behaviour:
///
/// 1. **Envelope** — could a committee ever fan this out? A committee builds
///    chat completions from a `messages` array, so a `/v1/completions` request
///    has nothing to fan out and must not be held to the chat contract below.
/// 2. **Chat contract** — `messages` must be a present, non-empty array.
///    Without this a request like `{"model":"mesh"}` or a string `messages`
///    field reached the workers, which fabricated a 200 answer from nothing
///    (homelab API-1 defect). This must run *before* the content check so a
///    malformed chat body gets this precise error rather than being diverted
///    into single-model routing and failing downstream with a vaguer one.
/// 3. **Content** — what did the client declare? Media has no aggregation
///    semantics for a committee, and a committee cannot stream for real.
fn committee_admission(
    effective_model: Option<&str>,
    forwarded_path: &str,
    body: &serde_json::Value,
) -> CommitteeAdmission {
    let request = automatic::AutomaticRequest {
        model: effective_model,
        // The forwarded path: `/v1/responses` is normalised onto chat
        // completions before routing, so it stays committee-eligible.
        path: forwarded_path,
        body,
    };
    if let automatic::ServingMode::SingleModel(reason) = automatic::envelope_mode(request) {
        return CommitteeAdmission::ServeFromSingleModel(reason);
    }
    match body.get("messages") {
        Some(serde_json::Value::Array(messages)) if !messages.is_empty() => {}
        _ => return CommitteeAdmission::Reject("MoA requires a non-empty `messages` array"),
    }
    if let automatic::ServingMode::SingleModel(reason) = automatic::content_mode(body) {
        return CommitteeAdmission::ServeFromSingleModel(reason);
    }
    CommitteeAdmission::Admitted
}

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
    route_observer: OpenAiRouteObserver<'_>,
) -> MoaDispatchResult {
    if !effective_model.is_some_and(automatic::is_directive) {
        return MoaDispatchResult::Passthrough(tcp_stream);
    }

    request.ensure_body_json();
    let Some(body_json) = request.body_json.clone() else {
        return match proxy::send_400_observed(
            tcp_stream,
            "MoA requires a JSON body",
            route_observer,
        )
        .await
        {
            Ok(()) => MoaDispatchResult::Responded(400),
            Err(_) => MoaDispatchResult::Dropped("moa_response_write_failed"),
        };
    };

    match committee_admission(effective_model, &request.path, &body_json) {
        CommitteeAdmission::Admitted => {}
        CommitteeAdmission::ServeFromSingleModel(reason) => {
            tracing::info!(
                reason = reason.as_str(),
                "MoA: serving {} from a single model",
                automatic::DIRECTIVE
            );
            // The directive stays in the request *unrewritten*. The caller
            // recognises it as automatic and runs the full selector (media
            // capability filter, readiness, affinity, context fit). Rewriting
            // to a concrete model here would make the request look explicitly
            // addressed and skip exactly the media filter a media request
            // needs.
            return MoaDispatchResult::Passthrough(tcp_stream);
        }
        CommitteeAdmission::Reject(message) => {
            return match proxy::send_400_observed(tcp_stream, message, route_observer).await {
                Ok(()) => MoaDispatchResult::Responded(400),
                Err(_) => MoaDispatchResult::Dropped("moa_response_write_failed"),
            };
        }
    }

    let enable_thinking = effective_enable_thinking_for_moa(&body_json);

    let Some(mut config) = build_moa_config(node, targets, required_tokens).await else {
        // Graceful degradation: MoA needs ≥2 workers, but a lone node (or a
        // mesh with a single model) should still answer a `model=mesh`
        // request rather than 503. Rewrite the virtual model to a real served
        // model and fall through to normal single-model routing by handing the
        // stream back. `mesh` thus works everywhere: passthrough on one node,
        // committee once a second worker joins.
        return degrade_to_single_model(node, targets, tcp_stream, request, route_observer).await;
    };
    config.enable_thinking = enable_thinking;

    run_moa_turn(
        tcp_stream,
        body_json,
        &config,
        request.response_adapter,
        route_observer,
    )
    .await
}

pub(in crate::network::openai) mod context_selection;
mod pool;
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
    route_observer: OpenAiRouteObserver<'_>,
) -> MoaDispatchResult {
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
        return progress::run_moa_turn_with_progress(
            tcp_stream,
            moa_body,
            config,
            response_adapter,
        )
        .await;
    }

    let moa_result = moa::handle_turn(config, &moa_body).await;
    let usage = moa_token_usage(&moa_result);
    let extra_headers = build_moa_headers(&moa_result);
    match write_moa_response(
        tcp_stream,
        &moa_result,
        &extra_headers,
        was_streaming,
        response_adapter,
        route_observer,
    )
    .await
    {
        Ok(status) => completed_moa_response(status, usage),
        Err(_) => MoaDispatchResult::Dropped("moa_response_write_failed"),
    }
}

fn completed_moa_response(status_code: u16, usage: Option<TokenUsage>) -> MoaDispatchResult {
    match status_code {
        200..=299 => usage.map_or(MoaDispatchResult::Responded(status_code), |usage| {
            MoaDispatchResult::RespondedWithUsage { status_code, usage }
        }),
        _ => MoaDispatchResult::FailedWithStatus {
            status_code,
            reason: "moa_turn_failed",
        },
    }
}

fn moa_token_usage(result: &moa::TurnResult) -> Option<TokenUsage> {
    crate::network::openai::response::parse_token_usage_from_json_body(
        result.response_body.to_string().as_bytes(),
    )
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

#[cfg(test)]
mod usage_tests {
    use super::*;

    #[test]
    fn moa_usage_is_authoritative_and_complete() {
        let body = serde_json::json!({
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        });
        assert_eq!(
            crate::network::openai::response::parse_token_usage_from_json_body(
                body.to_string().as_bytes()
            ),
            Some(TokenUsage {
                prompt_tokens: Some(12),
                completion_tokens: Some(7),
                total_tokens: Some(19),
            })
        );
    }

    #[test]
    fn moa_failure_status_discards_usage_and_remains_exact() {
        let usage = Some(TokenUsage {
            prompt_tokens: Some(12),
            completion_tokens: Some(7),
            total_tokens: Some(19),
        });
        assert!(matches!(
            completed_moa_response(502, usage),
            MoaDispatchResult::FailedWithStatus {
                status_code: 502,
                reason: "moa_turn_failed",
            }
        ));
    }
}
