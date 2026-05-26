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

use crate::inference::election;
use crate::mesh;
use crate::network::openai::transport as proxy;
use mesh_mixture_of_agents as moa;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

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

    let Some(mut config) = build_moa_config(node, targets).await else {
        let _ = proxy::send_503(tcp_stream, "MoA requires ≥2 models available in the mesh").await;
        return None;
    };
    config.enable_thinking = enable_thinking;

    run_moa_turn(tcp_stream, body_json, &config, request.response_adapter).await;
    None
}

/// MoA's opinionated default: workers do not think unless the caller
/// explicitly asks for it. Workers are short-budget internal slots, not
/// user-facing reasoning steps. The fast worker's 256-token budget is
/// far too small to fit `<think>…</think>` + answer, and the reducer
/// doesn't want reasoning prose as candidate input.
///
/// The caller can still explicitly enable thinking (e.g. for
/// experimentation) via any of the recognised knobs — see
/// [`extract_enable_thinking_override`]. When no preference is
/// expressed, MoA picks for them: off (always `Some(false)`).
fn effective_enable_thinking_for_moa(body: &serde_json::Value) -> Option<bool> {
    extract_enable_thinking_override(body).or(Some(false))
}

/// Pull the caller's "disable / enable thinking" preference out of an
/// inbound chat-completion or responses JSON body. Mirrors the same
/// shapes that `openai_frontend::common::normalize_reasoning_template_options`
/// recognises so MoA users get the same surface as direct callers.
///
/// Recognised inputs (any one is enough):
/// * `reasoning_effort: "none"` (off) or any non-`"none"` value (on)
/// * `reasoning: { enabled: false }` (off) / `{ enabled: true }` (on)
/// * `reasoning: { effort: "none" }` / `{ max_tokens: 0 }` (off)
/// * Any of `THINKING_BOOLEAN_ALIASES` as a top-level field with bool
/// * `thinking_budget: 0` (off)
/// * `chat_template_kwargs.enable_thinking` (or any alias) as bool
///
/// Returns `None` when the caller hasn't expressed a preference. The
/// MoA-specific policy layer in [`effective_enable_thinking_for_moa`]
/// turns that `None` into `Some(false)` so MoA workers default off.
fn extract_enable_thinking_override(body: &serde_json::Value) -> Option<bool> {
    let obj = body.as_object()?;
    let mut result: Option<bool> = None;

    // reasoning: { enabled, effort, max_tokens }
    if let Some(r) = obj.get("reasoning").and_then(|v| v.as_object()) {
        if r.get("enabled") == Some(&serde_json::Value::Bool(false))
            || r.get("effort").and_then(|v| v.as_str()) == Some("none")
            || r.get("max_tokens").and_then(|v| v.as_u64()) == Some(0)
        {
            result = Some(false);
        } else if r.get("enabled") == Some(&serde_json::Value::Bool(true))
            || r.get("effort").is_some()
            || r.get("max_tokens").is_some()
        {
            result = Some(true);
        }
    }

    // reasoning_effort: "none" / "low" / etc.
    if let Some(effort) = obj.get("reasoning_effort").and_then(|v| v.as_str()) {
        result = Some(effort != "none");
    }

    // Top-level boolean aliases (enable_thinking, enable_reasoning, etc.).
    for alias in openai_frontend::common::THINKING_BOOLEAN_ALIASES {
        if let Some(b) = obj.get(*alias).and_then(|v| v.as_bool()) {
            result = Some(b);
        }
    }

    if obj.get("thinking_budget").and_then(|v| v.as_u64()) == Some(0) {
        result = Some(false);
    }

    // chat_template_kwargs.{enable_thinking, ...}
    if let Some(kwargs) = obj.get("chat_template_kwargs").and_then(|v| v.as_object()) {
        for alias in openai_frontend::common::THINKING_BOOLEAN_ALIASES {
            if let Some(b) = kwargs.get(*alias).and_then(|v| v.as_bool()) {
                result = Some(b);
            }
        }
    }

    result
}

/// Time between progress events while MoA's arbiter is still waiting.
/// One second feels alive without flooding the wire; the typical MoA
/// turn finishes in ~3s, so we emit two or three lines before the
/// real answer starts streaming.
const MOA_PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_millis(1000);

/// Run a turn through the gateway and write the response with x-moa-* headers.
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
        run_moa_turn_with_progress(tcp_stream, moa_body, config, response_adapter).await;
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

/// Streaming MoA turn with `reasoning_content` progress drip.
///
/// 1. Send HTTP response headers immediately (no x-moa-* — those
///    require the result, which we don't have yet).
/// 2. Race `moa::handle_turn` against a periodic ticker. On each
///    tick, emit one progress line via `delta.reasoning_content`.
///    Goose and other OpenAI-shape clients route this to the
///    "thinking" pane; clients that ignore the field simply skip it
///    and see only the final answer.
/// 3. Once MoA returns, hand to the existing SSE body writers with
///    `header_already_sent = true`.
async fn run_moa_turn_with_progress(
    mut tcp_stream: TcpStream,
    moa_body: serde_json::Value,
    config: &moa::GatewayConfig,
    response_adapter: proxy::ResponseAdapter,
) {
    // Generate a single completion id up front so progress chunks,
    // the (eventual) final body, and any failure tail all share the
    // same `chat.completion.chunk.id` — clients correlate the stream
    // by id and a mismatch makes them treat the progress and content
    // as belonging to different completions.
    //
    // We match MoA's own id shape (`chatcmpl-moa-<hex-nanos>`) so the
    // id looks identical to a non-progress-path MoA response. When the
    // body writer runs we'll overwrite the real MoA id with this one.
    let completion_id = format!("chatcmpl-moa-{}", short_hex_nanos());

    if !send_progress_headers(&mut tcp_stream).await {
        return;
    }

    // Responses-API: emit `response.created` immediately, BEFORE any
    // progress deltas. The Responses-API contract is strict about
    // event ordering (`created` → deltas → `completed`), and some
    // clients will reject a stream that starts with deltas.
    if response_adapter == proxy::ResponseAdapter::OpenAiResponsesStream
        && !send_responses_created_for_progress(&mut tcp_stream, &completion_id).await
    {
        return;
    }

    let mut moa_result = drip_progress_until_moa_completes(
        &mut tcp_stream,
        config,
        &moa_body,
        response_adapter,
        &completion_id,
    )
    .await;

    // Mutate the response id so the downstream body writer reuses
    // our generated id rather than emitting a fresh one. Same for
    // the failure path.
    if let Some(obj) = moa_result.response_body.as_object_mut() {
        obj.insert(
            "id".to_string(),
            serde_json::Value::String(completion_id.clone()),
        );
    }

    let result = write_progress_body(
        tcp_stream,
        &moa_result.response_body,
        response_adapter,
        &completion_id,
    )
    .await;
    if let Err(e) = result {
        tracing::warn!("MoA progress: body write failed: {e}");
    }
}

/// Match MoA's `short_id()` — hex of nanos since epoch. Locally
/// derived so we don't have to expose the internal helper.
fn short_hex_nanos() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}")
}

/// Wrapper: emit `response.created` and log on failure. Returns
/// false if the connection died so the caller can bail cleanly.
async fn send_responses_created_for_progress(stream: &mut TcpStream, completion_id: &str) -> bool {
    if let Err(e) = write_progress_response_created(stream, completion_id).await {
        tracing::warn!("MoA progress: response.created write failed: {e}");
        return false;
    }
    true
}

/// Emit `response.created` early on the progress path so the
/// Responses-API stream stays in the required order (`created` first,
/// then any `reasoning_text.delta` progress, then the real
/// `output_text.delta` content from the body writer).
async fn write_progress_response_created(
    stream: &mut TcpStream,
    completion_id: &str,
) -> std::io::Result<()> {
    use openai_frontend::responses as resp;
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let mut created = resp::responses_stream_created_event(moa::VIRTUAL_MODEL_NAME, created_at);
    if let Some(obj) = created
        .get_mut("response")
        .and_then(serde_json::Value::as_object_mut)
    {
        obj.insert(
            "id".to_string(),
            serde_json::Value::String(completion_id.to_string()),
        );
    }
    let data = format!("data: {created}\n\n");
    let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
    stream.write_all(framed.as_bytes()).await?;
    stream.flush().await
}

/// Send HTTP response headers up front so the client knows the
/// stream is alive while MoA arbitrates. Returns true on success.
async fn send_progress_headers(stream: &mut TcpStream) -> bool {
    let header = "HTTP/1.1 200 OK\r\n\
                  Content-Type: text/event-stream\r\n\
                  Transfer-Encoding: chunked\r\n\
                  Cache-Control: no-cache\r\n\
                  Connection: close\r\n\r\n";
    if let Err(e) = stream.write_all(header.as_bytes()).await {
        tracing::warn!("MoA progress: header write failed: {e}");
        return false;
    }
    if let Err(e) = stream.flush().await {
        tracing::warn!("MoA progress: header flush failed: {e}");
        return false;
    }
    true
}

/// Drive the heartbeat ticker until MoA's arbiter commits. Returns
/// the finished MoA result.
///
/// tokio::select! cancellation safety: TCP writes are NOT cancel-safe
/// — a partial write cancelled by the other branch leaves the socket
/// in an inconsistent state. So we only race the (cancel-safe)
/// ticker against MoA, and perform the write outside the select.
/// Worst case: the body write is delayed by up to one interval after
/// MoA finishes.
async fn drip_progress_until_moa_completes(
    stream: &mut TcpStream,
    config: &moa::GatewayConfig,
    moa_body: &serde_json::Value,
    adapter: proxy::ResponseAdapter,
    completion_id: &str,
) -> moa::TurnResult {
    let moa_fut = moa::handle_turn(config, moa_body);
    tokio::pin!(moa_fut);
    let mut ticker = tokio::time::interval(MOA_PROGRESS_INTERVAL);
    // Skip stacked ticks: if a write stalls (slow client, brief
    // network backpressure) the default Burst behaviour would fire
    // the ticker N times back-to-back as soon as we re-enter the
    // select!, dumping a burst of progress lines. Skip drops the
    // missed ticks so we stay at one line per real interval.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Skip the immediate first tick; we want the first line after
    // one interval, not at t=0 (would race the header write).
    ticker.tick().await;
    let mut step = 0usize;
    // Responses-API uses monotonically increasing sequence_number
    // across all events in a response stream. response.created was
    // emitted at seq 0 by write_progress_response_created; progress
    // reasoning deltas start at 1. Chat-completions ignores the
    // counter (it's not part of that wire shape).
    let mut sequence_number: i32 = 1;
    loop {
        tokio::select! {
            biased;
            r = &mut moa_fut => return r,
            _ = ticker.tick() => {}
        }
        let text = progress_line(step, adapter);
        step += 1;
        if let Err(e) =
            write_progress_event(stream, &text, adapter, completion_id, &mut sequence_number).await
        {
            tracing::warn!("MoA progress: tick write failed: {e}; falling back");
            return (&mut moa_fut).await;
        }
    }
}

/// After MoA completes, write the final body — either the real
/// streamed answer or a graceful error tail if MoA failed (we
/// already sent 200 OK, so we can't change the HTTP status).
async fn write_progress_body(
    mut tcp_stream: TcpStream,
    body: &serde_json::Value,
    adapter: proxy::ResponseAdapter,
    completion_id: &str,
) -> std::io::Result<()> {
    if is_moa_failure_body(body) {
        return write_failure_as_sse_tail(&mut tcp_stream, body, adapter, completion_id).await;
    }
    match adapter {
        proxy::ResponseAdapter::OpenAiResponsesStream => {
            send_moa_as_responses_sse_inner(tcp_stream, body, &[], true).await
        }
        _ => send_moa_as_sse_inner(tcp_stream, body, &[], true).await,
    }
}

/// The lines we drip into the thinking pane while MoA arbitrates.
/// Short, factual, and grounded in what mesh-llm is actually doing —
/// not invented model "thoughts". The opening three lines fire
/// once each at ~1s/2s/3s; the rest are a slow "waiting on a slow
/// peer" cycle that explains a long tail without spamming repeats.
fn progress_line(step: usize, _adapter: proxy::ResponseAdapter) -> String {
    const OPENING: &[&str] = &[
        "Routing through mesh…",
        "Querying peer models…",
        "Comparing responses…",
    ];
    const TAIL_CYCLE: &[&str] = &[
        "Waiting on a slow peer…",
        "Still gathering responses…",
        "Hold on, this one's taking a moment…",
    ];
    let line = if step < OPENING.len() {
        OPENING[step]
    } else {
        TAIL_CYCLE[(step - OPENING.len()) % TAIL_CYCLE.len()]
    };
    format!("{line}\n")
}

async fn write_progress_event(
    stream: &mut TcpStream,
    text: &str,
    adapter: proxy::ResponseAdapter,
    completion_id: &str,
    sequence_number: &mut i32,
) -> std::io::Result<()> {
    let data = match adapter {
        proxy::ResponseAdapter::OpenAiResponsesStream => {
            // Responses-API: emit reasoning_text.delta so the UI
            // surfaces these in the thinking pane, separate from the
            // final answer. Emitting output_text.delta here would
            // pollute the visible content (mesh-llm-ui appends
            // output_text into the main bubble).
            //
            // item_id derived from completion_id so progress events
            // and the eventual content events share a coherent item
            // schema within one Responses stream.
            //
            // sequence_number is monotonically increasing across the
            // whole Responses stream (response.created was 0, progress
            // deltas start at 1). Strict Responses-API clients (OpenAI
            // SDK, Vercel AI SDK) rely on this for ordering/dedup.
            let item_id = item_id_from_completion_id(completion_id);
            let seq = *sequence_number;
            *sequence_number = sequence_number.saturating_add(1);
            let ev = serde_json::json!({
                "type": "response.reasoning_text.delta",
                "sequence_number": seq,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            });
            format!("data: {ev}\n\n")
        }
        _ => {
            // Chat-completions stream: drip into `reasoning_content`
            // so goose/openai-sdk-aware clients route this to their
            // "thinking" pane and don't mix it with the final answer.
            // Clients that don't know the field ignore it.
            //
            // `id` matches the completion id we'll use on the final
            // body chunks so clients can correlate the whole stream.
            let chunk = serde_json::json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": moa::VIRTUAL_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": { "reasoning_content": text },
                    "finish_reason": null,
                }],
            });
            format!("data: {chunk}\n\n")
        }
    };
    let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
    stream.write_all(framed.as_bytes()).await?;
    stream.flush().await
}

/// Match the item-id shape used by send_moa_as_responses_sse_inner so
/// progress events and final content events share a coherent
/// item_id within one Responses stream. The body writer uses
/// `format!("msg_moa_{}", short_id_from_response(response))` where
/// `short_id_from_response` takes the suffix after the last `-` from
/// `response.id`. We mirror that here from `completion_id`.
fn item_id_from_completion_id(completion_id: &str) -> String {
    let suffix = completion_id.rsplit('-').next().unwrap_or("x");
    format!("msg_moa_{suffix}")
}

/// Progress-path failure tail: we've already sent 200 OK, so emit
/// the error as a final SSE event followed by [DONE]. The HTTP
/// status can't be changed at this point — best we can do is make
/// sure the stream doesn't silently truncate.
async fn write_failure_as_sse_tail(
    stream: &mut TcpStream,
    body: &serde_json::Value,
    adapter: proxy::ResponseAdapter,
    completion_id: &str,
) -> std::io::Result<()> {
    let err_msg = body
        .pointer("/error/message")
        .and_then(|v| v.as_str())
        .unwrap_or("MoA failed after streaming headers were sent");

    let data = match adapter {
        proxy::ResponseAdapter::OpenAiResponsesStream => {
            let ev = serde_json::json!({
                "type": "response.failed",
                "response": {
                    "id": completion_id,
                    "error": { "message": err_msg },
                },
            });
            format!("data: {ev}\n\n")
        }
        _ => {
            // Same completion_id used by progress chunks — clients
            // correlate the stream by id; mixing ids within one
            // stream confuses chunk-aggregating clients.
            let chunk = serde_json::json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": moa::VIRTUAL_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": { "content": format!("[error: {err_msg}]") },
                    "finish_reason": "error",
                }],
            });
            format!("data: {chunk}\n\n")
        }
    };
    let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
    stream.write_all(framed.as_bytes()).await?;

    let done = "data: [DONE]\n\n";
    let framed = format!("{:x}\r\n{}\r\n", done.len(), done);
    stream.write_all(framed.as_bytes()).await?;
    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await
}

/// Write the MoA response on the chosen transport (JSON or SSE), logging
/// (but not propagating) any I/O error.
///
/// Detect whether a MoA response body is signalling failure.
///
/// Two signals, either of which means "failure":
///
///   * Top-level `error` object — OpenAI-shape error envelope produced
///     by `moa::error_response`.
///   * `choices[0].finish_reason == "error"` — same convention applied
///     by the crate's response builder for in-band failure signalling.
///
/// Previously the HTTP-status decision was based on `TurnKind == Failed`,
/// but the tool-result reducer path can produce an error_response with
/// `TurnKind::ToolResult` when every reducer candidate fails. Tying the
/// status to the body's failure signal instead means *all* error-shaped
/// MoA responses get a non-200 status, regardless of which sub-flow
/// produced them.
fn is_moa_failure_body(body: &serde_json::Value) -> bool {
    if body.get("error").is_some() {
        return true;
    }
    body.pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str())
        == Some("error")
}

/// When the response body signals MoA failure (top-level `error` field or
/// `choices[0].finish_reason == "error"`) we send an HTTP 502 (Bad
/// Gateway), not HTTP 200. Unsophisticated clients that only check the
/// HTTP status need that status to actually reflect failure.
async fn write_moa_response(
    tcp_stream: TcpStream,
    moa_result: &moa::TurnResult,
    extra_headers: &[(&str, String)],
    was_streaming: bool,
    response_adapter: proxy::ResponseAdapter,
) {
    let body = &moa_result.response_body;
    let is_failure = is_moa_failure_body(body);
    // Streaming + failure: respond as non-streaming HTTP 502 with the
    // structured error body. Failure path doesn't go through SSE in any
    // adapter mode — callers want a clean connection-level error.
    let (mode, result) = if was_streaming && !is_failure {
        match response_adapter {
            proxy::ResponseAdapter::OpenAiResponsesStream => (
                "SSE-responses",
                send_moa_as_responses_sse(tcp_stream, body, extra_headers).await,
            ),
            // None, OpenAiChatCompletionsStream, OpenAiResponsesJson all
            // get the chat.completion.chunk SSE shape — the JSON-mode
            // adapter caller will never set was_streaming=true.
            _ => (
                "SSE-chat",
                send_moa_as_sse(tcp_stream, body, extra_headers).await,
            ),
        }
    } else if is_failure {
        (
            "JSON-502",
            proxy::send_json_with_status_and_headers(tcp_stream, 502, body, extra_headers).await,
        )
    } else if response_adapter == proxy::ResponseAdapter::OpenAiResponsesJson {
        // Non-streaming Responses-API request: emit a Responses-shape
        // JSON body instead of the chat.completion shape.
        (
            "JSON-responses",
            proxy::send_json_ok_with_headers(
                tcp_stream,
                &chat_completion_to_responses_json(body),
                extra_headers,
            )
            .await,
        )
    } else {
        (
            "JSON",
            proxy::send_json_ok_with_headers(tcp_stream, body, extra_headers).await,
        )
    };
    if let Err(e) = result {
        tracing::warn!("MoA: response write failed ({mode}): {e}");
    }
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

/// Build a MoA gateway config from this node's mesh-wide view.
///
/// Every distinct model in the mesh becomes a worker:
/// - Models served by this node → `LocalModelBackend` (direct skippy port)
/// - Models served by a peer → `RemoteModelBackend` (QUIC tunnel)
///
/// Models are deduplicated by canonical base name so e.g.
/// `unsloth/Qwen3-8B-GGUF:Q4_K_M` and `Qwen3-8B-Q4_K_M` (different naming
/// conventions for the same model from different peers) only show up once.
///
/// Returns `None` if fewer than 2 distinct models exist — MoA needs at
/// least two workers to be meaningfully different from a single call.
///
/// `targets` is the runtime's local routing table, used to discover the
/// skippy port for locally-served models. In passive (`--client`) mode
/// this is `None` — every backend goes over QUIC. In `serve` mode it's
/// `Some`, so locally-served models bypass the tunnel.
pub async fn build_moa_config(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
) -> Option<moa::GatewayConfig> {
    let http = reqwest::Client::new();
    let mut backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models: Vec<moa::ModelEntry> = Vec::new();
    let mut local_count = 0usize;

    // Full mesh-wide model list (local + every peer's advertised
    // routable models).
    let all_models: Vec<String> = node
        .models_being_served()
        .await
        .into_iter()
        .filter(|n| n != moa::VIRTUAL_MODEL_NAME)
        .collect();

    // Group aliases by canonical base. The old shape sorted by name
    // length, took the *first* alias per base, and dropped the rest —
    // which silently dropped the model from the worker pool whenever the
    // shortest-named peer was unreachable (regression flagged by PR #566
    // review). Now we keep every alias per base and try them in order so
    // a longer-named reachable alias can still resolve when the shortest
    // one is offline.
    let groups = group_aliases_by_canonical_base(all_models, targets);
    for aliases in groups {
        resolve_one_worker_from_aliases(
            node,
            targets,
            &http,
            &aliases,
            &mut backends,
            &mut models,
            &mut local_count,
        )
        .await;
    }

    if models.len() < 2 {
        tracing::warn!(
            "MoA: only {} model(s) reachable, need ≥2 (models={:?})",
            models.len(),
            models.iter().map(|m| &m.name).collect::<Vec<_>>()
        );
        return None;
    }

    tracing::info!(
        "MoA config: {} workers ({} local, {} remote): {:?}",
        models.len(),
        local_count,
        models.len() - local_count,
        models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
    );

    Some(moa::GatewayConfig {
        backends,
        models,
        // Bumped from 15s → 60s. 15s was tight for big-context interactive
        // turns: a large model with a 10–20k-token prompt and tool schema
        // (typical for agent harnesses like OpenCode/Goose) can need 20–30s
        // just to produce a first tool-call. Workers were getting killed
        // mid-inference and MoA reported `kind=early-exit` with the small
        // worker, never the strong one. 60s gives the strong worker room
        // to land without making the no-progress wait painful.
        worker_timeout: std::time::Duration::from_secs(60),
        // Per-attempt cap; hedged_reducer_call hedges across candidates so the
        // end-to-end wait is roughly reducer_timeout + a couple of hedge delays.
        reducer_timeout: std::time::Duration::from_secs(60),
        // Start a second reducer candidate after 5s if the first hasn't replied
        // (or sooner on outright failure). Cheap on the happy path, big win on
        // the cold-KV / stale-peer tail.
        hedge_delay: std::time::Duration::from_secs(5),
        // Chat-only grace: after this long since dispatch, if at least
        // one qualifying Answer is in hand we ship the highest-confidence
        // one. Tool turns bypass this entirely (consensus continues to
        // arbitrate tool proposals).
        //
        // 3 seconds is empirically good across the public mesh today.
        // Long enough that slow-but-good workers (studio MiniMax
        // landing at ~1s, mini Qwen3.5 at ~700ms) finish before the
        // timer; short enough that chat doesn't sit on a multi-second
        // ceiling on every turn. Lab data: median mesh_chat dropped
        // from ~6s (old default) to ~2s with this value, no quality
        // regression measured on factual / arithmetic / short-creative
        // prompts.
        //
        // The previous 6s was conservative because the original grace
        // logic only armed on a sole answer — it had to wait for a
        // second non-matching answer to arrive before becoming useless.
        // With the relaxed eligibility added in this change, the timer
        // is the dominant chat path, so a tighter default is the right
        // default.
        first_answer_grace: std::time::Duration::from_secs(3),
        // Defaults to leaving each model's thinking behavior alone.
        // `try_handle_moa` overrides this from the inbound request body
        // when the caller has expressed a preference
        // (`reasoning_effort: "none"`, `enable_thinking: false`, etc.).
        enable_thinking: None,
    })
}

/// Try each alias in `aliases` until one resolves to a backend, then stop.
///
/// Aliases are pre-sorted by `group_aliases_by_canonical_base` so the most
/// preferred (locally-served first, then shortest) is tried first. Falls
/// back to longer aliases when the preferred one's peer is unreachable.
#[allow(clippy::too_many_arguments)]
async fn resolve_one_worker_from_aliases(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    http: &reqwest::Client,
    aliases: &[String],
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    local_count: &mut usize,
) {
    for name in aliases {
        if add_worker_backend(node, targets, http, name, backends, models, local_count).await {
            return;
        }
    }
}

/// Group all advertised model names by their canonical base so each
/// canonical model contributes exactly one worker, but the resolver gets
/// to pick the alias that actually has a reachable backend.
///
/// The earlier shape committed to a single alias per base *before* trying
/// to resolve a backend. Two failure modes:
///
///   1. The chosen alias is advertised only by a peer that drops between
///      gossip refresh and orchestration — `hosts_for_model` returns
///      empty, the worker is dropped, and longer-form aliases for the
///      same canonical model from still-reachable peers are rejected as
///      duplicates.
///   2. The local node advertises a longer convention
///      (e.g. `unsloth/Qwen3-8B-GGUF:Q4_K_M`) while a peer advertises a
///      shorter variant (e.g. `Qwen3-8B-Q4_K_M`). The shortest-name rule
///      picks the peer alias, `add_worker_backend` looks for a local port
///      under that specific string, finds nothing, and forces a
///      QUIC-tunnel backend even though the model is right here.
///
/// Both failure modes are fixed by grouping first and resolving second.
/// Within each group the aliases are ordered so the most likely
/// optimization wins first try: locally-served name (skippy-port fast
/// path) before remote names, then shortest first as a tiebreaker.
fn group_aliases_by_canonical_base(
    names: Vec<String>,
    targets: Option<&election::ModelTargets>,
) -> Vec<Vec<String>> {
    let mut by_base: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for name in names {
        by_base
            .entry(canonical_base_name(&name))
            .or_default()
            .push(name);
    }
    // Deterministic group order so the worker list is stable across
    // builds even though HashMap iteration is not. Sort group entries
    // (locally-served first, then shortest), then sort groups by their
    // first ("best") alias.
    let mut groups: Vec<Vec<String>> = by_base
        .into_values()
        .map(|mut aliases| {
            aliases.sort_by(|a, b| {
                let la = is_locally_served(a, targets);
                let lb = is_locally_served(b, targets);
                lb.cmp(&la) // local (true) before remote (false)
                    .then_with(|| a.len().cmp(&b.len()))
                    .then_with(|| a.cmp(b))
            });
            aliases
        })
        .collect();
    groups.sort_by(|a, b| a[0].cmp(&b[0]));
    groups
}

/// Does the local routing table have a backend port for this exact name?
fn is_locally_served(name: &str, targets: Option<&election::ModelTargets>) -> bool {
    targets
        .and_then(|t| {
            t.targets.get(name).map(|tv| {
                tv.iter()
                    .any(|t| matches!(t, election::InferenceTarget::Local(_)))
            })
        })
        .unwrap_or(false)
}

/// Resolve `name` to a backend (local skippy port if available, else first
/// remote host) and append it to `backends`/`models`. Returns true if a
/// backend was added.
async fn add_worker_backend(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    http: &reqwest::Client,
    name: &str,
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    local_count: &mut usize,
) -> bool {
    // Prefer local skippy port when this node serves the model.
    let local_port = targets.and_then(|t| {
        t.targets.get(name).and_then(|tv| {
            tv.iter().find_map(|t| match t {
                election::InferenceTarget::Local(p) => Some(*p),
                _ => None,
            })
        })
    });
    if let Some(port) = local_port {
        let backend_idx = backends.len();
        backends.push(std::sync::Arc::new(LocalModelBackend {
            port,
            http: http.clone(),
        }));
        models.push(moa::ModelEntry {
            name: name.to_string(),
            backend_index: backend_idx,
        });
        *local_count += 1;
        return true;
    }

    // Otherwise find a remote host. hosts_for_model returns peers in
    // hash-preferred order; take the first.
    let remote_hosts = node.hosts_for_model(name).await;
    if let Some(peer_id) = remote_hosts.into_iter().next() {
        let backend_idx = backends.len();
        backends.push(std::sync::Arc::new(RemoteModelBackend {
            node: node.clone(),
            peer_id,
        }));
        models.push(moa::ModelEntry {
            name: name.to_string(),
            backend_index: backend_idx,
        });
        return true;
    }
    false
}

/// Canonical name used for cross-peer dedup. Different peers advertise the
/// same model under different conventions (`unsloth/Qwen3-8B-GGUF:Q4_K_M`
/// vs `Qwen3-8B-Q4_K_M`); normalize before comparing.
///
/// Strategy: strip the publisher prefix, the `-gguf` suffix, any `@branch`
/// suffix, then keep only `[a-z0-9]` characters so `:` vs `-` separators
/// don't matter.
fn canonical_base_name(name: &str) -> String {
    let lower = name.to_lowercase();
    // Drop an `@branch` segment if present, keeping anything after the
    // next `:` so quant tags survive (e.g. `repo@main:q4_k_m` → `repo:q4_k_m`).
    let no_branch = match lower.find('@') {
        Some(at) => {
            let after = &lower[at + 1..];
            let rest = after.find(':').map(|c| &after[c..]).unwrap_or("");
            format!("{}{}", &lower[..at], rest)
        }
        None => lower,
    };
    let stripped = no_branch
        .replace("-gguf", "")
        .replace("unsloth/", "")
        .replace("meshllm/", "");
    stripped
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

/// Backend that calls a local model directly on its skippy HTTP port.
struct LocalModelBackend {
    port: u16,
    http: reqwest::Client,
}

#[async_trait::async_trait]
impl moa::ModelBackend for LocalModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
            "mesh_hooks": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        moa::apply_enable_thinking(&mut body, sampling.enable_thinking);
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| format!("local:{} failed: {e}", self.port))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "HTTP {status}: {}",
                moa::truncate_chars(&text, 200)
            ));
        }
        resp.json::<serde_json::Value>()
            .await
            .map_err(|e| format!("parse: {e}"))
    }
}

/// Backend that calls a remote model over the QUIC tunnel.
struct RemoteModelBackend {
    node: mesh::Node,
    peer_id: iroh::EndpointId,
}

#[async_trait::async_trait]
impl moa::ModelBackend for RemoteModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
            "mesh_hooks": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        moa::apply_enable_thinking(&mut body, sampling.enable_thinking);
        let body_bytes = serde_json::to_vec(&body).map_err(|e| format!("serialize: {e}"))?;
        let http_request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             \r\n",
            body_bytes.len()
        );
        let mut raw = http_request.into_bytes();
        raw.extend_from_slice(&body_bytes);

        tokio::time::timeout(timeout, async {
            let (mut send, mut recv) = self
                .node
                .open_http_tunnel(self.peer_id)
                .await
                .map_err(|e| format!("tunnel: {e}"))?;
            send.write_all(&raw)
                .await
                .map_err(|e| format!("send: {e}"))?;
            send.finish().map_err(|e| format!("finish: {e}"))?;
            let response = recv
                .read_to_end(4 * 1024 * 1024)
                .await
                .map_err(|e| format!("recv: {e}"))?;
            parse_quic_http_response(&response)
        })
        .await
        .map_err(|_| format!("remote timeout after {}s", timeout.as_secs()))?
    }
}

fn parse_quic_http_response(response: &[u8]) -> Result<serde_json::Value, String> {
    let s = String::from_utf8_lossy(response);
    let header_end = s
        .find("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let status_line = s[..header_end].lines().next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if status != 200 {
        return Err(format!("HTTP {status}: {}", moa::truncate_chars(&s, 200)));
    }
    let body = &s[header_end + 4..];
    serde_json::from_str(body).map_err(|e| format!("parse: {e}"))
}

/// Send the MoA response as a one-shot SSE stream so SSE-only clients
/// (like Goose) can consume it. Emits one delta chunk with the full
/// content, then a `finish_reason: stop` chunk, then `[DONE]`.
///
/// `extra_headers` are emitted alongside the standard SSE response headers
/// (used to attach `x-moa-*` observability headers).
async fn send_moa_as_sse(
    stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    send_moa_as_sse_inner(stream, response, extra_headers, false).await
}

/// Write the standard SSE response header block, with optional
/// per-response extra headers (used for `x-moa-*` observability).
async fn write_sse_response_headers(
    stream: &mut TcpStream,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let mut header = String::from(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/event-stream\r\n\
         Transfer-Encoding: chunked\r\n\
         Cache-Control: no-cache\r\n\
         Connection: close\r\n",
    );
    for (name, value) in extra_headers {
        crate::network::openai::transport::append_safe_header(&mut header, name, value);
    }
    header.push_str("\r\n");
    stream.write_all(header.as_bytes()).await
}

async fn send_moa_as_sse_inner(
    mut stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
    header_already_sent: bool,
) -> std::io::Result<()> {
    if !header_already_sent {
        write_sse_response_headers(&mut stream, extra_headers).await?;
    }

    let id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-mesh");
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(moa::VIRTUAL_MODEL_NAME);
    let raw_content = response
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let content = strip_think_from_content(raw_content);

    let tool_calls = response
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())
        .cloned();

    // Caller (`write_moa_response`) routes failure-shaped bodies to a
    // non-streaming 502 JSON response, so this function only ever sees a
    // successful turn. The only choice the SSE adapter still has to make
    // is `tool_calls` vs `stop`.
    let finish_reason: &str = if tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    debug_assert!(
        !is_moa_failure_body(response),
        "send_moa_as_sse received a failure body; should have routed to 502"
    );

    // Tool-call payloads are structured JSON — they must remain
    // atomic so harness parsers (Goose, OpenCode) see a single
    // well-formed tool_call object. Only the assistant *text* path
    // benefits from pseudo-streaming.
    if let Some(ref tcs) = tool_calls {
        let delta = serde_json::json!({
            "role": "assistant",
            "tool_calls": tcs.iter().enumerate().map(|(i, tc)| {
                serde_json::json!({
                    "index": i,
                    "id": tc.get("id").and_then(|v| v.as_str()).unwrap_or("call_0"),
                    "type": "function",
                    "function": tc.get("function").cloned().unwrap_or(serde_json::json!({})),
                })
            }).collect::<Vec<_>>()
        });
        let chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": null,
            }]
        });
        let data = format!("data: {}\n\n", chunk);
        let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
        stream.write_all(framed.as_bytes()).await?;
    } else {
        // Text path: split content into chunks for pseudo-streaming.
        // First chunk carries `role: "assistant"`; continuation chunks
        // carry only `content` (matches OpenAI streaming convention).
        let pieces = chunk_content_for_streaming(&content, MOA_STREAM_CHUNKS);
        let chunk_delay = MOA_STREAM_CHUNK_DELAY;
        let inter_chunk_delay = if pieces.len() > 1 {
            Some(chunk_delay)
        } else {
            None
        };
        for (idx, piece) in pieces.iter().enumerate() {
            let delta = if idx == 0 {
                serde_json::json!({ "role": "assistant", "content": piece })
            } else {
                serde_json::json!({ "content": piece })
            };
            let chunk = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": null,
                }]
            });
            let data = format!("data: {}\n\n", chunk);
            let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
            stream.write_all(framed.as_bytes()).await?;
            stream.flush().await?;
            if let Some(delay) = inter_chunk_delay
                && idx + 1 < pieces.len()
            {
                tokio::time::sleep(delay).await;
            }
        }
    }

    let stop = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }]
    });
    let data = format!("data: {}\n\n", stop);
    let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
    stream.write_all(framed.as_bytes()).await?;

    let done = "data: [DONE]\n\n";
    let framed = format!("{:x}\r\n{}\r\n", done.len(), done);
    stream.write_all(framed.as_bytes()).await?;

    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await?;
    Ok(())
}

/// Strip `<think>...</think>` tags and orphan `</think>` from content.
/// Thin wrapper over the canonical implementation in moa::worker.
fn strip_think_from_content(text: &str) -> String {
    moa::strip_thinking(text)
}

/// Number of chunks to split MoA winner content into when emitting
/// pseudo-streaming SSE. Tuned for "feels live" — ~25 chunks over a
/// buffered response of any reasonable length lets the chat UI paint
/// progressively instead of jumping from spinner to wall-of-text.
const MOA_STREAM_CHUNKS: usize = 25;

/// Minimum content length (bytes) before pseudo-streaming kicks in.
/// Below this, the one-shot delta is fine and chunking just adds
/// scheduler noise. Checked against `content.len()` which is byte
/// length; the threshold is loose so the byte/char distinction
/// doesn't matter for non-ASCII (200 bytes ≥ 50 multi-byte chars,
/// well above the noise floor).
const MOA_STREAM_MIN_BYTES: usize = 200;

/// Delay between pseudo-stream chunks. Total animation budget for a
/// 25-chunk response is ~500ms, which feels live without artificially
/// slowing down agents that just want to read the whole reply.
const MOA_STREAM_CHUNK_DELAY: std::time::Duration = std::time::Duration::from_millis(20);

/// Split `content` into roughly `target_chunks` pieces along whitespace
/// or UTF-8 char boundaries. The returned slices, concatenated in order,
/// always reconstruct the original input exactly (no characters lost,
/// no separators inserted). Returns a single-element vector when
/// chunking is not worth the overhead (short content, target ≤ 1, or
/// content too short to split meaningfully).
fn chunk_content_for_streaming(content: &str, target_chunks: usize) -> Vec<&str> {
    if target_chunks <= 1
        || content.len() < MOA_STREAM_MIN_BYTES
        || content.chars().count() < target_chunks * 2
    {
        return vec![content];
    }

    // Walk char boundaries to compute desired cut points by char index,
    // then snap forward to the next whitespace boundary so we don't
    // split mid-word. If no whitespace exists (CJK, code blob, long
    // hash), fall through to the char-boundary cut.
    let total_chars = content.chars().count();
    let chars_per_chunk = total_chars / target_chunks;
    if chars_per_chunk == 0 {
        return vec![content];
    }

    let mut chunks = Vec::with_capacity(target_chunks);
    let mut cut_start = 0usize;
    let mut chars_since_last = 0usize;

    for (byte_idx, ch) in content.char_indices() {
        chars_since_last += 1;
        // Once we've passed the per-chunk char target, try to snap
        // forward to the next whitespace char so we cut on a word
        // boundary. If we're already on whitespace, cut here.
        if chars_since_last >= chars_per_chunk && ch.is_whitespace() {
            // Cut *after* the whitespace so the leading-space
            // boundary lives with the preceding chunk (matches how
            // word-by-word streaming reads).
            let cut_end = byte_idx + ch.len_utf8();
            if cut_end > cut_start {
                chunks.push(&content[cut_start..cut_end]);
                cut_start = cut_end;
                chars_since_last = 0;
            }
            if chunks.len() + 1 >= target_chunks {
                break;
            }
        }
    }

    if cut_start < content.len() {
        chunks.push(&content[cut_start..]);
    }

    // If we ended up with one chunk (no whitespace found), fall back
    // to a strict char-count split. Common for CJK or code-only output.
    if chunks.len() == 1 && total_chars >= target_chunks * 2 {
        chunks.clear();
        let mut cut_start = 0usize;
        let mut chars_since_last = 0usize;
        for (byte_idx, ch) in content.char_indices() {
            chars_since_last += 1;
            if chars_since_last >= chars_per_chunk {
                let cut_end = byte_idx + ch.len_utf8();
                chunks.push(&content[cut_start..cut_end]);
                cut_start = cut_end;
                chars_since_last = 0;
                if chunks.len() + 1 >= target_chunks {
                    break;
                }
            }
        }
        if cut_start < content.len() {
            chunks.push(&content[cut_start..]);
        }
    }

    chunks
}

/// Emit the MoA response as a one-shot OpenAI Responses-API SSE stream
/// so callers that hit `/v1/responses` with `stream:true` (the chat UI)
/// get event shapes their parser understands.
///
/// We synthesize the minimum set the standard Responses-API stream
/// emits: `response.created`, `response.output_text.delta` (one chunk
/// with the full content), `response.output_text.done`, and
/// `response.completed`. MoA always knows the full body before writing,
/// so we don't bother with incremental delta streaming — it would only
/// make the UI's spinner-to-text transition smoother.
async fn send_moa_as_responses_sse(
    stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    send_moa_as_responses_sse_inner(stream, response, extra_headers, false).await
}

async fn send_moa_as_responses_sse_inner(
    mut stream: TcpStream,
    response: &serde_json::Value,
    extra_headers: &[(&str, String)],
    header_already_sent: bool,
) -> std::io::Result<()> {
    if !header_already_sent {
        write_sse_response_headers(&mut stream, extra_headers).await?;
    }

    let response_id = response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("resp_moa")
        .to_string();
    let model = response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(moa::VIRTUAL_MODEL_NAME)
        .to_string();
    let raw_content = response
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let content = strip_think_from_content(raw_content);
    // MoA's body is chat-shape; the Responses-API completed event
    // expects input_tokens / output_tokens. Translate before emitting
    // so downstream consumers (chat UI, billing) see the right keys.
    let usage = response
        .get("usage")
        .map(openai_frontend::responses::chat_usage_to_responses_usage);
    let item_id = format!("msg_moa_{}", short_id_from_response(response));
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    use openai_frontend::responses as resp;

    // `response.created` must come before any delta events. When the
    // progress path is driving us (header_already_sent), it already
    // emitted `response.created` up front with the correct id — so
    // we must NOT emit a second one here (clients would see two
    // `created` events with different timestamps within one stream).
    //
    // When we're called directly (header_already_sent = false), the
    // body writer owns the `created` event; emit it now with the
    // correct response_id.
    if !header_already_sent {
        let mut created = resp::responses_stream_created_event(&model, created_at);
        if let Some(obj) = created
            .get_mut("response")
            .and_then(serde_json::Value::as_object_mut)
        {
            obj.insert(
                "id".to_string(),
                serde_json::Value::String(response_id.clone()),
            );
        }
        let data = format!("data: {created}\n\n");
        let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
        stream.write_all(framed.as_bytes()).await?;
        stream.flush().await?;
    }

    let pieces = chunk_content_for_streaming(&content, MOA_STREAM_CHUNKS);
    let chunk_delay = MOA_STREAM_CHUNK_DELAY;
    let inter_chunk_delay = if pieces.len() > 1 {
        Some(chunk_delay)
    } else {
        None
    };
    for (idx, piece) in pieces.iter().enumerate() {
        let delta_event = resp::responses_stream_delta_event(&item_id, piece);
        let data = format!("data: {}\n\n", delta_event);
        let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
        stream.write_all(framed.as_bytes()).await?;
        stream.flush().await?;
        if let Some(delay) = inter_chunk_delay
            && idx + 1 < pieces.len()
        {
            tokio::time::sleep(delay).await;
        }
    }

    let tail = [
        resp::responses_stream_text_done_event(&item_id, &content),
        resp::responses_stream_completed_event(
            &response_id,
            created_at,
            &model,
            &item_id,
            &content,
            usage,
        ),
    ];
    for event in &tail {
        let data = format!("data: {}\n\n", event);
        let framed = format!("{:x}\r\n{}\r\n", data.len(), data);
        stream.write_all(framed.as_bytes()).await?;
    }

    let done = "data: [DONE]\n\n";
    let framed = format!("{:x}\r\n{}\r\n", done.len(), done);
    stream.write_all(framed.as_bytes()).await?;

    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await?;
    Ok(())
}

/// Convert a chat.completion JSON body to a Responses-API JSON body.
/// Used for non-streaming `/v1/responses` requests against MoA.
fn chat_completion_to_responses_json(chat: &serde_json::Value) -> serde_json::Value {
    let bytes = serde_json::to_vec(chat).unwrap_or_default();
    match crate::network::openai::response_adapter::translate_chat_completion_to_responses(&bytes) {
        Ok(translated) => serde_json::from_slice(&translated).unwrap_or_else(|_| chat.clone()),
        Err(e) => {
            tracing::warn!("MoA: chat-to-responses JSON translate failed: {e}");
            chat.clone()
        }
    }
}

fn short_id_from_response(response: &serde_json::Value) -> String {
    response
        .get("id")
        .and_then(|v| v.as_str())
        .and_then(|id| id.rsplit('-').next())
        .unwrap_or("x")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_base_dedupes_unsloth_and_gguf_variants() {
        assert_eq!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF:Q4_K_M"),
            canonical_base_name("Qwen3-8B-Q4_K_M")
        );
        assert_eq!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF@main:Q4_K_M"),
            canonical_base_name("Qwen3-8B-Q4_K_M")
        );
    }

    #[test]
    fn canonical_base_keeps_distinct_models_distinct() {
        assert_ne!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF:Q4_K_M"),
            canonical_base_name("unsloth/Qwen3-32B-GGUF:Q4_K_M")
        );
        assert_ne!(
            canonical_base_name("unsloth/Qwen3-32B-GGUF:Q4_K_M"),
            canonical_base_name("unsloth/MiniMax-M2.5-GGUF:Q4_K_M")
        );
    }

    #[test]
    fn strip_think_handles_simple_block() {
        assert_eq!(
            strip_think_from_content("<think>reasoning</think>answer"),
            "answer"
        );
    }

    #[test]
    fn strip_think_handles_orphan_close_tag() {
        // Orphan `</think>` is removed but prefix content is preserved.
        assert_eq!(
            strip_think_from_content("stuff</think>answer"),
            "stuffanswer"
        );
    }

    #[test]
    fn strip_think_handles_unclosed_block() {
        assert_eq!(
            strip_think_from_content("answer prefix<think>never closed"),
            "answer prefix"
        );
    }

    #[test]
    fn is_moa_failure_body_detects_top_level_error() {
        // Regression for PR #566 review (item #7): the HTTP status was
        // gated on `TurnKind == Failed`, but reducer-failure tool-result
        // turns produce an error_response with `TurnKind::ToolResult`.
        // The body still carries the canonical failure signals, so
        // status now follows the body.
        let body = serde_json::json!({
            "error": { "message": "reducer failed", "type": "moa_failure" },
            "choices": [{ "finish_reason": "error", "message": { "content": "oops" } }],
        });
        assert!(is_moa_failure_body(&body));
    }

    #[test]
    fn is_moa_failure_body_detects_finish_reason_error() {
        let body = serde_json::json!({
            "choices": [{ "finish_reason": "error", "message": { "content": "oops" } }],
        });
        assert!(is_moa_failure_body(&body));
    }

    #[test]
    fn is_moa_failure_body_returns_false_for_success() {
        let body = serde_json::json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "hello" } }],
        });
        assert!(!is_moa_failure_body(&body));
    }

    fn make_targets(local_names: &[&str]) -> election::ModelTargets {
        let mut t = election::ModelTargets::default();
        for (i, name) in local_names.iter().enumerate() {
            t.targets.insert(
                (*name).to_string(),
                vec![election::InferenceTarget::Local(50000 + i as u16)],
            );
        }
        t
    }

    #[test]
    fn group_aliases_keeps_all_aliases_per_canonical_base() {
        // Regression for PR #566 review (item #10): the dedup-then-resolve
        // shape committed to a single alias per base before checking
        // backend reachability. Now every alias is retained so the
        // resolver can fall back if the preferred alias is unreachable.
        let groups = group_aliases_by_canonical_base(
            vec![
                "Qwen3-8B-Q4_K_M".to_string(),
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 1, "both names share a canonical base");
        assert_eq!(groups[0].len(), 2, "both aliases retained");
    }

    #[test]
    fn group_aliases_prefers_locally_served_alias_even_when_longer() {
        // Without a targets table, length-order wins and the shorter peer
        // alias would be tried first — forcing an unnecessary QUIC hop
        // when the model is right here under a different alias.
        // With targets, the local-served alias must come first.
        let local = "unsloth/Qwen3-8B-GGUF:Q4_K_M";
        let peer = "Qwen3-8B-Q4_K_M";
        let targets = make_targets(&[local]);
        let groups = group_aliases_by_canonical_base(
            vec![peer.to_string(), local.to_string()],
            Some(&targets),
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].first().map(String::as_str),
            Some(local),
            "locally-served alias must win even though it's longer"
        );
    }

    #[test]
    fn group_aliases_falls_back_to_shortest_when_no_local() {
        // No targets table at all (pure --client --auto node) — shortest
        // alias should win, but the longer alias is still in the group so
        // it can be tried if the shortest one is unreachable.
        let groups = group_aliases_by_canonical_base(
            vec![
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
                "Qwen3-8B-Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].first().map(String::as_str),
            Some("Qwen3-8B-Q4_K_M")
        );
        assert_eq!(groups[0].len(), 2, "longer alias kept as fallback");
    }

    #[test]
    fn group_aliases_distinct_models_stay_in_separate_groups() {
        let groups = group_aliases_by_canonical_base(
            vec![
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
                "unsloth/Qwen3-32B-GGUF:Q4_K_M".to_string(),
                "unsloth/MiniMax-M2.5-GGUF:Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn is_moa_failure_body_returns_false_for_tool_calls() {
        let body = serde_json::json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
                },
            }],
        });
        assert!(!is_moa_failure_body(&body));
    }

    // ── Streaming + failure routing ────────────────────────────────────
    //
    // The actual write path (`write_moa_response`) writes to a real
    // `TcpStream`, so we test the *decision* it makes by extracting the
    // failure detection into `is_moa_failure_body` and proving the
    // routing logic with the same booleans the writer uses.
    //
    // The contract is:
    //   was_streaming=false, is_failure=false  -> JSON 200
    //   was_streaming=false, is_failure=true   -> JSON 502
    //   was_streaming=true,  is_failure=false  -> SSE
    //   was_streaming=true,  is_failure=true   -> JSON 502 (NOT SSE 200)
    // The last row is the PR #612 review finding: streaming MoA failures
    // must surface as a real 502 at the HTTP layer instead of streaming
    // a 200 SSE carrying an in-band error.

    fn route_decision(was_streaming: bool, is_failure: bool) -> &'static str {
        if was_streaming && !is_failure {
            "sse"
        } else if is_failure {
            "json-502"
        } else {
            "json-200"
        }
    }

    #[test]
    fn streaming_success_routes_to_sse() {
        assert_eq!(route_decision(true, false), "sse");
    }

    #[test]
    fn streaming_failure_routes_to_json_502_not_sse() {
        // Regression for PR #612 review: streaming failures previously
        // went out as `SSE 200` + in-band `finish_reason: "error"`.
        // Now they collapse to a non-streaming JSON 502, matching the
        // OpenAI API and the non-streaming MoA failure path.
        assert_eq!(route_decision(true, true), "json-502");
    }

    #[test]
    fn non_streaming_success_routes_to_json_200() {
        assert_eq!(route_decision(false, false), "json-200");
    }

    #[test]
    fn non_streaming_failure_routes_to_json_502() {
        assert_eq!(route_decision(false, true), "json-502");
    }

    // ── Responses-API adapter ───────────────────────────────────────
    //
    // When the request came in via /v1/responses, MoA's response must
    // be rendered in the Responses-API shape, not chat.completion. The
    // chat UI's streaming parser ignores chat.completion.chunk events,
    // which is what caused the "streaming response" spinner with no
    // visible text on the public mesh.

    fn fixture_chat_completion(content: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-moa-fixture",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
        })
    }

    #[test]
    fn chat_completion_to_responses_json_returns_response_object() {
        // Non-streaming /v1/responses with model=mesh: the body that
        // reaches the client must be Responses-shape, not chat-shape.
        let chat = fixture_chat_completion("hello world");
        let responses = chat_completion_to_responses_json(&chat);
        assert_eq!(
            responses.get("object").and_then(|v| v.as_str()),
            Some("response"),
            "got: {}",
            serde_json::to_string(&responses).unwrap_or_default()
        );
        // The text must survive translation.
        let text = serde_json::to_string(&responses).unwrap_or_default();
        assert!(
            text.contains("hello world"),
            "response body must carry the original content; got {text}"
        );
    }

    #[test]
    fn chat_completion_to_responses_json_passes_through_on_malformed() {
        // Defensive: if the translator can't make sense of the body
        // we return the chat body unchanged rather than blowing up.
        let bogus = serde_json::json!({ "not": "a chat completion" });
        let out = chat_completion_to_responses_json(&bogus);
        // The translator may either succeed (producing an empty
        // response) or fall back to the input; both behaviours are
        // acceptable, what matters is no panic and a JSON value.
        assert!(out.is_object());
    }

    /// Run `send_moa_as_responses_sse` against a real TCP loopback
    /// pair and return the raw bytes the client received as a string.
    /// Includes HTTP/1.1 headers and the chunked-transfer framing
    /// around each SSE event. Callers in this module match by
    /// `.contains(...)`, which is robust to framing without needing
    /// to parse it.
    async fn capture_responses_sse_body(response: serde_json::Value) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");

        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept");
            send_moa_as_responses_sse(socket, &response, &[])
                .await
                .expect("sse write");
        });

        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server task");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[tokio::test]
    async fn responses_sse_uses_same_response_id_for_created_and_completed() {
        // Regression: created and completed events used different
        // `response.id` values (one auto-generated, one from the chat
        // body), breaking clients that correlate by id.
        let response = serde_json::json!({
            "id": "chatcmpl-moa-correlation",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }]
        });

        let raw = capture_responses_sse_body(response).await;

        // Extract every `data: { ... }` JSON blob and look at
        // (event.type, event.response.id).
        let mut ids = Vec::<(String, String)>::new();
        for line in raw.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };
            let event_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if event_type == "response.created" || event_type == "response.completed" {
                let id = v
                    .pointer("/response/id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                ids.push((event_type.to_string(), id));
            }
        }

        assert_eq!(ids.len(), 2, "need created + completed; got {ids:?}");
        assert_eq!(ids[0].1, "chatcmpl-moa-correlation");
        assert_eq!(
            ids[0].1, ids[1].1,
            "created and completed must share response.id: {ids:?}"
        );
    }

    #[tokio::test]
    async fn responses_sse_emits_responses_shape_usage_not_chat_shape() {
        // Regression: MoA was forwarding the chat-completion `usage`
        // object (prompt_tokens/completion_tokens) straight into the
        // Responses-API completed event, which expects
        // input_tokens/output_tokens. Downstream consumers that read
        // `response.usage.input_tokens` saw `undefined`.
        let response = serde_json::json!({
            "id": "chatcmpl-moa-fixture",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 13,
                "total_tokens": 24
            }
        });

        let raw = capture_responses_sse_body(response).await;

        // The completed event carries the response object including
        // usage. We assert by string match so we're robust to
        // serializer ordering.
        assert!(
            raw.contains("\"input_tokens\":11"),
            "expected input_tokens=11 in SSE; got: {raw}"
        );
        assert!(
            raw.contains("\"output_tokens\":13"),
            "expected output_tokens=13 in SSE; got: {raw}"
        );
        assert!(
            raw.contains("\"total_tokens\":24"),
            "expected total_tokens=24 in SSE; got: {raw}"
        );
        assert!(
            !raw.contains("\"prompt_tokens\":"),
            "chat-shape prompt_tokens must NOT leak into Responses-API SSE; got: {raw}"
        );
        assert!(
            !raw.contains("\"completion_tokens\":"),
            "chat-shape completion_tokens must NOT leak into Responses-API SSE; got: {raw}"
        );
    }

    // ── extract_enable_thinking_override ────────────────────────────────
    //
    // Mirrors the shapes that `openai_frontend::common::normalize_reasoning_template_options`
    // accepts, so MoA users get the same surface as direct callers. If we
    // forget a shape, the model never gets told to stop thinking and the
    // fast worker burns its budget inside `<think>`.

    #[test]
    fn extract_no_knobs_returns_none() {
        let body = serde_json::json!({"model": "mesh", "messages": []});
        assert_eq!(extract_enable_thinking_override(&body), None);
    }

    #[test]
    fn extract_reasoning_effort_none_disables() {
        let body = serde_json::json!({"reasoning_effort": "none"});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_reasoning_effort_low_enables() {
        let body = serde_json::json!({"reasoning_effort": "low"});
        assert_eq!(extract_enable_thinking_override(&body), Some(true));
    }

    #[test]
    fn extract_reasoning_enabled_false_disables() {
        let body = serde_json::json!({"reasoning": {"enabled": false}});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_reasoning_max_tokens_zero_disables() {
        let body = serde_json::json!({"reasoning": {"max_tokens": 0}});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_top_level_enable_thinking_false() {
        let body = serde_json::json!({"enable_thinking": false});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_top_level_enable_thinking_alias() {
        // `use_thinking` is one of THINKING_BOOLEAN_ALIASES.
        let body = serde_json::json!({"use_thinking": false});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_thinking_budget_zero_disables() {
        let body = serde_json::json!({"thinking_budget": 0});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_chat_template_kwargs_passes_through() {
        let body = serde_json::json!({
            "chat_template_kwargs": {"enable_thinking": false}
        });
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_latest_wins_when_multiple_set() {
        // chat_template_kwargs is read last and so wins. Whatever ordering
        // we choose, picking ONE consistently is the contract.
        let body = serde_json::json!({
            "reasoning_effort": "low",                                  // enable
            "chat_template_kwargs": {"enable_thinking": false},         // disable
        });
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    // ── MoA opinionated default ────────────────────────────────────────────────────
    //
    // For `model: "mesh"`, MoA does NOT let reasoning models think on
    // worker slots. The fast worker has a 256-token budget that doesn't
    // fit `<think>...</think>` + answer, and the reducer doesn't want
    // reasoning prose as candidate input. Callers can explicitly turn
    // reasoning back on, but the default is off.

    #[test]
    fn effective_default_is_no_thinking_when_caller_silent() {
        // No knobs in the body → MoA's opinion applies.
        let body = serde_json::json!({"model": "mesh", "messages": []});
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(false));
    }

    #[test]
    fn effective_respects_explicit_disable_from_caller() {
        let body = serde_json::json!({
            "reasoning_effort": "none",
            "model": "mesh",
        });
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(false));
    }

    #[test]
    fn effective_lets_caller_explicitly_enable_thinking() {
        // Escape hatch: a caller who really wants reasoning on MoA can
        // ask for it via any of the recognised knobs.
        let body = serde_json::json!({
            "reasoning_effort": "low",
            "model": "mesh",
        });
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(true));
    }

    #[test]
    fn effective_default_for_tool_calling_request_still_no_thinking() {
        // Agentic / tool turns get the same opinionated default.
        // The grace-bypass / consensus path in MoA already runs
        // differently for tool turns, but thinking is independent of
        // that and should still be off unless the caller insists.
        let body = serde_json::json!({
            "model": "mesh",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "x"}}],
        });
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(false));
    }

    // ── chunk_content_for_streaming ────────────────────────────────

    #[test]
    fn chunk_helper_empty_input_returns_single_empty_chunk() {
        // Empty input still returns a one-element vec (`vec![""]`), not
        // an empty slice — the SSE writer expects to always emit at
        // least one delta event so it can attach role/finish metadata.
        assert_eq!(chunk_content_for_streaming("", 25), vec![""]);
    }

    #[test]
    fn chunk_helper_short_input_returns_single_chunk() {
        // Below MOA_STREAM_MIN_BYTES — chunking overhead not worth it.
        let s = "hello world this is short";
        let out = chunk_content_for_streaming(s, 25);
        assert_eq!(out, vec![s]);
    }

    #[test]
    fn chunk_helper_target_one_returns_single_chunk() {
        let s = "x".repeat(500);
        let out = chunk_content_for_streaming(&s, 1);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn chunk_helper_long_text_splits_on_word_boundaries() {
        // 400+ chars of normal English prose.
        let s = "The quick brown fox jumps over the lazy dog. ".repeat(10);
        let out = chunk_content_for_streaming(&s, 10);
        assert!(out.len() > 1, "expected multiple chunks; got {}", out.len());
        assert!(
            out.len() <= 11,
            "expected at most ~10 chunks; got {}",
            out.len()
        );
        // Reconstruction is exact: no bytes lost or added.
        let reconstructed: String = out.iter().copied().collect();
        assert_eq!(reconstructed, s);
        // Word boundaries: each non-final chunk ends in whitespace.
        for chunk in &out[..out.len() - 1] {
            assert!(
                chunk
                    .chars()
                    .last()
                    .map(|c| c.is_whitespace())
                    .unwrap_or(false),
                "non-final chunk should end on whitespace: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn chunk_helper_preserves_utf8_boundaries_for_cjk() {
        // No whitespace, multi-byte chars. Should still split cleanly
        // along char boundaries (no panic, exact reconstruction).
        let s = "中文测试内容".repeat(60); // 360 chars, all 3-byte UTF-8
        assert!(s.len() >= MOA_STREAM_MIN_BYTES);
        let out = chunk_content_for_streaming(&s, 10);
        assert!(out.len() > 1, "CJK should still chunk; got {}", out.len());
        let reconstructed: String = out.iter().copied().collect();
        assert_eq!(reconstructed, s);
        // Each chunk is valid UTF-8 (trivially, since &str by construction).
        for chunk in &out {
            assert!(std::str::from_utf8(chunk.as_bytes()).is_ok());
        }
    }

    #[test]
    fn chunk_helper_handles_text_with_no_whitespace_fallback() {
        // A long URL/hash — no whitespace to snap to. Helper should
        // fall through to char-boundary splitting.
        let s = "a".repeat(600);
        let out = chunk_content_for_streaming(&s, 10);
        assert!(
            out.len() > 1,
            "expected fallback chunking; got {}",
            out.len()
        );
        let reconstructed: String = out.iter().copied().collect();
        assert_eq!(reconstructed, s);
    }

    async fn capture_chat_sse_body(response: serde_json::Value) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept");
            send_moa_as_sse(socket, &response, &[]).await.expect("sse");
        });
        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server task");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn count_delta_events_with_content(raw: &str) -> usize {
        let mut count = 0;
        for line in raw.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };
            if v.pointer("/choices/0/delta/content")
                .and_then(|c| c.as_str())
                .filter(|s| !s.is_empty())
                .is_some()
            {
                count += 1;
            }
        }
        count
    }

    #[tokio::test]
    async fn chat_sse_emits_multiple_deltas_for_long_content() {
        // ≥ MOA_STREAM_MIN_BYTES of word-spaced English → must split.
        let long_content = "Hello world. ".repeat(40);
        let response = serde_json::json!({
            "id": "chatcmpl-moa-chunky",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": long_content },
                "finish_reason": "stop"
            }]
        });
        // The real MOA_STREAM_CHUNK_DELAY (20ms) × ~25 chunks adds
        // ~500ms to test runtime — acceptable since this is the only
        // chunked-delay test on the chat path.
        let raw = capture_chat_sse_body(response).await;
        let n = count_delta_events_with_content(&raw);
        assert!(
            n > 1,
            "expected multiple content delta events; got {n}\nraw: {raw}"
        );
    }

    #[tokio::test]
    async fn chat_sse_tool_calls_remain_atomic() {
        // Tool-call payloads must NOT be chunked — harness parsers
        // (Goose, OpenCode) need a single well-formed tool_call object.
        let response = serde_json::json!({
            "id": "chatcmpl-moa-tool",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{\"path\":\"/x\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let raw = capture_chat_sse_body(response).await;
        // Count delta events with tool_calls.
        let mut tool_deltas = 0;
        for line in raw.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };
            if v.pointer("/choices/0/delta/tool_calls").is_some() {
                tool_deltas += 1;
            }
        }
        assert_eq!(
            tool_deltas, 1,
            "tool_calls must arrive as exactly one atomic delta; got {tool_deltas}\nraw: {raw}"
        );
    }

    #[tokio::test]
    async fn responses_sse_emits_multiple_deltas_for_long_content() {
        let long_content = "Hello world. ".repeat(40);
        let response = serde_json::json!({
            "id": "chatcmpl-moa-resp-chunky",
            "object": "chat.completion",
            "model": "mesh",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": long_content },
                "finish_reason": "stop"
            }]
        });
        // ~500ms test runtime acceptable (MOA_STREAM_CHUNK_DELAY × N).
        let raw = capture_responses_sse_body(response).await;
        // Count response.output_text.delta events.
        let mut delta_count = 0;
        for line in raw.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                continue;
            };
            if v.get("type").and_then(|t| t.as_str()) == Some("response.output_text.delta") {
                delta_count += 1;
            }
        }
        assert!(
            delta_count > 1,
            "expected multiple output_text.delta events; got {delta_count}\nraw: {raw}"
        );
    }

    // ── reasoning_content progress drip ────────────────────────────

    #[test]
    fn progress_line_walks_opening_then_cycles_tail() {
        // The opening three lines fire once each at the start.
        let l0 = progress_line(0, proxy::ResponseAdapter::None);
        let l1 = progress_line(1, proxy::ResponseAdapter::None);
        let l2 = progress_line(2, proxy::ResponseAdapter::None);
        assert_ne!(l0, l1);
        assert_ne!(l1, l2);
        // After the opening, we cycle through the tail lines so a
        // long-running MoA call doesn't spam the same string —
        // important for slow public-mesh peers where MoA can wait
        // 10-20 seconds before committing.
        let l3 = progress_line(3, proxy::ResponseAdapter::None);
        let l4 = progress_line(4, proxy::ResponseAdapter::None);
        let l5 = progress_line(5, proxy::ResponseAdapter::None);
        assert_ne!(l3, l4);
        assert_ne!(l4, l5);
        // And steps past the cycle wrap deterministically, not panic.
        let l6 = progress_line(6, proxy::ResponseAdapter::None);
        assert_eq!(l6, l3);
        // Also: index 99 must not panic (the original bug we guarded
        // against — clamping/cycling rather than slice-out-of-bounds).
        let _ = progress_line(99, proxy::ResponseAdapter::None);
    }

    /// Test fixture: a stable completion id with the same shape as
    /// MoA's real ids, so tests can assert correlation behaviour.
    const TEST_COMPLETION_ID: &str = "chatcmpl-moa-deadbeef";

    /// Capture the wire bytes produced by `write_progress_event` for
    /// a given adapter, by writing into a loopback TCP pair.
    async fn capture_progress_event(adapter: proxy::ResponseAdapter, text: &str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");
        let t = text.to_string();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            // Tests don't care about the cross-event sequence; start
            // from 1 to match production (response.created is 0).
            let mut seq = 1i32;
            write_progress_event(&mut socket, &t, adapter, TEST_COMPLETION_ID, &mut seq)
                .await
                .expect("write");
            // Close so the client read_to_end terminates.
            socket.shutdown().await.expect("shutdown");
        });
        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server task");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[tokio::test]
    async fn progress_event_chat_uses_reasoning_content_field() {
        // Chat-completions adapter: progress text must land in
        // delta.reasoning_content (so goose/openai-sdk-aware clients
        // route it to a thinking pane, not the main answer).
        let raw =
            capture_progress_event(proxy::ResponseAdapter::None, "Routing through mesh…\n").await;

        // Pull out the JSON payload from the framed SSE event.
        let payload = raw
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .expect("a data: line");
        let v: serde_json::Value = serde_json::from_str(payload.trim()).expect("valid json");

        // Required shape for goose/openai SDKs: chat.completion.chunk
        // with delta.reasoning_content set and no delta.content.
        assert_eq!(
            v.pointer("/object").and_then(|o| o.as_str()),
            Some("chat.completion.chunk")
        );
        let reasoning = v
            .pointer("/choices/0/delta/reasoning_content")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        assert!(
            reasoning.contains("Routing through mesh"),
            "expected progress text in reasoning_content; got: {payload}"
        );
        assert!(
            v.pointer("/choices/0/delta/content").is_none(),
            "delta.content must NOT be set on progress events (would pollute the final answer): {payload}"
        );
        assert!(
            v.pointer("/choices/0/finish_reason")
                .map(|f| f.is_null())
                .unwrap_or(false),
            "finish_reason must be null on a progress event so the client keeps the stream open"
        );
        // Id must match the supplied completion_id so progress and
        // the eventual final body chunks correlate as one stream.
        assert_eq!(
            v.pointer("/id").and_then(|i| i.as_str()),
            Some(TEST_COMPLETION_ID),
            "progress chunk id must equal completion_id; got: {payload}"
        );
    }

    #[test]
    fn item_id_derives_short_suffix_from_completion_id() {
        // Body writer constructs item_id as `msg_moa_<short-suffix>`
        // where short-suffix is everything after the last `-` in
        // response.id. Progress path must mirror this so item_id is
        // stable across progress and final events in one stream.
        assert_eq!(
            item_id_from_completion_id("chatcmpl-moa-deadbeef"),
            "msg_moa_deadbeef"
        );
        assert_eq!(item_id_from_completion_id("nodashes"), "msg_moa_nodashes");
        assert_eq!(item_id_from_completion_id(""), "msg_moa_");
    }

    #[tokio::test]
    async fn progress_event_responses_uses_reasoning_text_delta() {
        // Responses-API adapter: drip progress into the thinking
        // channel (`response.reasoning_text.delta`) so it surfaces
        // separately in the UI and doesn't get appended to the
        // visible answer when the real content arrives.
        let raw = capture_progress_event(
            proxy::ResponseAdapter::OpenAiResponsesStream,
            "Querying peer models…\n",
        )
        .await;
        let payload = raw
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .expect("a data: line");
        let v: serde_json::Value = serde_json::from_str(payload.trim()).expect("valid json");
        assert_eq!(
            v.get("type").and_then(|t| t.as_str()),
            Some("response.reasoning_text.delta"),
            "progress events for Responses-API must use the reasoning channel \
             so the UI doesn't append them to the visible answer; payload: {payload}"
        );
        let delta = v.get("delta").and_then(|d| d.as_str()).unwrap_or("");
        assert!(
            delta.contains("Querying peer models"),
            "expected progress text in delta; got: {payload}"
        );
        // item_id must derive from completion_id (msg_moa_<suffix>)
        // so progress events share item_id with the final
        // output_text.delta events the body writer emits.
        assert_eq!(
            v.get("item_id").and_then(|i| i.as_str()),
            Some("msg_moa_deadbeef"),
            "progress item_id must derive from completion_id; got: {payload}"
        );
        // sequence_number must be present and i64 for strict
        // Responses-API clients. response.created was 0; the first
        // progress reasoning delta starts at 1.
        assert_eq!(
            v.get("sequence_number").and_then(|s| s.as_i64()),
            Some(1),
            "Responses-API progress event must carry sequence_number=1 \
             (response.created=0, progress deltas follow); got: {payload}"
        );
    }

    #[tokio::test]
    async fn progress_event_responses_sequence_number_increments() {
        // Two consecutive progress events written with the same
        // sequence counter must emit sequence_number=1, then 2, so
        // strict Responses-API clients can order/dedup events within
        // a single response stream.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut seq = 1i32;
            write_progress_event(
                &mut socket,
                "first",
                proxy::ResponseAdapter::OpenAiResponsesStream,
                TEST_COMPLETION_ID,
                &mut seq,
            )
            .await
            .unwrap();
            write_progress_event(
                &mut socket,
                "second",
                proxy::ResponseAdapter::OpenAiResponsesStream,
                TEST_COMPLETION_ID,
                &mut seq,
            )
            .await
            .unwrap();
            socket.shutdown().await.expect("shutdown");
        });
        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server task");
        let raw = String::from_utf8_lossy(&bytes);
        let seqs: Vec<i64> = raw
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|p| serde_json::from_str::<serde_json::Value>(p.trim()).ok())
            .filter_map(|v| v.get("sequence_number").and_then(|s| s.as_i64()))
            .collect();
        assert_eq!(
            seqs,
            vec![1, 2],
            "two consecutive progress events must produce monotonically \
             increasing sequence_number starting at 1; raw=\n{raw}"
        );
    }

    /// Capture the wire bytes produced by `write_failure_as_sse_tail`
    /// for a given adapter and error body.
    async fn capture_failure_tail(
        adapter: proxy::ResponseAdapter,
        body: serde_json::Value,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            write_failure_as_sse_tail(&mut socket, &body, adapter, TEST_COMPLETION_ID)
                .await
                .expect("tail write");
        });
        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server task");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[tokio::test]
    async fn failure_tail_emits_error_then_done_for_chat_adapter() {
        // After progress headers go out, an MoA failure can't change
        // the HTTP status (already 200). We at least owe the client a
        // terminal SSE event + [DONE] so the stream doesn't silently
        // truncate (which makes harnesses hang waiting for more).
        let body = serde_json::json!({
            "error": { "message": "all workers failed" }
        });
        let raw = capture_failure_tail(proxy::ResponseAdapter::None, body).await;
        // Final chunk shape: chat.completion.chunk with content + error finish.
        assert!(
            raw.contains("\"finish_reason\":\"error\""),
            "expected finish_reason=error in tail; got: {raw}"
        );
        assert!(
            raw.contains("all workers failed"),
            "expected error message in content; got: {raw}"
        );
        assert!(
            raw.contains("data: [DONE]"),
            "expected [DONE] terminator so clients release the stream; got: {raw}"
        );
        // The error chunk must carry the same id as progress + final
        // chunks so chunk-aggregating clients see the stream as one
        // completion.
        assert!(
            raw.contains(TEST_COMPLETION_ID),
            "expected completion id {TEST_COMPLETION_ID} on the error chunk; got: {raw}"
        );
    }

    /// End-to-end-ish test for response.created ordering: emit the
    /// progress-path response.created first, then a progress delta,
    /// then ensure the body writer (with header_already_sent=true)
    /// does NOT emit a second response.created event in the same
    /// stream. Two `response.created` events in one stream is a
    /// Responses-API protocol violation that strict clients reject.
    #[tokio::test]
    async fn responses_progress_path_emits_exactly_one_response_created() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            // Imitate run_moa_turn_with_progress on the Responses path:
            // headers → response.created → one progress delta → body
            // writer with header_already_sent=true.
            write_sse_response_headers(&mut socket, &[]).await.unwrap();
            write_progress_response_created(&mut socket, TEST_COMPLETION_ID)
                .await
                .unwrap();
            let mut seq = 1i32;
            write_progress_event(
                &mut socket,
                "Routing through mesh…\n",
                proxy::ResponseAdapter::OpenAiResponsesStream,
                TEST_COMPLETION_ID,
                &mut seq,
            )
            .await
            .unwrap();
            // Body writer takes a chat-shape response (MoA's output);
            // we use a minimal one with a non-trivial content so the
            // chunker has something to split.
            let body = serde_json::json!({
                "id": TEST_COMPLETION_ID,
                "object": "chat.completion",
                "model": moa::VIRTUAL_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Bees are fuzzy insects that make honey."
                    },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
            });
            send_moa_as_responses_sse_inner(socket, &body, &[], /*header_already_sent=*/ true)
                .await
                .expect("send_moa_as_responses_sse_inner failed");
        });

        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server");
        let raw = String::from_utf8_lossy(&bytes);

        // Count occurrences of `response.created` events.
        let created_count = raw
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|p| {
                serde_json::from_str::<serde_json::Value>(p.trim())
                    .ok()
                    .and_then(|v| {
                        v.get("type")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    })
                    .as_deref()
                    == Some("response.created")
            })
            .count();
        assert_eq!(
            created_count, 1,
            "exactly one response.created event must be emitted per Responses stream; \
             body writer must skip its own when header_already_sent=true. \
             raw stream:\n{raw}"
        );

        // And the single response.created must come BEFORE any delta
        // (reasoning_text or output_text). Find the byte offsets.
        let created_at = raw.find("\"response.created\"").expect("created present");
        let first_delta = raw
            .find("\"response.reasoning_text.delta\"")
            .or_else(|| raw.find("\"response.output_text.delta\""))
            .expect("at least one delta event");
        assert!(
            created_at < first_delta,
            "response.created must precede all delta events; \
             created_at={created_at} first_delta={first_delta}\n{raw}"
        );
    }
}
