//! Streaming MoA progress drip: heartbeat-style `reasoning_content` /
//! `response.reasoning_text.delta` events while the arbiter is still
//! waiting, plus the body-write hand-off once MoA commits.
//!
//! Extracted from `moa_gateway` so the gateway entry point stays
//! focused on routing, scoring, and worker dispatch — and so the
//! 2,000-line file limit isn't blown by the streaming UX code that
//! has accreted around the live-feel improvements.
//!
//! See the parent module for the gateway entry, body writers
//! (`send_moa_as_*_sse_inner`), and the chunking helpers that this
//! module hands the final answer off to.

use super::streaming::{
    ProgressContinuation, final_text_stream_mode_for_result, is_moa_failure_body,
    send_moa_as_responses_sse_inner, send_moa_as_sse_inner,
};
use crate::network::openai::transport as proxy;
use mesh_mixture_of_agents as moa;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

/// Time between progress events while MoA's arbiter is still waiting.
/// One second feels alive without flooding the wire; the typical MoA
/// turn finishes in ~3s, so we emit two or three lines before the
/// real answer starts streaming.
const MOA_PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_millis(1000);

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
pub(super) async fn run_moa_turn_with_progress(
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

    let Some(progress_created_at) = send_progress_response_created_if_responses(
        &mut tcp_stream,
        &completion_id,
        response_adapter,
    )
    .await
    else {
        return;
    };

    let Some((moa_result, continuation)) = drip_progress_phase(
        &mut tcp_stream,
        config,
        &moa_body,
        response_adapter,
        &completion_id,
        progress_created_at,
    )
    .await
    else {
        return;
    };

    if let Err(e) = write_progress_body(
        tcp_stream,
        &moa_result,
        response_adapter,
        &completion_id,
        continuation,
    )
    .await
    {
        tracing::warn!("MoA progress: body write failed: {e}");
    }
}

/// Run the progress phase end-to-end: drip heartbeat lines until MoA
/// finishes, then rewrite the body's id to match `completion_id` and
/// build the `ProgressContinuation` the body writer will consume.
/// Returns `None` if the client disconnected (caller must bail and let
/// the pinned MoA future drop, cancelling the work).
async fn drip_progress_phase(
    tcp_stream: &mut TcpStream,
    config: &moa::GatewayConfig,
    moa_body: &serde_json::Value,
    response_adapter: proxy::ResponseAdapter,
    completion_id: &str,
    progress_created_at: Option<i64>,
) -> Option<(moa::TurnResult, Option<ProgressContinuation>)> {
    let (mut moa_result, next_sequence_number) = match drip_progress_until_moa_completes(
        tcp_stream,
        config,
        moa_body,
        response_adapter,
        completion_id,
    )
    .await
    {
        Ok(pair) => pair,
        Err(ClientGone) => {
            // Returning None drops the pinned moa::handle_turn future
            // in our caller, which cancels worker dispatch / reducer
            // calls at their next `.await`. Avoids burning ~60s of
            // peer compute for a dead request.
            tracing::info!("MoA progress: client disconnected mid-progress; cancelling MoA turn");
            return None;
        }
    };
    overwrite_response_id(&mut moa_result.response_body, completion_id);
    let continuation = progress_created_at.map(|created_at| ProgressContinuation {
        created_at,
        next_sequence_number,
    });
    Some((moa_result, continuation))
}

/// Emit the early Responses-API `response.created` event when the
/// adapter requires it. Returns:
///   - `Some(Some(created_at))` — Responses path, event written.
///   - `Some(None)` — Chat-completions path, no event needed.
///   - `None` — write failed; caller should bail out.
///
/// The `Option<i64>` lets the caller thread the timestamp into the
/// body writer so `response.completed` matches `response.created`.
async fn send_progress_response_created_if_responses(
    stream: &mut TcpStream,
    completion_id: &str,
    adapter: proxy::ResponseAdapter,
) -> Option<Option<i64>> {
    if adapter != proxy::ResponseAdapter::OpenAiResponsesStream {
        return Some(None);
    }
    // Responses-API contract is strict: created → deltas → completed.
    // Some clients reject a stream that starts with a delta.
    let created_at = send_responses_created_for_progress(stream, completion_id).await?;
    Some(Some(created_at))
}

/// Force the body's `id` to match the progress phase's completion id
/// so clients see a single id across progress chunks and the final
/// body. Without this MoA's own id (created independently) would
/// appear on the body and clients would treat the progress and
/// content as belonging to different completions.
fn overwrite_response_id(body: &mut serde_json::Value, completion_id: &str) {
    if let Some(obj) = body.as_object_mut() {
        obj.insert(
            "id".to_string(),
            serde_json::Value::String(completion_id.to_string()),
        );
    }
}

/// Marker for "client disconnected mid-progress; cancel MoA work".
#[derive(Debug, Clone, Copy)]
struct ClientGone;

/// Match MoA's `short_id()` — hex of nanos since epoch. Locally
/// derived so we don't have to expose the internal helper.
fn short_hex_nanos() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}")
}

/// Wrapper: emit `response.created` and log on failure. Returns the
/// `created_at` baked into the event so the caller can thread it to
/// the body writer (so `response.completed` carries the same value).
/// Returns `None` if the connection died so the caller can bail.
async fn send_responses_created_for_progress(
    stream: &mut TcpStream,
    completion_id: &str,
) -> Option<i64> {
    match write_progress_response_created(stream, completion_id).await {
        Ok(created_at) => Some(created_at),
        Err(e) => {
            tracing::warn!("MoA progress: response.created write failed: {e}");
            None
        }
    }
}

/// Emit `response.created` early on the progress path so the
/// Responses-API stream stays in the required order (`created` first,
/// then any `reasoning_text.delta` progress, then the real
/// `output_text.delta` content from the body writer). Returns the
/// `created_at` value written so the caller can thread it onward.
async fn write_progress_response_created(
    stream: &mut TcpStream,
    completion_id: &str,
) -> std::io::Result<i64> {
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
    stream.flush().await?;
    Ok(created_at)
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
) -> Result<(moa::TurnResult, i32), ClientGone> {
    let moa_fut = moa::handle_turn(config, moa_body);
    drip_progress_against_future(
        stream,
        moa_fut,
        adapter,
        completion_id,
        MOA_PROGRESS_INTERVAL,
    )
    .await
}

/// Generic core of `drip_progress_until_moa_completes`: race a
/// caller-supplied future producing `TurnResult` against a progress
/// ticker, writing one progress event per tick. Extracted so tests
/// can substitute a hand-rolled future (e.g., a never-finishing one
/// to verify drop-on-client-disconnect cancellation behaviour)
/// without spinning up a real MoA gateway.
async fn drip_progress_against_future<F>(
    stream: &mut TcpStream,
    moa_fut: F,
    adapter: proxy::ResponseAdapter,
    completion_id: &str,
    tick_interval: std::time::Duration,
) -> Result<(moa::TurnResult, i32), ClientGone>
where
    F: std::future::Future<Output = moa::TurnResult>,
{
    tokio::pin!(moa_fut);
    let mut ticker = tokio::time::interval(tick_interval);
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
            r = &mut moa_fut => return Ok((r, sequence_number)),
            _ = ticker.tick() => {}
        }
        let text = progress_line(step, adapter);
        step += 1;
        if let Err(e) =
            write_progress_event(stream, &text, adapter, completion_id, &mut sequence_number).await
        {
            // Progress write failed — almost always means the client
            // closed the connection. Returning Err here drops the
            // pinned MoA future, cancelling worker dispatch and any
            // reducer call mid-flight at their next .await.
            // Previously we awaited the future to completion, burning
            // peer compute (~60s of timeouts) for a dead request.
            tracing::warn!(
                "MoA progress: tick write failed: {e}; client likely gone, cancelling MoA turn"
            );
            return Err(ClientGone);
        }
    }
}

/// After MoA completes, write the final body — either the real
/// streamed answer or a graceful error tail if MoA failed (we
/// already sent 200 OK, so we can't change the HTTP status).
///
/// `continuation` is `Some` only on the Responses-API progress path
/// and carries the running sequence_number + the original created_at
/// so the body writer can emit a wire-monotonic stream.
async fn write_progress_body(
    mut tcp_stream: TcpStream,
    moa_result: &moa::TurnResult,
    adapter: proxy::ResponseAdapter,
    completion_id: &str,
    continuation: Option<ProgressContinuation>,
) -> std::io::Result<()> {
    let body = &moa_result.response_body;
    if is_moa_failure_body(body) {
        return write_failure_as_sse_tail(&mut tcp_stream, body, adapter, completion_id).await;
    }
    let text_stream_mode = final_text_stream_mode_for_result(moa_result);
    match adapter {
        proxy::ResponseAdapter::OpenAiResponsesStream => {
            send_moa_as_responses_sse_inner(
                tcp_stream,
                body,
                &[],
                true,
                text_stream_mode,
                continuation,
            )
            .await
        }
        _ => send_moa_as_sse_inner(tcp_stream, body, &[], true, text_stream_mode).await,
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

#[cfg(test)]
mod tests {
    use super::super::streaming::MoaFinalTextStreamMode;
    use super::*;

    /// Test fixture: a stable completion id with the same shape as
    /// MoA's real ids, so tests can assert correlation behaviour.
    const TEST_COMPLETION_ID: &str = "chatcmpl-moa-deadbeef";

    #[test]
    fn progress_line_walks_opening_then_cycles_tail() {
        // First 3 lines are the opening; we never want to repeat one
        // of those within the first three ticks.
        let a = progress_line(0, proxy::ResponseAdapter::None);
        let b = progress_line(1, proxy::ResponseAdapter::None);
        let c = progress_line(2, proxy::ResponseAdapter::None);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        // After the opening, the tail cycles so we never go silent.
        let d = progress_line(3, proxy::ResponseAdapter::None);
        let e = progress_line(4, proxy::ResponseAdapter::None);
        let f = progress_line(5, proxy::ResponseAdapter::None);
        let g = progress_line(6, proxy::ResponseAdapter::None);
        assert_ne!(d, e);
        assert_ne!(e, f);
        // step=6 is one full TAIL_CYCLE past step=3 → must be equal.
        assert_eq!(d, g, "tail must cycle so a slow MoA turn keeps printing");
    }

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
        let payload = raw
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .expect("a data: line");
        let v: serde_json::Value = serde_json::from_str(payload.trim()).expect("valid json");
        assert_eq!(
            v.pointer("/choices/0/delta/reasoning_content")
                .and_then(|s| s.as_str()),
            Some("Routing through mesh…\n"),
            "progress events for chat adapter must drip into \
             delta.reasoning_content so they don't pollute the answer; payload: {payload}"
        );
        assert!(
            v.pointer("/choices/0/delta/content").is_none(),
            "must NOT emit visible content for progress chunks; payload: {payload}"
        );
        // Completion id correlation: progress and final body chunks
        // share the same id so clients aggregate them as one
        // chat.completion stream.
        assert_eq!(
            v.get("id").and_then(|i| i.as_str()),
            Some(TEST_COMPLETION_ID),
            "progress chunks must reuse completion_id so clients correlate the stream; \
             payload: {payload}"
        );
    }

    #[test]
    fn item_id_derives_short_suffix_from_completion_id() {
        // Body writer constructs item_id as `msg_moa_<short-suffix>`
        // where short suffix is the part after the last `-` of the
        // response.id. Progress path must mirror this so item_id is
        // consistent across the whole Responses stream.
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

    /// Capture the wire bytes from the failure-tail SSE writer for a
    /// given adapter. Used to assert the [DONE] terminator and the
    /// completion-id correlation invariant.
    async fn capture_failure_tail(
        adapter: proxy::ResponseAdapter,
        body: serde_json::Value,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            write_failure_as_sse_tail(&mut socket, &body, adapter, TEST_COMPLETION_ID)
                .await
                .expect("write");
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
    async fn failure_tail_emits_error_then_done_for_chat_adapter() {
        // After progress headers are sent we can't change HTTP status,
        // so MoA failure must surface as an in-band error chunk
        // followed by [DONE]. Clients that consume chat.completion
        // chunks rely on [DONE] to close the stream.
        let body = serde_json::json!({
            "error": { "message": "All workers failed", "code": "all_workers_failed" }
        });
        let raw = capture_failure_tail(proxy::ResponseAdapter::None, body).await;
        assert!(
            raw.contains("\"finish_reason\":\"error\""),
            "failure tail must set finish_reason=error; got: {raw}"
        );
        assert!(
            raw.contains("[DONE]"),
            "failure tail must terminate the SSE stream with [DONE]; got: {raw}"
        );
        // Completion id correlation: the failure chunk must share
        // the id used by any earlier progress chunks in the same
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
            super::super::streaming::write_sse_response_headers(&mut socket, &[])
                .await
                .unwrap();
            let created_at = write_progress_response_created(&mut socket, TEST_COMPLETION_ID)
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
            let continuation = Some(ProgressContinuation {
                created_at,
                next_sequence_number: seq,
            });
            send_moa_as_responses_sse_inner(
                socket,
                &body,
                &[],
                /*header_already_sent=*/ true,
                MoaFinalTextStreamMode::ChunkedCommittedText,
                continuation,
            )
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

    /// Stream-wide invariants for the Responses-API progress path:
    ///   1. sequence_number is strictly monotonically increasing
    ///      across created → progress deltas → output deltas →
    ///      text_done → completed (no resets, no duplicates).
    ///   2. response.created and response.completed carry the SAME
    ///      created_at — strict Responses clients use it to
    ///      correlate the response object across events.
    #[tokio::test]
    async fn responses_progress_path_emits_monotonic_sequence_and_stable_created_at() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            super::super::streaming::write_sse_response_headers(&mut socket, &[])
                .await
                .unwrap();
            let created_at = write_progress_response_created(&mut socket, TEST_COMPLETION_ID)
                .await
                .unwrap();
            let mut seq = 1i32;
            // Two progress events so we can verify seq increments
            // both within the progress phase AND across the
            // progress → body boundary.
            write_progress_event(
                &mut socket,
                "first\n",
                proxy::ResponseAdapter::OpenAiResponsesStream,
                TEST_COMPLETION_ID,
                &mut seq,
            )
            .await
            .unwrap();
            write_progress_event(
                &mut socket,
                "second\n",
                proxy::ResponseAdapter::OpenAiResponsesStream,
                TEST_COMPLETION_ID,
                &mut seq,
            )
            .await
            .unwrap();
            // MoA-shape body with enough content to produce >1
            // output_text.delta from the chunker.
            let body = serde_json::json!({
                "id": TEST_COMPLETION_ID,
                "object": "chat.completion",
                "model": moa::VIRTUAL_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Bees are fuzzy insects that produce honey by visiting many \
                                    different flowers, which is also why they help pollinate plants \
                                    and keep ecosystems healthy across temperate climates."
                    },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 30, "total_tokens": 31 }
            });
            let continuation = Some(ProgressContinuation {
                created_at,
                next_sequence_number: seq,
            });
            send_moa_as_responses_sse_inner(
                socket,
                &body,
                &[],
                true,
                MoaFinalTextStreamMode::ChunkedCommittedText,
                continuation,
            )
            .await
            .expect("send_moa_as_responses_sse_inner failed");
        });

        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        use tokio::io::AsyncReadExt;
        let mut bytes = Vec::new();
        client.read_to_end(&mut bytes).await.expect("read");
        server.await.expect("server");
        let raw = String::from_utf8_lossy(&bytes);

        // Parse all SSE events and collect sequence_number + the
        // response.created/completed created_at values.
        let mut seqs: Vec<i64> = Vec::new();
        let mut created_at_in_created: Option<i64> = None;
        let mut created_at_in_completed: Option<i64> = None;
        for line in raw.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(payload.trim()) else {
                continue;
            };
            if let Some(s) = v.get("sequence_number").and_then(|s| s.as_i64()) {
                seqs.push(s);
            }
            let ev_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if ev_type == "response.created" {
                created_at_in_created = v.pointer("/response/created_at").and_then(|t| t.as_i64());
            }
            if ev_type == "response.completed" {
                created_at_in_completed =
                    v.pointer("/response/created_at").and_then(|t| t.as_i64());
            }
        }

        // 1. Sequence is strictly monotonically increasing with no
        //    duplicates and no resets.
        assert!(
            !seqs.is_empty(),
            "expected sequence_numbers on the wire; raw=\n{raw}"
        );
        let mut sorted = seqs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            seqs, sorted,
            "sequence_numbers must be strictly monotonic with no duplicates; \
             observed={seqs:?}; raw=\n{raw}"
        );

        // 2. response.created and response.completed share created_at.
        let c1 = created_at_in_created.expect("response.created carries created_at");
        let c2 = created_at_in_completed.expect("response.completed carries created_at");
        assert_eq!(
            c1, c2,
            "response.created.created_at ({c1}) must equal \
             response.completed.created_at ({c2}) for the same response stream; \
             raw=\n{raw}"
        );
    }

    /// A pending future that records whether it was dropped without
    /// completing. Used to verify drip_progress_against_future
    /// cancels (drops) the MoA work when the client disconnects
    /// rather than awaiting it to completion.
    struct DropTrackingPendingFuture {
        dropped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl std::future::Future for DropTrackingPendingFuture {
        type Output = moa::TurnResult;
        fn poll(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            std::task::Poll::Pending
        }
    }

    impl Drop for DropTrackingPendingFuture {
        fn drop(&mut self) {
            self.dropped
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// When a progress write fails (client disconnected), the
    /// in-flight MoA future must be DROPPED (cancelled at the next
    /// .await point), not awaited to completion. Awaiting would burn
    /// peer compute and reducer budget for ~60s on a request whose
    /// client is already gone.
    #[tokio::test(start_paused = true)]
    async fn progress_drops_moa_future_when_client_disconnects() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");

        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_for_task = dropped.clone();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            // Build a pending MoA future that flags itself when dropped.
            let pending = DropTrackingPendingFuture {
                dropped: dropped_for_task,
            };
            // Use a small tick interval so the first progress write
            // happens quickly under tokio's paused clock advance.
            let result = drip_progress_against_future(
                &mut socket,
                pending,
                proxy::ResponseAdapter::OpenAiResponsesStream,
                TEST_COMPLETION_ID,
                std::time::Duration::from_millis(50),
            )
            .await;
            // Expect ClientGone (write failure on closed socket).
            assert!(
                matches!(result, Err(ClientGone)),
                "expected ClientGone after socket drop, got Ok(..)"
            );
        });

        // Connect, then drop the client so subsequent writes fail.
        let client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        drop(client);

        // Advance paused clock past one tick so the ticker fires and
        // the server attempts a progress write, which will fail and
        // cause drip_progress_against_future to return ClientGone.
        // The pending MoA future must be dropped as the function
        // returns (it's pinned on the stack of
        // drip_progress_against_future).
        tokio::time::advance(std::time::Duration::from_millis(200)).await;
        server.await.expect("server task");

        assert!(
            dropped.load(Ordering::SeqCst),
            "MoA future must be dropped (cancelled) when the client disconnects, \
             not awaited to completion"
        );
    }
}
