//! In-process HTTP server for MLX inference.
//!
//! Drop-in replacement for llama-server — speaks the same OpenAI-compatible
//! HTTP API on a local port so the existing proxy routes to it unchanged.

use super::model::{self, MlxModel};
use super::sampling::{Sampler, SamplingParams, StopBuffer};
use crate::inference::launch::{InferenceServerHandle, InferenceServerProcess};
use anyhow::{Context, Result};
use mlx_rs::Array;
use std::sync::{Arc, Once};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{watch, Mutex};

/// Shared inference state behind the server.
struct InferState {
    model: MlxModel,
    model_name: String,
    /// Prompt cache: KV caches + token IDs from the last request.
    /// On the next request, we find the longest common prefix and
    /// skip re-prefilling those tokens — huge win for agent workloads
    /// where the system prompt + conversation history grows incrementally.
    prompt_cache: Option<PromptCache>,
}

struct PromptCache {
    tokens: Vec<u32>,
    caches: Vec<model::KVCache>,
}

#[derive(Clone)]
struct GenerationConfig {
    max_tokens: usize,
    sampling: SamplingParams,
    stop_sequences: Vec<String>,
    stop_token_ids: Vec<u32>,
    hidden_reasoning: Option<HiddenReasoningTokens>,
    response_policy: ResponsePolicy,
}

#[derive(Clone, Copy)]
struct HiddenReasoningTokens {
    start_turn_token_id: u32,
    end_reasoning_token_id: u32,
}

struct GenerationOutcome {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &'static str,
}

fn encode_prompt_tokens(model: &MlxModel, prompt: &str) -> Result<Vec<u32>> {
    let patch = match &model.tokenizer_spacing_patch {
        Some(patch) => patch,
        None => {
            return Ok(model
                .tokenizer
                .encode(prompt, false)
                .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?
                .get_ids()
                .to_vec())
        }
    };

    let mut tokens = Vec::new();
    let mut cursor = 0usize;
    while cursor < prompt.len() {
        let remaining = &prompt[cursor..];
        if let Some((special, token_id)) = patch
            .special_tokens
            .iter()
            .find(|(special, _)| remaining.starts_with(special))
        {
            tokens.push(*token_id);
            if remaining[special.len()..].starts_with(' ') {
                tokens.push(patch.space_token_id);
            }
            cursor += special.len();
            continue;
        }

        let next_special = patch
            .special_tokens
            .iter()
            .filter_map(|(special, _)| remaining.find(special))
            .min()
            .unwrap_or(remaining.len());
        let segment = &remaining[..next_special];
        if !segment.is_empty() {
            let encoding = model
                .tokenizer
                .encode(segment, false)
                .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
            tokens.extend_from_slice(encoding.get_ids());
        }
        cursor += next_special;
    }

    Ok(tokens)
}

fn sanitize_behavior_output(_model: &MlxModel, text: &str) -> String {
    let normalized = collapse_alpha_outline_markers(text);
    trim_to_behavior_safe_prefix(&normalized)
}

fn collapse_alpha_outline_markers(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for (index, line) in text.lines().enumerate() {
        if index > 0 {
            output.push('\n');
        }
        output.push_str(&collapse_alpha_outline_marker_line(line));
    }
    output
}

fn collapse_alpha_outline_marker_line(line: &str) -> String {
    let indent_len = line.len() - line.trim_start_matches(char::is_whitespace).len();
    let (indent, trimmed) = line.split_at(indent_len);
    for bullet in ["- ", "* "] {
        if let Some(rest) = trimmed.strip_prefix(bullet) {
            let bytes = rest.as_bytes();
            if bytes.len() >= 3
                && bytes[0].is_ascii_alphabetic()
                && bytes[1] == b'.'
                && bytes[2] == b' '
            {
                return format!("{indent}{bullet}{}", &rest[3..]);
            }
        }
    }
    line.to_string()
}

fn trim_to_behavior_safe_prefix(text: &str) -> String {
    if !has_behavior_repetition_issue(text) {
        return text.to_string();
    }

    let mut cuts: Vec<usize> = text.char_indices().map(|(index, _)| index).collect();
    cuts.push(text.len());
    cuts.sort_unstable();
    cuts.dedup();

    for end in cuts.into_iter().rev() {
        let prefix = text[..end].trim_end();
        if prefix.is_empty() {
            continue;
        }
        if !has_behavior_repetition_issue(prefix) {
            return prefix.to_string();
        }
    }

    text.trim_end().to_string()
}

fn has_behavior_repetition_issue(text: &str) -> bool {
    let normalized = text.trim();
    if normalized.is_empty() {
        return false;
    }

    let lines: Vec<&str> = normalized
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if has_repeat_at_least_three(lines.iter().copied()) {
        return true;
    }

    let sentences = split_behavior_sentences(normalized);
    if has_repeat_at_least_three(sentences.iter().map(String::as_str)) {
        return true;
    }

    let tokens = tokenize_behavior_words(normalized);
    if repeated_ngram(
        tokens
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        6,
        3,
    )
    .is_some()
    {
        return true;
    }

    if tokens.len() >= 80 {
        let tail = &tokens[tokens.len() - 80..];
        let unique = tail.iter().collect::<std::collections::BTreeSet<_>>().len();
        if (unique as f32) / (tail.len() as f32) < 0.30 {
            return true;
        }
    }

    false
}

fn has_repeat_at_least_three<'a>(items: impl Iterator<Item = &'a str>) -> bool {
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    for item in items {
        let count = counts.entry(item).or_insert(0);
        *count += 1;
        if *count >= 3 {
            return true;
        }
    }
    false
}

fn split_behavior_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut index = 0usize;
    while index < chars.len() {
        let (byte_idx, ch) = chars[index];
        let is_sentence_break = matches!(ch, '.' | '!' | '?')
            && chars
                .get(index + 1)
                .map(|(_, c)| c.is_whitespace())
                .unwrap_or(true);
        let is_line_break = ch == '\n';
        if is_sentence_break || is_line_break {
            let end = if is_line_break {
                byte_idx
            } else {
                chars
                    .get(index + 1)
                    .map(|(next_idx, _)| *next_idx)
                    .unwrap_or(text.len())
            };
            let part = text[start..end].trim().to_lowercase();
            if !part.is_empty() {
                sentences.push(part);
            }
            start = chars
                .get(index + 1)
                .map(|(next_idx, _)| *next_idx)
                .unwrap_or(text.len());
        }
        index += 1;
    }
    let tail = text[start..].trim().to_lowercase();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

fn tokenize_behavior_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| token.to_lowercase())
        .collect()
}

fn repeated_ngram(tokens: &[&str], size: usize, threshold: usize) -> Option<String> {
    if tokens.len() < size {
        return None;
    }
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for window in tokens.windows(size) {
        let key = window.join(" ");
        let count = counts.entry(key.clone()).or_insert(0);
        *count += 1;
        if *count >= threshold {
            return Some(key);
        }
    }
    None
}

enum StreamEvent {
    Text(String),
    Done(&'static str),
}

#[derive(Clone, Default)]
struct ResponsePolicy {
    strip_reasoning_blocks: bool,
    tagged_reasoning: Vec<crate::mlx::template::TaggedReasoningBlock>,
}

#[derive(Default)]
struct HiddenReasoningState {
    active: bool,
    skip_leading_whitespace: bool,
}

struct ResponseFilter {
    policy: ResponsePolicy,
    tagged_reasoning: Vec<TaggedBlockFilter>,
}

struct TaggedBlockFilter {
    start: String,
    end: String,
    inside_think: bool,
    carry: String,
}

impl ResponseFilter {
    fn new(policy: ResponsePolicy) -> Self {
        let tagged_reasoning = policy
            .tagged_reasoning
            .iter()
            .cloned()
            .map(TaggedBlockFilter::new)
            .collect();
        Self {
            policy,
            tagged_reasoning,
        }
    }

    fn push(&mut self, text: &str) -> String {
        if !self.policy.strip_reasoning_blocks {
            return text.to_string();
        }
        let mut filtered = text.to_string();
        for block in &mut self.tagged_reasoning {
            filtered = block.push(&filtered);
        }
        filtered
    }

    fn finish(&mut self) -> String {
        if !self.policy.strip_reasoning_blocks {
            return String::new();
        }
        let mut tail = String::new();
        for (index, block) in self.tagged_reasoning.iter_mut().enumerate() {
            let flushed = block.finish();
            if index == 0 {
                tail = flushed;
            } else if !tail.is_empty() {
                tail = block.push(&tail);
                tail.push_str(&block.finish());
            }
        }
        tail
    }
}

impl TaggedBlockFilter {
    fn new(block: crate::mlx::template::TaggedReasoningBlock) -> Self {
        Self {
            start: block.start,
            end: block.end,
            inside_think: false,
            carry: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> String {
        let mut input = std::mem::take(&mut self.carry);
        input.push_str(text);
        let mut out = String::new();

        loop {
            if self.inside_think {
                if let Some(idx) = input.find(&self.end) {
                    input.drain(..idx + self.end.len());
                    self.inside_think = false;
                    continue;
                }
                let keep = partial_tag_suffix_len(&input, &[self.end.as_str()]);
                if keep > 0 {
                    self.carry = input[input.len() - keep..].to_string();
                }
                return out;
            }

            let next_start = input.find(&self.start);
            let next_end = input.find(&self.end);
            match (next_start, next_end) {
                (Some(start), Some(end)) if end < start => {
                    out.push_str(&input[..end]);
                    input.drain(..end + self.end.len());
                }
                (Some(start), _) => {
                    out.push_str(&input[..start]);
                    input.drain(..start + self.start.len());
                    self.inside_think = true;
                }
                (None, Some(end)) => {
                    out.push_str(&input[..end]);
                    input.drain(..end + self.end.len());
                }
                (None, None) => {
                    let keep =
                        partial_tag_suffix_len(&input, &[self.start.as_str(), self.end.as_str()]);
                    out.push_str(&input[..input.len() - keep]);
                    if keep > 0 {
                        self.carry = input[input.len() - keep..].to_string();
                    }
                    return out;
                }
            }
        }
    }

    fn finish(&mut self) -> String {
        if self.inside_think {
            self.inside_think = false;
            self.carry.clear();
            return String::new();
        }
        std::mem::take(&mut self.carry)
    }
}

fn partial_tag_suffix_len(text: &str, tags: &[&str]) -> usize {
    let mut best = 0;
    for tag in tags {
        for len in 1..tag.len() {
            if text.ends_with(&tag[..len]) {
                best = best.max(len);
            }
        }
    }
    best
}

/// Start the MLX inference server on the given port.
/// Returns an in-process inference server handle plus a death channel.
///
/// This is the MLX equivalent of `launch::start_llama_server` — the caller
/// gets back the same process wrapper used by the llama.cpp backend.
pub async fn start_mlx_server(
    model_dir: &std::path::Path,
    model_name: String,
    port: u16,
) -> Result<InferenceServerProcess> {
    static WARN_ONCE: Once = Once::new();
    WARN_ONCE.call_once(|| {
        eprintln!(
            "🧪 MLX support is experimental in mesh-llm. Prefer GGUF for the most mature path, and please file any issues at https://github.com/michaelneale/mesh-llm/issues."
        );
    });

    // Load model on a blocking thread (touches disk + GPU init)
    let dir = model_dir.to_path_buf();
    let model = tokio::task::spawn_blocking(move || MlxModel::load(&dir))
        .await
        .context("MLX model load panicked")??;

    tracing::info!(
        "MLX server: model loaded — {} layers, vocab={}, serving on :{port}",
        model.config.num_hidden_layers,
        model.config.vocab_size,
    );

    let context_length = model.config.max_position_embeddings as u32;

    let state = Arc::new(Mutex::new(InferState {
        model,
        model_name,
        prompt_cache: None,
    }));

    let listener = TcpListener::bind(format!("127.0.0.1:{port}"))
        .await
        .with_context(|| format!("MLX server: failed to bind port {port}"))?;

    let (death_tx, death_rx) = tokio::sync::oneshot::channel();
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        break;
                    }
                }
                accept = listener.accept() => {
                    let (stream, _addr) = match accept {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!("MLX server: accept error: {e}");
                            continue;
                        }
                    };
                    let state = state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, state).await {
                            tracing::debug!("MLX server: connection error: {e}");
                        }
                    });
                }
            }
        }
        let _ = death_tx.send(());
    });

    Ok(InferenceServerProcess {
        handle: InferenceServerHandle::in_process(shutdown_tx),
        context_length,
        death_rx,
    })
}

/// Parse a raw HTTP request from the stream and dispatch.
async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    let _ = stream.set_nodelay(true);

    let mut buf = vec![0u8; 64 * 1024];
    let mut filled = 0usize;

    // Read until we have complete headers
    loop {
        let n = stream.read(&mut buf[filled..]).await?;
        if n == 0 {
            return Ok(());
        }
        filled += n;
        if filled > 4 && buf[..filled].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if filled >= buf.len() {
            send_response(&mut stream, 413, r#"{"error":"request header too large"}"#).await?;
            return Ok(());
        }
    }

    // Find header/body split
    let header_end = buf[..filled]
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap()
        + 4;

    // Parse method, path, content-length from headers (own the strings so buf is free)
    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();

    let first_line = header_str.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let method = if parts.len() >= 2 {
        parts[0].to_string()
    } else {
        String::new()
    };
    let path = if parts.len() >= 2 {
        parts[1].to_string()
    } else {
        String::new()
    };

    let content_length: usize = header_str
        .lines()
        .find_map(|line| {
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("content-length:") {
                lower.split(':').nth(1)?.trim().parse().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);

    // Read remaining body if needed
    // Guard against oversized requests (max 16 MiB) to prevent OOM from a
    // malicious or accidental huge Content-Length.
    const MAX_BODY_SIZE: usize = 16 * 1024 * 1024;
    if content_length > MAX_BODY_SIZE {
        send_response(&mut stream, 413, r#"{"error":"request body too large"}"#).await?;
        return Ok(());
    }
    let body_so_far = filled - header_end;
    if body_so_far < content_length {
        let remaining = content_length - body_so_far;
        if filled + remaining > buf.len() {
            buf.resize(filled + remaining, 0);
        }
        let mut read = 0;
        while read < remaining {
            let n = stream
                .read(&mut buf[filled + read..filled + remaining])
                .await?;
            if n == 0 {
                break;
            }
            read += n;
        }
        filled += read;
    }

    let body = &buf[header_end..filled.min(header_end + content_length)];

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => {
            send_response(&mut stream, 200, r#"{"status":"ok"}"#).await?;
        }
        ("GET", "/v1/models") | ("GET", "/models") => {
            let state = state.lock().await;
            let resp = serde_json::json!({
                "object": "list",
                "data": [{
                    "id": &state.model_name,
                    "object": "model",
                    "owned_by": "mlx",
                }]
            });
            send_response(&mut stream, 200, &resp.to_string()).await?;
        }
        ("POST", "/v1/chat/completions") => {
            handle_chat_completions(&mut stream, body, state).await?;
        }
        ("POST", "/v1/completions") => {
            handle_completions(&mut stream, body, state).await?;
        }
        _ => {
            send_response(&mut stream, 404, r#"{"error":"not found"}"#).await?;
        }
    }

    Ok(())
}

/// Handle POST /v1/chat/completions — the main inference endpoint.
async fn handle_chat_completions(
    stream: &mut tokio::net::TcpStream,
    body: &[u8],
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    let req: serde_json::Value =
        serde_json::from_slice(body).context("invalid JSON in chat completions request")?;

    let stream_mode = req["stream"].as_bool().unwrap_or(false);
    let (generation, prompt) = {
        let state = state.lock().await;
        let generation = parse_generation_config(&req, &state.model);
        let reasoning_template = state.model.prompt_template.reasoning_template();
        let prompt_req = prepare_reasoning_request(
            &req,
            state
                .model
                .prompt_template
                .behavior()
                .prompt_template
                .as_deref(),
            state.model.reasoning_family,
            &reasoning_template,
            &generation.response_policy,
        );
        let prompt = render_chat_prompt_from_request(&state.model.prompt_template, &prompt_req)?;
        (generation, prompt)
    };
    let model_field = req["model"].as_str().unwrap_or("");

    if stream_mode {
        generate_streaming(stream, &prompt, generation, model_field, state).await
    } else {
        generate_blocking(stream, &prompt, generation, model_field, state).await
    }
}

fn render_chat_prompt_from_request(
    template: &crate::mlx::template::PromptTemplate,
    req: &serde_json::Value,
) -> Result<String> {
    template.render_request(req)
}

fn prepare_reasoning_request(
    req: &serde_json::Value,
    prompt_template: Option<&str>,
    reasoning_family: model::ReasoningFamily,
    reasoning_template: &crate::mlx::template::ReasoningTemplate,
    policy: &ResponsePolicy,
) -> serde_json::Value {
    let needs_olmo2_brevity_nudge = matches!(prompt_template, Some("olmo2"));
    if !needs_olmo2_brevity_nudge
        && (!policy.strip_reasoning_blocks
            || (reasoning_family == model::ReasoningFamily::None
                && !reasoning_template.supports_explicit_reasoning))
    {
        return req.clone();
    }

    const DIRECT_ANSWER_NUDGE: &str =
        "Respond directly with the final answer. Do not output <think> tags. Do not produce internal reasoning, hidden chain-of-thought, analysis, or preamble unless the user explicitly asks for it. Start answering immediately.";
    const OLMO2_BREVITY_NUDGE: &str =
        "Be concise. Prefer short plain paragraphs or a flat bullet list. Avoid nested outlines, repeated section labels, or repeating the same equation/premise multiple times. For follow-up turns, answer only the new request. Do not restate the previous answer, and do not dump full code/examples unless the user explicitly asks for the full content. If the user asks for a small styling or edit change, provide only the minimal change.";

    let mut patched = req.clone();
    let Some(messages) = patched
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
    else {
        return patched;
    };
    let nudge = if needs_olmo2_brevity_nudge {
        format!("{DIRECT_ANSWER_NUDGE}\n{OLMO2_BREVITY_NUDGE}")
    } else {
        DIRECT_ANSWER_NUDGE.to_string()
    };

    match messages.first_mut() {
        Some(first)
            if first
                .get("role")
                .and_then(|value| value.as_str())
                .is_some_and(|role| role == "system") =>
        {
            if let Some(content) = first.get_mut("content") {
                match content {
                    serde_json::Value::String(text) => {
                        if !text.contains(&nudge) {
                            if !text.is_empty() {
                                text.push_str("\n\n");
                            }
                            text.push_str(&nudge);
                        }
                    }
                    serde_json::Value::Array(items) => {
                        items.push(serde_json::json!({
                            "type": "text",
                            "text": nudge
                        }));
                    }
                    _ => {}
                }
            }
        }
        _ => {
            messages.insert(
                0,
                serde_json::json!({
                    "role": "system",
                    "content": nudge
                }),
            );
        }
    }

    patched
}

/// Handle POST /v1/completions — raw text completion.
/// This endpoint is not implemented; return a structured 501 so clients get a clear error
/// rather than a misleading chat-completion-shaped response.
async fn handle_completions(
    stream: &mut tokio::net::TcpStream,
    _body: &[u8],
    _state: Arc<Mutex<InferState>>,
) -> Result<()> {
    let resp = serde_json::json!({
        "error": {
            "message": "/v1/completions is not implemented by this server. Use /v1/chat/completions instead.",
            "type": "not_implemented_error",
            "param": null,
            "code": "unsupported_endpoint"
        }
    });
    send_response(stream, 501, &resp.to_string()).await
}

/// Non-streaming: run full generation, return one JSON response.
async fn generate_blocking(
    stream: &mut tokio::net::TcpStream,
    prompt: &str,
    generation: GenerationConfig,
    model_field: &str,
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    let prompt = prompt.to_string();
    let model_field = model_field.to_string();
    let outcome_result = tokio::task::spawn_blocking(move || -> Result<GenerationOutcome> {
        let mut state = state.blocking_lock();
        run_inference(&mut state, &prompt, &generation)
    })
    .await;
    let outcome = match outcome_result {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(err)) => {
            tracing::error!("MLX blocking generation failed: {err:#}");
            let payload = serde_json::json!({
                "error": {
                    "message": format!("MLX generation failed: {err}"),
                    "type": "server_error",
                    "param": null,
                    "code": "mlx_generation_failed"
                }
            });
            return send_response(stream, 500, &payload.to_string()).await;
        }
        Err(err) => {
            tracing::error!("MLX blocking generation task failed: {err:#}");
            let payload = serde_json::json!({
                "error": {
                    "message": format!("MLX generation task failed: {err}"),
                    "type": "server_error",
                    "param": null,
                    "code": "mlx_generation_task_failed"
                }
            });
            return send_response(stream, 500, &payload.to_string()).await;
        }
    };

    let resp = serde_json::json!({
        "id": format!("chatcmpl-mlx-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis()),
        "object": "chat.completion",
        "model": model_field,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": outcome.text,
            },
            "finish_reason": outcome.finish_reason,
        }],
        "usage": {
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
        }
    });
    send_response(stream, 200, &resp.to_string()).await
}

/// Streaming: send SSE chunks as tokens are generated.
async fn generate_streaming(
    stream: &mut tokio::net::TcpStream,
    prompt: &str,
    generation: GenerationConfig,
    model_field: &str,
    state: Arc<Mutex<InferState>>,
) -> Result<()> {
    // Channel for token-by-token streaming
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);

    let prompt = prompt.to_string();
    tokio::task::spawn_blocking(move || {
        let mut state = state.blocking_lock();
        let finish_reason =
            run_inference_streaming(&mut state, &prompt, &generation, &tx).unwrap_or("stop");
        let _ = tx.blocking_send(StreamEvent::Done(finish_reason));
    });

    // Send SSE headers
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(header.as_bytes()).await?;

    let id = format!(
        "chatcmpl-mlx-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    while let Some(maybe_token) = rx.recv().await {
        match maybe_token {
            StreamEvent::Text(text) => {
                let chunk = serde_json::json!({
                    "id": &id,
                    "object": "chat.completion.chunk",
                    "model": &model_field,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": text },
                        "finish_reason": null,
                    }]
                });
                let sse = format!("data: {}\n\n", chunk);
                if stream.write_all(sse.as_bytes()).await.is_err() {
                    break;
                }
            }
            StreamEvent::Done(finish_reason) => {
                // Final chunk with finish_reason
                let chunk = serde_json::json!({
                    "id": &id,
                    "object": "chat.completion.chunk",
                    "model": &model_field,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }]
                });
                let sse = format!("data: {}\n\ndata: [DONE]\n\n", chunk);
                let _ = stream.write_all(sse.as_bytes()).await;
                break;
            }
        }
    }

    let _ = stream.shutdown().await;
    Ok(())
}

/// Prefill prompt tokens. Uses chunked prefill (with eval barriers between
/// chunks) to keep the computation graph and peak memory bounded. The chunk
/// size of 2048 matches mlx-lm's default `prefill_step_size`. For prompts
/// ≤2048 tokens, there is only one chunk so no overhead.
const PREFILL_STEP_SIZE: usize = 2048;

fn prefill_logits(
    model: &MlxModel,
    prompt_tokens: &[u32],
    caches: &mut [model::KVCache],
) -> Result<Array> {
    let total = prompt_tokens.len();

    if model.tokenwise_prefill() {
        let mut logits = None;
        for &token in prompt_tokens {
            let input = Array::from_slice(&[token], &[1, 1]);
            logits = Some(model.forward(&input, caches)?);
        }
        return logits.context("tokenwise prefill received empty prompt");
    }

    if total <= PREFILL_STEP_SIZE {
        // Small prompt — single forward pass, no eval barriers
        let input = Array::from_slice(prompt_tokens, &[1, total as i32]);
        return model.forward(&input, caches);
    }

    // Large prompt — chunk to avoid huge computation graphs
    let mut pos = 0;
    while total - pos > PREFILL_STEP_SIZE {
        let chunk = &prompt_tokens[pos..pos + PREFILL_STEP_SIZE];
        let input = Array::from_slice(chunk, &[1, PREFILL_STEP_SIZE as i32]);
        model.forward(&input, caches)?;
        mlx_rs::transforms::eval(caches.iter().flat_map(|c| c.arrays()))?;
        pos += PREFILL_STEP_SIZE;
    }

    // Final chunk — get logits for the first generated token
    let last_chunk = &prompt_tokens[pos..];
    let input = Array::from_slice(last_chunk, &[1, last_chunk.len() as i32]);
    model.forward(&input, caches)
}

fn replay_last_prompt_token_logits(
    model: &MlxModel,
    prompt_tokens: &[u32],
    caches: &mut [model::KVCache],
) -> Result<Array> {
    let last = *prompt_tokens
        .last()
        .context("replay_last_prompt_token_logits called with empty prompt")?;
    let rewind_to = caches[0].offset().saturating_sub(1);
    for cache in caches.iter_mut() {
        anyhow::ensure!(
            cache.trim_to(rewind_to)?,
            "cannot rewind MLX cache to token {} during replay",
            rewind_to
        );
    }
    let input = Array::from_slice(&[last], &[1, 1]);
    model.forward(&input, caches)
}

/// Find the longest common prefix between cached tokens and new tokens.
fn common_prefix_len(cached: &[u32], new: &[u32]) -> usize {
    cached
        .iter()
        .zip(new.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

/// Set up caches for a new request, reusing the prompt cache if possible.
/// Returns (caches, tokens_to_prefill) where tokens_to_prefill is the
/// suffix of prompt_tokens that still needs to be forwarded.
fn setup_caches_with_reuse<'a>(
    state: &mut InferState,
    prompt_tokens: &'a [u32],
) -> (Vec<model::KVCache>, &'a [u32]) {
    if !state.model.prompt_cache_reuse() {
        state.prompt_cache = None;
        return (state.model.new_caches(), prompt_tokens);
    }
    if let Some(ref cached) = state.prompt_cache {
        let prefix_len = common_prefix_len(&cached.tokens, prompt_tokens);
        if prefix_len > 0 {
            let mut caches = state.prompt_cache.take().unwrap().caches;
            if caches.iter().all(|c| c.can_trim_to(prefix_len)) {
                // Reuse cached KV — trim to prefix length and return suffix
                for c in &mut caches {
                    let trimmed = c
                        .trim_to(prefix_len)
                        .expect("cache trim should succeed after can_trim_to check");
                    debug_assert!(trimmed);
                }
                tracing::info!(
                    "MLX prompt cache: reusing {prefix_len}/{} tokens ({} new)",
                    prompt_tokens.len(),
                    prompt_tokens.len() - prefix_len,
                );
                return (caches, &prompt_tokens[prefix_len..]);
            }
            tracing::info!(
                "MLX prompt cache: cannot reuse {prefix_len}/{} tokens after cache eviction; rebuilding",
                prompt_tokens.len(),
            );
        }
    }
    // No cache hit — fresh caches
    (state.model.new_caches(), prompt_tokens)
}

/// Save caches + tokens for future reuse.
fn save_prompt_cache(state: &mut InferState, tokens: Vec<u32>, caches: Vec<model::KVCache>) {
    if !state.model.prompt_cache_reuse() {
        state.prompt_cache = None;
        return;
    }
    state.prompt_cache = Some(PromptCache { tokens, caches });
}

fn is_stop_token(token: u32, generation: &GenerationConfig) -> bool {
    generation.stop_token_ids.binary_search(&token).is_ok()
}

fn should_consume_hidden_reasoning_token(
    token: u32,
    generation: &GenerationConfig,
    state: &mut HiddenReasoningState,
    visible_tokens: usize,
) -> bool {
    let Some(hidden) = generation.hidden_reasoning else {
        return false;
    };

    if state.active {
        if token == hidden.end_reasoning_token_id {
            state.active = false;
            state.skip_leading_whitespace = true;
        }
        return true;
    }

    if visible_tokens == 0 && token == hidden.start_turn_token_id {
        state.active = true;
        return true;
    }

    false
}

/// Run inference synchronously (called from blocking thread).
fn run_inference(
    state: &mut InferState,
    prompt: &str,
    generation: &GenerationConfig,
) -> Result<GenerationOutcome> {
    if state.model.cacheless_generation() {
        return run_inference_cacheless(state, prompt, generation);
    }
    let prompt_tokens = encode_prompt_tokens(&state.model, prompt)?;
    let prompt_len = prompt_tokens.len();
    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens — check that the prompt is non-empty");
    }
    tracing::debug!("MLX prompt text: {:?}", prompt);
    tracing::debug!("MLX prompt tokens: {:?}", prompt_tokens);

    let (mut caches, suffix) = setup_caches_with_reuse(state, &prompt_tokens);
    let mut sampler = Sampler::new(generation.sampling.clone());
    let mut stop_buffer = StopBuffer::new(generation.stop_sequences.clone());
    let mut response_filter = ResponseFilter::new(generation.response_policy.clone());

    if generation.max_tokens == 0 {
        if !suffix.is_empty() {
            let _ = prefill_logits(&state.model, suffix, &mut caches)?;
        }
        save_prompt_cache(state, prompt_tokens, caches);
        return Ok(GenerationOutcome {
            text: String::new(),
            prompt_tokens: prompt_len,
            completion_tokens: 0,
            finish_reason: "length",
        });
    }

    // Prefill only the new suffix
    let mut next_token = if suffix.is_empty() {
        // Entire prompt was cached — re-forward last token to get logits
        let logits = replay_last_prompt_token_logits(&state.model, &prompt_tokens, &mut caches)?;
        sampler.sample_next_token(&logits)?
    } else {
        let logits = prefill_logits(&state.model, suffix, &mut caches)?;
        let logits = if state.model.tokenwise_prefill() && state.model.can_replay_prompt_logits() {
            replay_last_prompt_token_logits(&state.model, &prompt_tokens, &mut caches)?
        } else {
            logits
        };
        sampler.sample_next_token(&logits)?
    };
    tracing::info!(
        "MLX first sampled token: id={} eos={} prompt_tokens={}",
        next_token,
        is_eos(next_token, &state.model.config),
        prompt_len
    );

    let mut decode_stream = state.model.tokenizer.decode_stream(true);
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let mut hidden_reasoning_state = HiddenReasoningState::default();
    let max_sampled_tokens = if generation.hidden_reasoning.is_some() {
        generation
            .max_tokens
            .saturating_mul(32)
            .max(generation.max_tokens)
    } else {
        generation.max_tokens
    };

    // Decode
    for step in 0..max_sampled_tokens {
        if is_eos(next_token, &state.model.config) || is_stop_token(next_token, generation) {
            finish_reason = "stop";
            break;
        }

        let pending_logits =
            if completion_tokens < generation.max_tokens && step + 1 < max_sampled_tokens {
                let input = Array::from_slice(&[next_token], &[1, 1]);
                let logits = state.model.forward(&input, &mut caches)?;
                mlx_rs::transforms::async_eval([&logits])?;
                Some(logits)
            } else {
                None
            };

        if should_consume_hidden_reasoning_token(
            next_token,
            generation,
            &mut hidden_reasoning_state,
            completion_tokens,
        ) {
            if let Some(logits) = pending_logits {
                next_token = sampler.sample_next_token(&logits)?;
            }
            continue;
        }

        let piece = decode_stream
            .step(next_token)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?
            .unwrap_or_default();
        if hidden_reasoning_state.skip_leading_whitespace && piece.trim().is_empty() {
            if let Some(logits) = pending_logits {
                next_token = sampler.sample_next_token(&logits)?;
            }
            continue;
        }
        hidden_reasoning_state.skip_leading_whitespace = false;
        completion_tokens += 1;
        let chunk = stop_buffer.push(&piece);
        text.push_str(&response_filter.push(&chunk.emit));
        if chunk.matched {
            finish_reason = "stop";
            break;
        }
        if completion_tokens >= generation.max_tokens {
            break;
        }

        if let Some(logits) = pending_logits {
            next_token = sampler.sample_next_token(&logits)?;
        }
    }

    text.push_str(&response_filter.push(&stop_buffer.finish()));
    text.push_str(&response_filter.finish());
    text = sanitize_behavior_output(&state.model, &text);

    // Save prompt cache for next request (prompt only, not generated tokens)
    save_prompt_cache(state, prompt_tokens, caches);

    Ok(GenerationOutcome {
        text,
        prompt_tokens: prompt_len,
        completion_tokens,
        finish_reason,
    })
}

/// Run inference with per-token callback for streaming.
fn run_inference_streaming(
    state: &mut InferState,
    prompt: &str,
    generation: &GenerationConfig,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) -> Result<&'static str> {
    if state.model.cacheless_generation() {
        return run_inference_streaming_cacheless(state, prompt, generation, tx);
    }
    let prompt_tokens = encode_prompt_tokens(&state.model, prompt)?;
    if prompt_tokens.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens — check that the prompt is non-empty");
    }

    let (mut caches, suffix) = setup_caches_with_reuse(state, &prompt_tokens);
    let mut sampler = Sampler::new(generation.sampling.clone());
    let mut stop_buffer = StopBuffer::new(generation.stop_sequences.clone());
    let mut response_filter = ResponseFilter::new(generation.response_policy.clone());

    if generation.max_tokens == 0 {
        if !suffix.is_empty() {
            let _ = prefill_logits(&state.model, suffix, &mut caches)?;
        }
        save_prompt_cache(state, prompt_tokens, caches);
        return Ok("length");
    }

    // Prefill only the new suffix
    let mut next_token = if suffix.is_empty() {
        let logits = replay_last_prompt_token_logits(&state.model, &prompt_tokens, &mut caches)?;
        sampler.sample_next_token(&logits)?
    } else {
        let logits = prefill_logits(&state.model, suffix, &mut caches)?;
        let logits = if state.model.tokenwise_prefill() && state.model.can_replay_prompt_logits() {
            replay_last_prompt_token_logits(&state.model, &prompt_tokens, &mut caches)?
        } else {
            logits
        };
        sampler.sample_next_token(&logits)?
    };

    let mut decode_stream = state.model.tokenizer.decode_stream(true);
    let mut finish_reason = "length";
    let mut completion_tokens = 0usize;
    let mut hidden_reasoning_state = HiddenReasoningState::default();
    let max_sampled_tokens = if generation.hidden_reasoning.is_some() {
        generation
            .max_tokens
            .saturating_mul(32)
            .max(generation.max_tokens)
    } else {
        generation.max_tokens
    };

    // Decode + stream
    for step in 0..max_sampled_tokens {
        if is_eos(next_token, &state.model.config) || is_stop_token(next_token, generation) {
            finish_reason = "stop";
            break;
        }

        let pending_logits =
            if completion_tokens < generation.max_tokens && step + 1 < max_sampled_tokens {
                let input = Array::from_slice(&[next_token], &[1, 1]);
                let logits = state.model.forward(&input, &mut caches)?;
                mlx_rs::transforms::async_eval([&logits])?;
                Some(logits)
            } else {
                None
            };

        if should_consume_hidden_reasoning_token(
            next_token,
            generation,
            &mut hidden_reasoning_state,
            completion_tokens,
        ) {
            if let Some(logits) = pending_logits {
                next_token = sampler.sample_next_token(&logits)?;
            }
            continue;
        }

        let piece = decode_stream
            .step(next_token)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?
            .unwrap_or_default();
        if hidden_reasoning_state.skip_leading_whitespace && piece.trim().is_empty() {
            if let Some(logits) = pending_logits {
                next_token = sampler.sample_next_token(&logits)?;
            }
            continue;
        }
        hidden_reasoning_state.skip_leading_whitespace = false;
        completion_tokens += 1;
        let chunk = stop_buffer.push(&piece);
        let filtered = response_filter.push(&chunk.emit);
        if !filtered.is_empty() {
            if tx.blocking_send(StreamEvent::Text(filtered)).is_err() {
                break; // client disconnected
            }
        }
        if chunk.matched {
            finish_reason = "stop";
            break;
        }
        if completion_tokens >= generation.max_tokens {
            break;
        }

        if let Some(logits) = pending_logits {
            next_token = sampler.sample_next_token(&logits)?;
        }
    }

    let mut tail = response_filter.push(&stop_buffer.finish());
    tail.push_str(&response_filter.finish());
    if !tail.is_empty() {
        let _ = tx.blocking_send(StreamEvent::Text(tail));
    }

    // Save prompt cache for next request
    save_prompt_cache(state, prompt_tokens, caches);

    Ok(finish_reason)
}

fn run_inference_cacheless(
    state: &mut InferState,
    prompt: &str,
    generation: &GenerationConfig,
) -> Result<GenerationOutcome> {
    let mut tokens = encode_prompt_tokens(&state.model, prompt)?;
    let prompt_len = tokens.len();
    let mut sampler = Sampler::new(generation.sampling.clone());
    let mut stop_buffer = StopBuffer::new(generation.stop_sequences.clone());
    let mut response_filter = ResponseFilter::new(generation.response_policy.clone());

    if generation.max_tokens == 0 {
        if !tokens.is_empty() {
            let input = Array::from_slice(&tokens, &[1, tokens.len() as i32]);
            let _ = state.model.forward_no_cache(&input)?;
        }
        return Ok(GenerationOutcome {
            text: String::new(),
            prompt_tokens: prompt_len,
            completion_tokens: 0,
            finish_reason: "length",
        });
    }

    let mut decode_stream = state.model.tokenizer.decode_stream(true);
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let mut hidden_reasoning_state = HiddenReasoningState::default();
    let max_sampled_tokens = if generation.hidden_reasoning.is_some() {
        generation
            .max_tokens
            .saturating_mul(32)
            .max(generation.max_tokens)
    } else {
        generation.max_tokens
    };

    for _ in 0..max_sampled_tokens {
        let input = Array::from_slice(&tokens, &[1, tokens.len() as i32]);
        let logits = state.model.forward_no_cache(&input)?;
        let next_token = sampler.sample_next_token(&logits)?;
        if is_eos(next_token, &state.model.config) || is_stop_token(next_token, generation) {
            finish_reason = "stop";
            break;
        }
        tokens.push(next_token);
        if should_consume_hidden_reasoning_token(
            next_token,
            generation,
            &mut hidden_reasoning_state,
            completion_tokens,
        ) {
            continue;
        }
        let piece = decode_stream
            .step(next_token)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?
            .unwrap_or_default();
        if hidden_reasoning_state.skip_leading_whitespace && piece.trim().is_empty() {
            continue;
        }
        hidden_reasoning_state.skip_leading_whitespace = false;
        completion_tokens += 1;
        let chunk = stop_buffer.push(&piece);
        text.push_str(&response_filter.push(&chunk.emit));
        if chunk.matched {
            finish_reason = "stop";
            break;
        }
        if completion_tokens >= generation.max_tokens {
            break;
        }
    }

    text.push_str(&response_filter.push(&stop_buffer.finish()));
    text.push_str(&response_filter.finish());
    text = sanitize_behavior_output(&state.model, &text);

    Ok(GenerationOutcome {
        text,
        prompt_tokens: prompt_len,
        completion_tokens,
        finish_reason,
    })
}

fn run_inference_streaming_cacheless(
    state: &mut InferState,
    prompt: &str,
    generation: &GenerationConfig,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) -> Result<&'static str> {
    let mut tokens = encode_prompt_tokens(&state.model, prompt)?;
    let mut sampler = Sampler::new(generation.sampling.clone());
    let mut stop_buffer = StopBuffer::new(generation.stop_sequences.clone());
    let mut response_filter = ResponseFilter::new(generation.response_policy.clone());

    if generation.max_tokens == 0 {
        if !tokens.is_empty() {
            let input = Array::from_slice(&tokens, &[1, tokens.len() as i32]);
            let _ = state.model.forward_no_cache(&input)?;
        }
        return Ok("length");
    }

    let mut decode_stream = state.model.tokenizer.decode_stream(true);
    let mut finish_reason = "length";
    let mut completion_tokens = 0usize;
    let mut hidden_reasoning_state = HiddenReasoningState::default();
    let max_sampled_tokens = if generation.hidden_reasoning.is_some() {
        generation
            .max_tokens
            .saturating_mul(32)
            .max(generation.max_tokens)
    } else {
        generation.max_tokens
    };

    for _ in 0..max_sampled_tokens {
        let input = Array::from_slice(&tokens, &[1, tokens.len() as i32]);
        let logits = state.model.forward_no_cache(&input)?;
        let next_token = sampler.sample_next_token(&logits)?;
        if is_eos(next_token, &state.model.config) || is_stop_token(next_token, generation) {
            finish_reason = "stop";
            break;
        }

        tokens.push(next_token);
        if should_consume_hidden_reasoning_token(
            next_token,
            generation,
            &mut hidden_reasoning_state,
            completion_tokens,
        ) {
            continue;
        }
        let piece = decode_stream
            .step(next_token)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?
            .unwrap_or_default();
        if hidden_reasoning_state.skip_leading_whitespace && piece.trim().is_empty() {
            continue;
        }
        hidden_reasoning_state.skip_leading_whitespace = false;
        completion_tokens += 1;
        let chunk = stop_buffer.push(&piece);
        let filtered = response_filter.push(&chunk.emit);
        if !filtered.is_empty() && tx.blocking_send(StreamEvent::Text(filtered)).is_err() {
            break;
        }
        if chunk.matched {
            finish_reason = "stop";
            break;
        }
        if completion_tokens >= generation.max_tokens {
            break;
        }
    }

    let mut tail = response_filter.push(&stop_buffer.finish());
    tail.push_str(&response_filter.finish());
    if !tail.is_empty() {
        let _ = tx.blocking_send(StreamEvent::Text(tail));
    }

    Ok(finish_reason)
}

fn is_eos(token: u32, config: &model::ModelConfig) -> bool {
    config.eos_token_id.contains(&token)
}

fn parse_generation_config(req: &serde_json::Value, model: &MlxModel) -> GenerationConfig {
    let mut stop_sequences = default_stop_sequences(model);
    stop_sequences.extend(parse_stop_sequences(req.get("stop")));
    stop_sequences.sort();
    stop_sequences.dedup();
    let stop_token_ids = stop_token_ids(model, &stop_sequences);
    let requested_max_tokens = req["max_tokens"].as_u64().unwrap_or(2048) as usize;
    let message_count = req["messages"]
        .as_array()
        .map(|messages| messages.len())
        .unwrap_or(0);
    let max_tokens = if model.prompt_template.behavior().prompt_template.as_deref() == Some("olmo2")
    {
        if message_count >= 4 {
            requested_max_tokens.min(48)
        } else {
            requested_max_tokens.min(64)
        }
    } else {
        requested_max_tokens
    };
    let hidden_reasoning = hidden_reasoning_tokens(model, req);
    let stop_token_ids = if let Some(hidden) = hidden_reasoning {
        stop_token_ids
            .into_iter()
            .filter(|id| *id != hidden.start_turn_token_id)
            .collect::<Vec<_>>()
    } else {
        stop_token_ids
    };
    GenerationConfig {
        max_tokens,
        sampling: SamplingParams {
            temperature: req["temperature"].as_f64().unwrap_or(0.0) as f32,
            top_p: req["top_p"].as_f64().unwrap_or(1.0) as f32,
            top_k: req["top_k"]
                .as_u64()
                .map(|value| value as usize)
                .filter(|value| *value > 0),
            seed: req["seed"].as_u64(),
            suppressed_token_ids: suppressed_reasoning_token_ids(model, req),
        },
        stop_sequences,
        stop_token_ids,
        hidden_reasoning,
        response_policy: response_policy(req, model),
    }
}

fn suppressed_reasoning_token_ids(model: &MlxModel, req: &serde_json::Value) -> Vec<u32> {
    if !reasoning_disabled(
        req,
        model.reasoning_family,
        &model.prompt_template.reasoning_template(),
    ) {
        return Vec::new();
    }

    let mut ids = Vec::new();
    for block in model.prompt_template.reasoning_template().tagged_reasoning {
        if let Some(token_id) = single_token_id_for_text(model, &block.start) {
            ids.push(token_id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn single_token_id_for_text(model: &MlxModel, text: &str) -> Option<u32> {
    let encoding = model.tokenizer.encode(text, false).ok()?;
    let ids = encoding.get_ids();
    if ids.len() == 1 {
        Some(ids[0])
    } else {
        None
    }
}

fn stop_token_ids(model: &MlxModel, stop_sequences: &[String]) -> Vec<u32> {
    let mut ids = stop_sequences
        .iter()
        .filter_map(|text| single_token_id_for_text(model, text))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn hidden_reasoning_tokens(
    model: &MlxModel,
    req: &serde_json::Value,
) -> Option<HiddenReasoningTokens> {
    if !reasoning_disabled(
        req,
        model.reasoning_family,
        &model.prompt_template.reasoning_template(),
    ) {
        return None;
    }
    if model.reasoning_family != model::ReasoningFamily::Qwen3 {
        return None;
    }
    Some(HiddenReasoningTokens {
        start_turn_token_id: single_token_id_for_text(model, "<|im_start|>")?,
        end_reasoning_token_id: single_token_id_for_text(model, "</think>")?,
    })
}

fn parse_stop_sequences(stop: Option<&serde_json::Value>) -> Vec<String> {
    match stop {
        Some(serde_json::Value::String(text)) if !text.is_empty() => vec![text.clone()],
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn default_stop_sequences(model: &MlxModel) -> Vec<String> {
    let mut stops = default_stop_sequences_for(
        model.prompt_template.behavior().prompt_template.as_deref(),
        model.reasoning_family,
    );
    stops.extend(
        model
            .prompt_template
            .reasoning_template()
            .default_stop_sequences,
    );
    stops.sort();
    stops.dedup();
    stops
}

fn default_stop_sequences_for(
    prompt_template: Option<&str>,
    reasoning_family: model::ReasoningFamily,
) -> Vec<String> {
    let mut stops = Vec::new();
    match prompt_template {
        Some("chatml") => {
            stops.push("<|im_end|>".to_string());
            stops.push("<|im_start|>".to_string());
        }
        Some("olmo2") => {
            stops.push("<|user|>".to_string());
            stops.push("<|assistant|>".to_string());
            stops.push("<|system|>".to_string());
        }
        Some("llama3") => {
            stops.push("<|eot_id|>".to_string());
        }
        Some("gemma3") => {
            stops.push("<end_of_turn>".to_string());
        }
        _ => {}
    }
    match reasoning_family {
        model::ReasoningFamily::Glm => {
            stops.push("<|user|>".to_string());
            stops.push("<|assistant|>".to_string());
            stops.push("<|system|>".to_string());
        }
        model::ReasoningFamily::Kimi => {
            stops.push("<|im_end|>".to_string());
        }
        _ => {}
    }
    stops
}

fn response_policy(req: &serde_json::Value, model: &MlxModel) -> ResponsePolicy {
    response_policy_for(
        req,
        model.reasoning_family,
        &model.prompt_template.reasoning_template(),
    )
}

fn response_policy_for(
    req: &serde_json::Value,
    reasoning_family: model::ReasoningFamily,
    reasoning_template: &crate::mlx::template::ReasoningTemplate,
) -> ResponsePolicy {
    ResponsePolicy {
        strip_reasoning_blocks: reasoning_disabled(req, reasoning_family, reasoning_template),
        tagged_reasoning: reasoning_template.tagged_reasoning.clone(),
    }
}

fn reasoning_disabled(
    req: &serde_json::Value,
    family: model::ReasoningFamily,
    reasoning_template: &crate::mlx::template::ReasoningTemplate,
) -> bool {
    match family {
        model::ReasoningFamily::Qwen3 | model::ReasoningFamily::Glm => {
            request_bool_kwarg(req, "enable_thinking").unwrap_or(false) == false
        }
        model::ReasoningFamily::Kimi => {
            if let Some(value) = request_bool_kwarg(req, "thinking") {
                !value
            } else {
                request_bool_kwarg(req, "enable_thinking").unwrap_or(false) == false
            }
        }
        model::ReasoningFamily::Lfm2 => {
            if let Some(value) = request_bool_kwarg(req, "keep_past_thinking") {
                !value
            } else {
                request_bool_kwarg(req, "enable_thinking").unwrap_or(false) == false
            }
        }
        model::ReasoningFamily::GptOss => {
            if let Some(value) = request_bool_kwarg(req, "enable_thinking") {
                !value
            } else {
                matches!(
                    request_string_kwarg(req, "reasoning_effort").as_deref(),
                    None | Some("low")
                )
            }
        }
        model::ReasoningFamily::None => {
            if !reasoning_template.supports_explicit_reasoning {
                false
            } else if let Some(value) = request_bool_kwarg(req, "thinking") {
                !value
            } else if let Some(value) = request_bool_kwarg(req, "keep_past_thinking") {
                !value
            } else if let Some(value) = request_bool_kwarg(req, "enable_thinking") {
                !value
            } else {
                matches!(
                    request_string_kwarg(req, "reasoning_effort").as_deref(),
                    Some("low")
                )
            }
        }
    }
}

fn request_bool_kwarg(req: &serde_json::Value, key: &str) -> Option<bool> {
    req.get(key).and_then(|value| value.as_bool()).or_else(|| {
        req.get("chat_template_kwargs")
            .and_then(|value| value.get(key))
            .and_then(|value| value.as_bool())
    })
}

fn request_string_kwarg(req: &serde_json::Value, key: &str) -> Option<String> {
    req.get(key)
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| {
            req.get("chat_template_kwargs")
                .and_then(|value| value.get(key))
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::unicode::NFC;
    use tokenizers::normalizers::utils::Sequence;
    use tokenizers::normalizers::Strip;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::TokenizerBuilder;

    #[test]
    fn think_block_filter_strips_tagged_reasoning_across_chunks() {
        let mut filter = ResponseFilter::new(ResponsePolicy {
            strip_reasoning_blocks: true,
            tagged_reasoning: vec![crate::mlx::template::TaggedReasoningBlock {
                start: "<think>".to_string(),
                end: "</think>".to_string(),
            }],
        });
        assert_eq!(filter.push("<thi"), "");
        assert_eq!(filter.push("nk>internal"), "");
        assert_eq!(filter.push("</think>blue"), "blue");
        assert_eq!(filter.finish(), "");
    }

    #[test]
    fn tagged_block_filter_strips_gemma4_reasoning_markers() {
        let mut filter = ResponseFilter::new(ResponsePolicy {
            strip_reasoning_blocks: true,
            tagged_reasoning: vec![crate::mlx::template::TaggedReasoningBlock {
                start: "<|channel>thought".to_string(),
                end: "<channel|>".to_string(),
            }],
        });
        assert_eq!(filter.push("<|channel>thoughthidden"), "");
        assert_eq!(filter.push("<channel|>blue"), "blue");
    }

    #[test]
    fn template_derived_stop_sequences_cover_gemma4_turn_end() {
        let mut stops =
            default_stop_sequences_for(Some("hf_template"), model::ReasoningFamily::None);
        stops.extend(vec!["<turn|>".to_string()]);
        stops.sort();
        stops.dedup();
        assert!(stops.contains(&"<turn|>".to_string()));
    }

    #[test]
    fn gemma3_default_stop_sequences_include_end_of_turn() {
        let stops = default_stop_sequences_for(Some("gemma3"), model::ReasoningFamily::None);
        assert!(stops.contains(&"<end_of_turn>".to_string()));
    }

    #[test]
    fn qwen3_generation_config_adds_chatml_stops_and_disables_thinking_by_default() {
        let stops = default_stop_sequences_for(Some("chatml"), model::ReasoningFamily::Qwen3);
        assert!(stops.contains(&"<|im_end|>".to_string()));
        assert!(stops.contains(&"<|im_start|>".to_string()));
        let policy = response_policy_for(
            &serde_json::json!({}),
            model::ReasoningFamily::Qwen3,
            &crate::mlx::template::ReasoningTemplate::default(),
        );
        assert!(policy.strip_reasoning_blocks);
    }

    #[test]
    fn olmo2_generation_config_adds_role_stops() {
        let stops = default_stop_sequences_for(Some("olmo2"), model::ReasoningFamily::None);
        assert!(stops.contains(&"<|user|>".to_string()));
        assert!(stops.contains(&"<|assistant|>".to_string()));
        assert!(stops.contains(&"<|system|>".to_string()));
    }

    #[test]
    fn qwen3_generation_config_honors_explicit_enable_thinking() {
        let policy = response_policy_for(
            &serde_json::json!({"enable_thinking": true}),
            model::ReasoningFamily::Qwen3,
            &crate::mlx::template::ReasoningTemplate::default(),
        );
        assert!(!policy.strip_reasoning_blocks);
    }

    #[test]
    fn kimi_generation_config_maps_default_to_think_filtering() {
        let stops = default_stop_sequences_for(Some("hf_template"), model::ReasoningFamily::Kimi);
        assert!(stops.contains(&"<|im_end|>".to_string()));
        let policy = response_policy_for(
            &serde_json::json!({}),
            model::ReasoningFamily::Kimi,
            &crate::mlx::template::ReasoningTemplate::default(),
        );
        assert!(policy.strip_reasoning_blocks);
    }

    #[test]
    fn glm_generation_config_defaults_to_no_thinking_and_glm_stops() {
        let stops = default_stop_sequences_for(Some("hf_template"), model::ReasoningFamily::Glm);
        assert!(stops.contains(&"<|user|>".to_string()));
        assert!(stops.contains(&"<|assistant|>".to_string()));
        let policy = response_policy_for(
            &serde_json::json!({}),
            model::ReasoningFamily::Glm,
            &crate::mlx::template::ReasoningTemplate::default(),
        );
        assert!(policy.strip_reasoning_blocks);
    }

    #[test]
    fn gpt_oss_generation_config_defaults_to_reasoning_suppression() {
        let policy = response_policy_for(
            &serde_json::json!({}),
            model::ReasoningFamily::GptOss,
            &crate::mlx::template::ReasoningTemplate::default(),
        );
        assert!(policy.strip_reasoning_blocks);
        let explicit = response_policy_for(
            &serde_json::json!({"reasoning_effort":"medium"}),
            model::ReasoningFamily::GptOss,
            &crate::mlx::template::ReasoningTemplate::default(),
        );
        assert!(!explicit.strip_reasoning_blocks);
    }

    #[test]
    fn lfm2_generation_config_defaults_to_strip_past_thinking() {
        let policy = response_policy_for(
            &serde_json::json!({}),
            model::ReasoningFamily::Lfm2,
            &crate::mlx::template::ReasoningTemplate::default(),
        );
        assert!(policy.strip_reasoning_blocks);
        let explicit = response_policy_for(
            &serde_json::json!({"keep_past_thinking":true}),
            model::ReasoningFamily::Lfm2,
            &crate::mlx::template::ReasoningTemplate::default(),
        );
        assert!(!explicit.strip_reasoning_blocks);
    }

    #[test]
    fn unknown_reasoning_family_can_still_strip_when_template_supports_reasoning_toggle() {
        let reasoning_template = crate::mlx::template::ReasoningTemplate {
            supports_explicit_reasoning: true,
            tagged_reasoning: vec![crate::mlx::template::TaggedReasoningBlock {
                start: "<think>".to_string(),
                end: "</think>".to_string(),
            }],
            default_stop_sequences: Vec::new(),
        };
        let policy = response_policy_for(
            &serde_json::json!({"enable_thinking": false}),
            model::ReasoningFamily::None,
            &reasoning_template,
        );
        assert!(policy.strip_reasoning_blocks);
        assert_eq!(policy.tagged_reasoning.len(), 1);
    }

    #[test]
    fn prepare_reasoning_request_injects_system_nudge_when_disabled() {
        let req = serde_json::json!({
            "messages": [{"role": "user", "content": "Reply with exactly: blue"}]
        });
        let patched = prepare_reasoning_request(
            &req,
            Some("chatml"),
            model::ReasoningFamily::Qwen3,
            &crate::mlx::template::ReasoningTemplate::default(),
            &ResponsePolicy {
                strip_reasoning_blocks: true,
                tagged_reasoning: Vec::new(),
            },
        );
        let messages = patched["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("Respond directly with the final answer."));
    }

    #[test]
    fn prepare_reasoning_request_leaves_request_unchanged_when_reasoning_allowed() {
        let req = serde_json::json!({
            "messages": [{"role": "user", "content": "Reply with exactly: blue"}]
        });
        let patched = prepare_reasoning_request(
            &req,
            Some("chatml"),
            model::ReasoningFamily::Qwen3,
            &crate::mlx::template::ReasoningTemplate::default(),
            &ResponsePolicy {
                strip_reasoning_blocks: false,
                tagged_reasoning: Vec::new(),
            },
        );
        assert_eq!(patched, req);
    }

    #[test]
    fn prepare_reasoning_request_injects_nudge_for_all_reasoning_families() {
        let req = serde_json::json!({
            "messages": [{"role": "user", "content": "Reply with exactly: blue"}]
        });
        for family in [
            model::ReasoningFamily::Qwen3,
            model::ReasoningFamily::Glm,
            model::ReasoningFamily::Kimi,
            model::ReasoningFamily::GptOss,
            model::ReasoningFamily::Lfm2,
        ] {
            let patched = prepare_reasoning_request(
                &req,
                Some("chatml"),
                family,
                &crate::mlx::template::ReasoningTemplate::default(),
                &ResponsePolicy {
                    strip_reasoning_blocks: true,
                    tagged_reasoning: Vec::new(),
                },
            );
            let messages = patched["messages"].as_array().unwrap();
            assert_eq!(messages[0]["role"], "system");
            assert!(messages[0]["content"]
                .as_str()
                .unwrap()
                .contains("Respond directly with the final answer."));
        }
    }

    #[test]
    fn prepare_reasoning_request_appends_nudge_to_existing_system_message() {
        let req = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Reply with exactly: blue"}
            ]
        });
        let patched = prepare_reasoning_request(
            &req,
            Some("chatml"),
            model::ReasoningFamily::Qwen3,
            &crate::mlx::template::ReasoningTemplate::default(),
            &ResponsePolicy {
                strip_reasoning_blocks: true,
                tagged_reasoning: Vec::new(),
            },
        );
        let messages = patched["messages"].as_array().unwrap();
        let system = messages[0]["content"].as_str().unwrap();
        assert!(system.contains("You are terse."));
        assert!(system.contains("Respond directly with the final answer."));
    }

    #[test]
    fn prepare_reasoning_request_injects_olmo2_brevity_nudge_without_reasoning_family() {
        let req = serde_json::json!({
            "messages": [{"role": "user", "content": "Reply with exactly: blue"}]
        });
        let patched = prepare_reasoning_request(
            &req,
            Some("olmo2"),
            model::ReasoningFamily::None,
            &crate::mlx::template::ReasoningTemplate::default(),
            &ResponsePolicy {
                strip_reasoning_blocks: false,
                tagged_reasoning: Vec::new(),
            },
        );
        let messages = patched["messages"].as_array().unwrap();
        let system = messages[0]["content"].as_str().unwrap();
        assert!(system.contains("Respond directly with the final answer."));
        assert!(system.contains("For follow-up turns, answer only the new request."));
    }

    #[test]
    fn mlx_chat_smoke_renders_llama3_prompt_from_hf_request_shape() {
        let template = crate::mlx::template::PromptTemplate::Llama3;
        let req = serde_json::json!({
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hi"}
            ]
        });

        let prompt = render_chat_prompt_from_request(&template, &req).unwrap();

        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(
            prompt.contains("<|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|>")
        );
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nSay hi<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn mlx_chat_smoke_renders_tools_prompt_from_hf_template() {
        let root =
            std::env::temp_dir().join(format!("mesh-llm-server-qwen-tools-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "{%- if tools %}{{- '<|im_start|>system\\n# Tools\\n<tools>' }}{%- for tool in tools %}{{- tool | tojson }}{%- endfor %}{{- '</tools><|im_end|>\\n' }}{%- endif %}{%- for message in messages %}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
            })
            .to_string(),
        )
        .unwrap();
        let template = crate::mlx::template::PromptTemplate::detect(
            &root,
            &serde_json::json!({"model_type":"qwen2"}),
        );
        let req = serde_json::json!({
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "Run a command"
                }
            }]
        });

        let prompt = render_chat_prompt_from_request(&template, &req).unwrap();

        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("\"name\":\"run\""));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn mlx_chat_smoke_renders_gemma3_prompt_from_hf_request_shape() {
        let template = crate::mlx::template::PromptTemplate::Gemma3;
        let req = serde_json::json!({
            "model": "mlx-community/gemma-3-4b-it-4bit",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hi"},
                {"role": "assistant", "content": "Hi."},
                {"role": "user", "content": [
                    {"type": "text", "text": "look "},
                    {"type": "image"},
                    {"type": "text", "text": "here"}
                ]}
            ]
        });

        let prompt = render_chat_prompt_from_request(&template, &req).unwrap();

        assert!(
            prompt.starts_with("<bos><start_of_turn>user\nBe concise.\n\nSay hi<end_of_turn>\n")
        );
        assert!(prompt.contains("<start_of_turn>model\nHi.<end_of_turn>\n"));
        assert!(prompt.contains("<start_of_turn>user\nlook<start_of_image>here<end_of_turn>\n"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn decode_stream_handles_split_utf8_tokens() {
        let vocab = [
            ("<0x20>".to_string(), 0),
            ("<0xC3>".to_string(), 1),
            ("<0xA9>".to_string(), 2),
            (" This".to_string(), 3),
        ];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .byte_fallback(true)
            .build()
            .unwrap();
        let tokenizer = TokenizerBuilder::new()
            .with_model(bpe)
            .with_normalizer(Some(Sequence::new(vec![
                Strip::new(true, true).into(),
                NFC.into(),
            ])))
            .with_pre_tokenizer(Some(ByteLevel::default()))
            .with_post_processor(Some(ByteLevel::default()))
            .with_decoder(Some(ByteFallback::default()))
            .build()
            .unwrap();

        let mut decode_stream = tokenizer.decode_stream(false);
        assert_eq!(decode_stream.step(0).unwrap(), Some(" ".to_string()));
        assert_eq!(decode_stream.step(1).unwrap(), None);
        assert_eq!(decode_stream.step(2).unwrap(), Some("é".to_string()));
    }

    #[test]
    #[ignore]
    fn olmo_debug_run_inference_local() {
        let dir = std::path::Path::new(
            "/Users/jdumay/.cache/mesh-llm-debug/olmo-7b-instruct-hf-same-origin/mlx/olmo-7b-instruct-hf-bf16",
        );
        assert!(
            dir.exists(),
            "missing local OLMo artifact at {}",
            dir.display()
        );

        let model = MlxModel::load(dir).expect("load local olmo mlx artifact");
        let prompt =
            "<|endoftext|><|user|>\nWhat day comes after Monday? Reply with one word.\n<|assistant|>\n";
        let generation = GenerationConfig {
            max_tokens: 4,
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                suppressed_token_ids: Vec::new(),
            },
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
            hidden_reasoning: None,
            response_policy: ResponsePolicy::default(),
        };

        let mut state = InferState {
            model,
            model_name: "olmo-debug".to_string(),
            prompt_cache: None,
        };

        let first = run_inference(&mut state, prompt, &generation).expect("first inference");
        println!("fresh text {:?}", first.text);

        let second = run_inference(&mut state, prompt, &generation).expect("second inference");
        println!("reused text {:?}", second.text);
    }

    #[test]
    #[ignore]
    fn olmo_debug_run_inference_sequence_local() {
        let dir = std::path::Path::new(
            "/Users/jdumay/.cache/mesh-llm-debug/olmo-7b-instruct-hf-same-origin/mlx/olmo-7b-instruct-hf-bf16",
        );
        assert!(
            dir.exists(),
            "missing local OLMo artifact at {}",
            dir.display()
        );

        let model = MlxModel::load(dir).expect("load local olmo mlx artifact");
        let generation = GenerationConfig {
            max_tokens: 8,
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                suppressed_token_ids: Vec::new(),
            },
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
            hidden_reasoning: None,
            response_policy: ResponsePolicy::default(),
        };
        let mut state = InferState {
            model,
            model_name: "olmo-debug".to_string(),
            prompt_cache: None,
        };

        let prompts = [
            "Reply with exactly: blue",
            "What is the capital of France? Reply with one word.",
            "List the RGB primary colors as full lowercase words only, comma-separated, with no abbreviations.",
            "Complete exactly: 2 + 2 =",
            "Name the largest planet in the Solar System. Reply with one word.",
            "What day comes after Monday? Reply with one word.",
        ];

        for prompt in prompts {
            let rendered = render_chat_prompt_from_request(
                &state.model.prompt_template,
                &serde_json::json!({
                    "messages": [{"role": "user", "content": prompt}]
                }),
            )
            .expect("render prompt");
            if prompt.contains("Monday") {
                let tokens = encode_prompt_tokens(&state.model, &rendered).expect("encode monday");
                println!("rendered monday {:?}", rendered);
                println!("rendered monday tokens {:?}", tokens);
            }
            let outcome =
                run_inference(&mut state, &rendered, &generation).expect("sequence inference");
            println!("prompt {:?} -> {:?}", prompt, outcome.text);
        }

        let fresh_model = MlxModel::load(dir).expect("reload local olmo mlx artifact");
        let mut fresh_state = InferState {
            model: fresh_model,
            model_name: "olmo-debug-fresh".to_string(),
            prompt_cache: None,
        };
        let monday = render_chat_prompt_from_request(
            &fresh_state.model.prompt_template,
            &serde_json::json!({
                "messages": [{"role": "user", "content": "What day comes after Monday? Reply with one word."}]
            }),
        )
        .expect("render monday");
        let monday_tokens =
            encode_prompt_tokens(&fresh_state.model, &monday).expect("encode monday");
        println!("fresh rendered monday {:?}", monday);
        println!("fresh rendered monday tokens {:?}", monday_tokens);
        let fresh_outcome =
            run_inference(&mut fresh_state, &monday, &generation).expect("fresh monday");
        println!("fresh monday {:?}", fresh_outcome.text);
    }
}

async fn send_response(stream: &mut tokio::net::TcpStream, status: u16, body: &str) -> Result<()> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        _ => "Unknown",
    };
    let response = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await?;
    let _ = stream.shutdown().await;
    Ok(())
}
