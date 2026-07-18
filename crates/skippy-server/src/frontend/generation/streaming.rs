use crate::frontend::generation::GeneratedText;
use crate::frontend::generation::GenerationCacheStats;
use crate::frontend::generation::GenerationMetrics;
use crate::frontend::generation::PreparedGenerationPrompt;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::tool_calls_requested;
use crate::frontend::generation::tool_calls_stream_delta;
use crate::frontend::tool_emulation;
use crate::frontend::util::detokenize_bytes_with_runtime;
use crate::frontend::util::finish_reason_for_generation;
use crate::frontend::util::saturating_u32;
use crate::frontend::util::token_is_eog_with_runtime;
use crate::frontend::util::trim_at_stop;
use crate::frontend::util::valid_utf8_prefix_len;
use crate::runtime_state::RuntimeState;
use openai_frontend::ChatCompletionChunk;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::CompletionChunk;
use openai_frontend::FinishReason;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use openai_frontend::Usage;
use serde_json::Value;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

pub(in crate::frontend) type GenerationStream =
    std::pin::Pin<Box<dyn futures_util::Stream<Item = OpenAiResult<GenerationStreamEvent>> + Send>>;

pub(in crate::frontend) enum GenerationStreamEvent {
    Delta(String),
    ReasoningDelta(String),
    ToolCalls(Value),
    Usage(Usage),
    Done(FinishReason),
}

pub(in crate::frontend) struct ChatOutputStreamParser {
    pub(in crate::frontend) backend: StageOpenAiBackend,
    pub(in crate::frontend) request: ChatCompletionRequest,
    pub(in crate::frontend) metadata: String,
    pub(in crate::frontend) emit_reasoning: bool,
    pub(in crate::frontend) text: String,
    pub(in crate::frontend) emitted_content: String,
    pub(in crate::frontend) emitted_reasoning_content: String,
    pub(in crate::frontend) emitted_tool_calls: bool,
}

impl ChatOutputStreamParser {
    pub(in crate::frontend) fn new(
        backend: StageOpenAiBackend,
        request: ChatCompletionRequest,
        metadata: String,
        emit_reasoning: bool,
    ) -> Self {
        Self {
            backend,
            request,
            metadata,
            emit_reasoning,
            text: String::new(),
            emitted_content: String::new(),
            emitted_reasoning_content: String::new(),
            emitted_tool_calls: false,
        }
    }

    pub(in crate::frontend) fn push_delta(
        &mut self,
        delta: &str,
    ) -> OpenAiResult<Vec<GenerationStreamEvent>> {
        self.text.push_str(delta);
        self.events_for_text(true)
    }

    pub(in crate::frontend) fn finish(
        &mut self,
        text: &str,
    ) -> OpenAiResult<Vec<GenerationStreamEvent>> {
        if self.text != text {
            self.text = text.to_string();
        }
        self.events_for_text(false)
    }

    pub(in crate::frontend) fn events_for_text(
        &mut self,
        is_partial: bool,
    ) -> OpenAiResult<Vec<GenerationStreamEvent>> {
        let Some(parsed) = self.backend.parse_chat_output(
            &self.text,
            &self.request,
            Some(&self.metadata),
            is_partial,
        )?
        else {
            return Ok(Vec::new());
        };
        let mut events = Vec::new();
        if self.emit_reasoning
            && let Some(delta) = suffix_delta(
                parsed.reasoning_content.as_deref(),
                &mut self.emitted_reasoning_content,
            )
        {
            events.push(GenerationStreamEvent::ReasoningDelta(delta));
        }
        if let Some(delta) = suffix_delta(parsed.content.as_deref(), &mut self.emitted_content) {
            events.push(GenerationStreamEvent::Delta(delta));
        }
        if let (true, Some(tool_calls)) =
            (!is_partial && !self.emitted_tool_calls, parsed.tool_calls)
        {
            self.emitted_tool_calls = true;
            events.push(GenerationStreamEvent::ToolCalls(tool_calls));
        }
        Ok(events)
    }

    pub(in crate::frontend) fn finish_reason(&self, fallback: FinishReason) -> FinishReason {
        if self.emitted_tool_calls {
            FinishReason::ToolCalls
        } else {
            fallback
        }
    }
}

pub(in crate::frontend) fn suffix_delta(
    current: Option<&str>,
    emitted: &mut String,
) -> Option<String> {
    let current = current?;
    let delta = current.strip_prefix(emitted.as_str())?;
    if delta.is_empty() {
        return None;
    }
    emitted.push_str(delta);
    Some(delta.to_string())
}

pub(in crate::frontend) fn generation_event_to_chat_chunk(
    event: OpenAiResult<GenerationStreamEvent>,
    model: &str,
) -> OpenAiResult<ChatCompletionChunk> {
    match event? {
        GenerationStreamEvent::Delta(delta) => {
            Ok(ChatCompletionChunk::delta(model.to_string(), delta))
        }
        GenerationStreamEvent::ReasoningDelta(delta) => Ok(ChatCompletionChunk {
            id: openai_frontend::completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: openai_frontend::now_unix_secs(),
            model: model.to_string(),
            choices: vec![openai_frontend::ChatCompletionChunkChoice {
                index: 0,
                delta: openai_frontend::ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: Some(delta),
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
        }),
        GenerationStreamEvent::ToolCalls(tool_calls) => Ok(ChatCompletionChunk {
            id: openai_frontend::completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: openai_frontend::now_unix_secs(),
            model: model.to_string(),
            choices: vec![openai_frontend::ChatCompletionChunkChoice {
                index: 0,
                delta: openai_frontend::ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(tool_calls_stream_delta(tool_calls)),
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
        }),
        GenerationStreamEvent::Usage(usage) => {
            Ok(ChatCompletionChunk::usage(model.to_string(), usage))
        }
        GenerationStreamEvent::Done(reason) => Ok(ChatCompletionChunk::done_with_reason(
            model.to_string(),
            reason,
        )),
    }
}

pub(in crate::frontend) fn generation_event_to_completion_chunk(
    event: OpenAiResult<GenerationStreamEvent>,
    model: &str,
) -> OpenAiResult<CompletionChunk> {
    match event? {
        GenerationStreamEvent::Delta(delta) => Ok(CompletionChunk::delta(model.to_string(), delta)),
        GenerationStreamEvent::ReasoningDelta(_) => {
            Ok(CompletionChunk::delta(model.to_string(), ""))
        }
        GenerationStreamEvent::ToolCalls(_) => Ok(CompletionChunk::delta(model.to_string(), "")),
        GenerationStreamEvent::Usage(usage) => Ok(CompletionChunk::usage(model.to_string(), usage)),
        GenerationStreamEvent::Done(reason) => {
            Ok(CompletionChunk::done_with_reason(model.to_string(), reason))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::frontend) enum TokenControl {
    Continue,
    Stop,
}

/// Whether generation for this request is running under tool-call emulation:
/// the request carries tools and the rendered prompt metadata does not report
/// native tool support (empty grammar triggers). Used to enable early
/// generation stop once a complete emulated tool call has been produced.
pub(in crate::frontend) fn emulation_generation_active(
    hook_request: Option<&ChatCompletionRequest>,
    prompt: &PreparedGenerationPrompt,
) -> bool {
    let Some(request) = hook_request else {
        return false;
    };
    if !tool_calls_requested(request) {
        return false;
    }
    prompt
        .chat_parse_metadata
        .as_deref()
        .map(|metadata| !tool_emulation::template_supports_native_tool_calls(metadata))
        .unwrap_or(false)
}

pub(in crate::frontend) struct TextGenerationCollector<'a, F>
where
    F: FnMut(&str) -> OpenAiResult<()>,
{
    runtime: Arc<Mutex<RuntimeState>>,
    stop_values: Vec<&'a str>,
    on_text_chunk: F,
    text: String,
    streamed_text_len: usize,
    max_stop_bytes: usize,
    generated_text_tokens: Vec<i32>,
    completion_tokens: usize,
    finish_reason: FinishReason,
    metrics: GenerationMetrics,
    emulation_active: bool,
}

impl<'a, F> TextGenerationCollector<'a, F>
where
    F: FnMut(&str) -> OpenAiResult<()>,
{
    pub(in crate::frontend) fn new(
        runtime: Arc<Mutex<RuntimeState>>,
        stop_values: Vec<&'a str>,
        on_text_chunk: F,
    ) -> Self {
        let max_stop_bytes = stop_values
            .iter()
            .map(|value| value.len())
            .max()
            .unwrap_or(0);
        Self {
            runtime,
            stop_values,
            on_text_chunk,
            text: String::new(),
            streamed_text_len: 0,
            max_stop_bytes,
            generated_text_tokens: Vec::new(),
            completion_tokens: 0,
            finish_reason: finish_reason_for_generation(true),
            metrics: GenerationMetrics::default(),
            emulation_active: false,
        }
    }

    /// Enables early generation stop once a complete emulated tool call is
    /// generated. Mirrors goose's `tool_call_emitted -> Stop` so the model does
    /// not ramble after emitting a call. Only active for emulated tools
    /// requests; native requests are unaffected.
    pub(in crate::frontend) fn with_emulation_stop(mut self, emulation_active: bool) -> Self {
        self.emulation_active = emulation_active;
        self
    }

    pub(in crate::frontend) fn push_token(&mut self, token: i32) -> OpenAiResult<TokenControl> {
        let eog_timer = Instant::now();
        if token_is_eog_with_runtime(&self.runtime, token)? {
            self.metrics.eog_check_ms += eog_timer.elapsed().as_secs_f64() * 1000.0;
            self.finish_reason = finish_reason_for_generation(false);
            return Ok(TokenControl::Stop);
        }
        self.metrics.eog_check_ms += eog_timer.elapsed().as_secs_f64() * 1000.0;
        self.completion_tokens += 1;
        self.generated_text_tokens.push(token);
        let detokenize_timer = Instant::now();
        let candidate_bytes =
            detokenize_bytes_with_runtime(&self.runtime, &self.generated_text_tokens)?;
        self.metrics.detokenize_ms += detokenize_timer.elapsed().as_secs_f64() * 1000.0;
        let valid_len = valid_utf8_prefix_len(&candidate_bytes);
        if valid_len > 0 {
            let candidate = std::str::from_utf8(&candidate_bytes[..valid_len])
                .map_err(|error| OpenAiError::backend(error.to_string()))?;
            if let Some(delta) = candidate.strip_prefix(&self.text) {
                if !delta.is_empty() {
                    self.text = candidate.to_string();
                }
            } else if candidate != self.text {
                self.text = candidate.to_string();
            }
        }
        if self
            .stop_values
            .iter()
            .any(|stop| !stop.is_empty() && self.text.contains(stop))
        {
            self.text = trim_at_stop(&self.text, &self.stop_values).to_string();
            self.emit_safe_delta(true)?;
            self.finish_reason = finish_reason_for_generation(false);
            return Ok(TokenControl::Stop);
        }
        if self.emulation_active && tool_emulation::emulated_tool_call_complete(&self.text) {
            self.emit_safe_delta(true)?;
            self.finish_reason = finish_reason_for_generation(false);
            return Ok(TokenControl::Stop);
        }
        self.emit_safe_delta(false)?;
        Ok(TokenControl::Continue)
    }

    pub(in crate::frontend) fn emit_safe_delta(&mut self, flush_all: bool) -> OpenAiResult<()> {
        let mut target_len = if flush_all || self.max_stop_bytes == 0 {
            self.text.len()
        } else {
            self.text
                .len()
                .saturating_sub(self.max_stop_bytes.saturating_sub(1))
        };
        while target_len > self.streamed_text_len && !self.text.is_char_boundary(target_len) {
            target_len -= 1;
        }
        if target_len < self.streamed_text_len {
            self.streamed_text_len = target_len;
            return Ok(());
        }
        if target_len > self.streamed_text_len {
            let delta = &self.text[self.streamed_text_len..target_len];
            let emit_timer = Instant::now();
            (self.on_text_chunk)(delta)?;
            self.metrics.text_emit_ms += emit_timer.elapsed().as_secs_f64() * 1000.0;
            self.streamed_text_len = target_len;
        }
        Ok(())
    }

    pub(in crate::frontend) fn finish(
        mut self,
        prompt_token_count: usize,
        cache_stats: GenerationCacheStats,
    ) -> OpenAiResult<GeneratedText> {
        self.emit_safe_delta(true)?;
        Ok(GeneratedText {
            prompt_tokens: saturating_u32(prompt_token_count),
            completion_tokens: saturating_u32(self.completion_tokens),
            cache_status: cache_stats.status,
            cached_prompt_tokens: cache_stats.cached_prompt_tokens,
            matched_prefix_tokens: cache_stats.matched_prefix_tokens,
            suffix_prefill_tokens: cache_stats.suffix_prefill_tokens,
            cache_hit_kind: cache_stats.hit_kind,
            native_mtp_stats: cache_stats.native_mtp_stats,
            native_mtp_decode_telemetry: cache_stats.native_mtp_decode_telemetry,
            verify_window_pipeline_stats: cache_stats.verify_window_pipeline_stats,
            speculative_stats: cache_stats.speculative_stats,
            prompt_ms: cache_stats.prompt_ms,
            predicted_ms: cache_stats.predicted_ms,
            text: self.text,
            finish_reason: self.finish_reason,
            detokenize_ms: self.metrics.detokenize_ms,
            text_emit_ms: self.metrics.text_emit_ms,
            eog_check_ms: self.metrics.eog_check_ms,
        })
    }
}
