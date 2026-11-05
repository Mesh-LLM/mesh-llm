use crate::frontend::generation::ParsedChatMessage;
use crate::frontend::generation::PreparedGenerationPrompt;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::chat_message_generation_value;
use crate::frontend::generation::effective_tools_for_choice;
use crate::frontend::generation::hook_injected_text;
use crate::frontend::generation::mid_generation_window_should_fire;
use crate::frontend::generation::parsed_chat_message_from_json;
use crate::frontend::generation::request_allowed_tool_names;
use crate::frontend::generation::run_blocking_generation_worker;
use crate::frontend::generation::tool_calls_requested;
use crate::frontend::tool_emulation;
use crate::frontend::util::openai_backend_error;
use crate::runtime_state::panic_recovery::lock_runtime;
use crate::telemetry::active_span::{ActiveSpan, SpanLevel};
use serde_json::Value;
use skippy_inference_api::GenerationHookSignals;
use skippy_inference_api::InferenceError;
use skippy_inference_api::InferenceRequestContext;
use skippy_inference_api::InferenceResult;
use skippy_inference_api::MessageContent;
use skippy_inference_api::PrefillHookSignals;
use skippy_inference_api::apply_chat_hook_outcome;
use skippy_inference_api::chat_skippy_hooks_enabled;
use skippy_inference_api::generation::GenerationRequest;
use skippy_runtime::ChatTemplateJsonOptions;
use skippy_runtime::ChatTemplateOptions;
use skippy_runtime::GenerationSignalWindow;
use skippy_runtime::MediaInput;
use skippy_runtime::TokenSignal;

/// Output of a single chat-template application, before it is folded into a
/// [`PreparedGenerationPrompt`].
struct RenderedChatPrompt {
    prompt: String,
    media: Vec<MediaInput>,
    metadata_json: String,
}

impl StageOpenAiBackend {
    /// Offload [`Self::prepare_chat_prompt`] to a blocking thread.
    ///
    /// Template rendering takes the runtime mutex and runs the template
    /// engine synchronously (FFI). Under a connect burst (e.g. a c256
    /// benchmark wave) running that on tokio workers starves the accept
    /// task: every worker blocks on the mutex or renders while SYNs pile
    /// up behind the listen backlog. Moving it to the blocking pool keeps
    /// the async workers (and therefore the accept loop) responsive. This
    /// mirrors the `spawn_blocking` already used for prompt tokenization.
    pub(super) async fn prepare_chat_prompt_offloaded(
        &self,
        request: &GenerationRequest,
        options: ChatTemplateOptions,
        context: InferenceRequestContext,
    ) -> InferenceResult<PreparedGenerationPrompt> {
        let backend = self.clone();
        let request = request.clone();
        run_blocking_generation_worker((), context, move |_| {
            backend.prepare_chat_prompt(&request, options)
        })
        .await
        .map_err(|error| {
            InferenceError::backend(format!("chat prompt preparation task failed: {error}"))
        })?
    }

    pub(super) fn prepare_chat_prompt(
        &self,
        request: &GenerationRequest,
        options: ChatTemplateOptions,
    ) -> InferenceResult<PreparedGenerationPrompt> {
        let marker = {
            let runtime = lock_runtime(&self.runtime);
            runtime.media_marker()
        };

        // Native path: render with tools and use the template's own metadata.
        let native = self.render_chat_prompt(request, &options, &marker, None, true)?;

        // If the request carries tools but the template does not support native
        // tool calling, re-render with server-side tool-call emulation: strip
        // tools, inject a text-convention instruction, and rewrite history so
        // the template never sees tool roles. Tool-capable templates keep the
        // native prompt unchanged.
        if tool_calls_requested(request)
            && tool_emulation::should_emulate_tool_calls(&native.metadata_json)
            && let Some(tools) = request.tools.as_ref()
            && let Some(instruction) = tool_emulation::build_emulation_instruction(tools)
        {
            let rewritten =
                tool_emulation::rewrite_history_for_emulation(&request.messages, &instruction);
            let emulated =
                self.render_chat_prompt(request, &options, &marker, Some(&rewritten), false)?;
            let (recurrent_cache_prefix_text, recurrent_system_prefix_text) = if self
                .renders_recurrent_checkpoints(&options, &emulated)
            {
                (
                    self.render_chat_prefix(request, &options, &marker, Some(&rewritten), false),
                    self.render_system_prefix(request, &options, &marker, &rewritten, false),
                )
            } else {
                (None, None)
            };
            return Ok(PreparedGenerationPrompt {
                text: emulated.prompt,
                media: emulated.media,
                chat_parse_metadata: Some(emulated.metadata_json),
                recurrent_cache_prefix_text,
                recurrent_system_prefix_text,
            });
        }

        let (recurrent_cache_prefix_text, recurrent_system_prefix_text) =
            if self.renders_recurrent_checkpoints(&options, &native) {
                (
                    self.render_chat_prefix(request, &options, &marker, None, true),
                    self.render_system_prefix(request, &options, &marker, &request.messages, true),
                )
            } else {
                (None, None)
            };
        Ok(PreparedGenerationPrompt {
            text: native.prompt,
            media: native.media,
            chat_parse_metadata: Some(native.metadata_json),
            recurrent_cache_prefix_text,
            recurrent_system_prefix_text,
        })
    }

    /// Recurrent checkpoints resume at rendered message boundaries, so a
    /// text-only prompt that asks for an assistant turn renders its candidates.
    fn renders_recurrent_checkpoints(
        &self,
        options: &ChatTemplateOptions,
        rendered: &RenderedChatPrompt,
    ) -> bool {
        rendered.media.is_empty()
            && options.add_assistant
            && self
                .kv
                .as_ref()
                .is_some_and(|kv| kv.retains_recurrent_checkpoints())
    }

    /// The messages without the assistant-generation suffix.
    fn render_chat_prefix(
        &self,
        request: &ChatCompletionRequest,
        options: &ChatTemplateOptions,
        marker: &str,
        messages: Option<&[ChatMessage]>,
        include_tools: bool,
    ) -> Option<String> {
        let mut prefix_options = options.clone();
        prefix_options.add_assistant = false;
        self.render_chat_prompt(request, &prefix_options, marker, messages, include_tools)
            .ok()
            .map(|rendered| rendered.prompt)
    }

    /// The leading system messages and an empty user turn, without the
    /// assistant-generation suffix.
    fn render_system_prefix(
        &self,
        request: &ChatCompletionRequest,
        options: &ChatTemplateOptions,
        marker: &str,
        messages: &[ChatMessage],
        include_tools: bool,
    ) -> Option<String> {
        let prefix = system_prefix_messages(messages)?;
        self.render_chat_prefix(request, options, marker, Some(&prefix), include_tools)
    }

    /// Applies the chat template for the given messages. When `messages` is
    /// `None`, the request's own messages are used. When `include_tools` is
    /// false, `tools`/`tool_choice` are omitted (used by the emulation path).
    fn render_chat_prompt(
        &self,
        request: &GenerationRequest,
        options: &ChatTemplateOptions,
        marker: &str,
        messages: Option<&[skippy_inference_api::generation::GenerationMessage]>,
        include_tools: bool,
    ) -> InferenceResult<RenderedChatPrompt> {
        let source_messages = messages.unwrap_or(&request.messages);
        let mut media = Vec::new();
        let template_messages = source_messages
            .iter()
            .map(|message| chat_message_generation_value(message, marker, &mut media))
            .collect::<InferenceResult<Vec<_>>>()?;
        let messages_json = serde_json::to_string(&template_messages).map_err(|error| {
            InferenceError::invalid_request(format!("serialize messages: {error}"))
        })?;
        let tools_json = if include_tools {
            effective_tools_for_choice(request.tools.as_ref(), request.tool_choice.as_ref())
                .map(|value| serde_json::to_string(&value))
                .transpose()
                .map_err(|error| {
                    InferenceError::invalid_request(format!("serialize tools: {error}"))
                })?
        } else {
            None
        };
        let tool_choice_json = if include_tools {
            request
                .tool_choice
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .map_err(|error| {
                    InferenceError::invalid_request(format!("serialize tool_choice: {error}"))
                })?
        } else {
            None
        };
        let reader = self.model_reader();
        let result = reader
            .apply_chat_template_json(
                &messages_json,
                ChatTemplateJsonOptions {
                    add_assistant: options.add_assistant,
                    enable_thinking: options.enable_thinking,
                    reasoning_format: options.reasoning_format,
                    chat_template_kwargs: options.chat_template_kwargs.clone(),
                    tools_json,
                    tool_choice_json,
                    parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
                    chat_template: options.chat_template.clone(),
                    use_jinja: options.use_jinja,
                    grammar: options.grammar.clone(),
                    json_schema: options.json_schema.clone(),
                    skip_chat_parsing: options.skip_chat_parsing,
                },
            )
            .map_err(openai_backend_error)?;
        crate::grammar_bounds::check_chat_metadata_grammar(&result.metadata_json)
            .map_err(InferenceError::invalid_request)?;
        Ok(RenderedChatPrompt {
            prompt: result.prompt,
            media,
            metadata_json: result.metadata_json,
        })
    }

    pub(super) fn parse_chat_output(
        &self,
        text: &str,
        request: &GenerationRequest,
        metadata: Option<&str>,
        is_partial: bool,
    ) -> InferenceResult<Option<ParsedChatMessage>> {
        let Some(metadata) = metadata else {
            return Ok(None);
        };
        let model = lock_runtime(&self.runtime).model.reader();
        self.parse_chat_output_with_reader(&model, text, request, metadata, is_partial)
    }

    pub(super) fn parse_chat_output_with_reader(
        &self,
        model: &skippy_runtime::StageModelReader,
        text: &str,
        request: &GenerationRequest,
        metadata: &str,
        is_partial: bool,
    ) -> InferenceResult<Option<ParsedChatMessage>> {
        if serde_json::from_str::<Value>(metadata)
            .ok()
            .and_then(|value| value.get("skip_chat_parsing").and_then(Value::as_bool))
            .unwrap_or(false)
        {
            return Ok(Some(ParsedChatMessage {
                content: Some(text.to_string()),
                reasoning_content: None,
                tool_calls: None,
            }));
        }

        // Emulation path: when tools were requested but the prompt was rendered
        // without native tool support, the model emitted TOOL_CALL text lines.
        // Parse those into OpenAI tool_calls instead of using the native parser.
        if tool_calls_requested(request)
            && !tool_emulation::template_supports_native_tool_calls(metadata)
        {
            return Ok(parse_emulated_chat_output(text, request, is_partial));
        }

        let parsed_json = model
            .parse_chat_response_json(text, metadata, is_partial)
            .map_err(openai_backend_error)?;
        Ok(parsed_chat_message_from_json(&parsed_json, request))
    }

    pub(super) fn tokenize(&self, prompt: &str) -> InferenceResult<Vec<i32>> {
        self.tokenize_with_options(prompt, true)
    }

    pub(super) fn tokenize_continuation(&self, text: &str) -> InferenceResult<Vec<i32>> {
        self.tokenize_with_options(text, false)
    }

    pub(super) fn tokenize_with_options(
        &self,
        text: &str,
        add_special: bool,
    ) -> InferenceResult<Vec<i32>> {
        self.model_reader()
            .tokenize(text, add_special)
            .map_err(openai_backend_error)
    }

    /// Clone the immutable model reader, so tokenizer and template FFI run
    /// without the runtime lock. Taking the lock to clone it still waits for
    /// whatever holds the runtime, such as a whole prefill iteration, so that
    /// wait is a span.
    pub(in crate::frontend) fn model_reader(&self) -> skippy_runtime::StageModelReader {
        let wait = ActiveSpan::start(
            Some(&self.telemetry),
            "stage.runtime_lock_wait",
            SpanLevel::Debug,
        );
        let reader = lock_runtime(&self.runtime).model.reader();
        wait.end();
        reader
    }

    pub(super) fn inject_hook_text_into_session(
        &self,
        session_id: &str,
        text: &str,
    ) -> InferenceResult<Option<i32>> {
        let token_ids = self.tokenize_continuation(text)?;
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() > 1 {
            let mut runtime = lock_runtime(&self.runtime);
            runtime
                .prefill(session_id, &token_ids[..token_ids.len() - 1])
                .map_err(openai_backend_error)?;
        }
        Ok(token_ids.last().copied())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn maybe_run_generation_hooks(
        &self,
        session_id: &str,
        hook_request: &mut Option<GenerationRequest>,
        hook_runtime: Option<&tokio::runtime::Handle>,
        decoded_tokens: usize,
        post_prefill_hook_checked: &mut bool,
        last_mid_generation_hook_at: &mut Option<usize>,
        token_signal: Option<TokenSignal>,
        signal_window: Option<GenerationSignalWindow>,
    ) -> InferenceResult<Option<i32>> {
        let Some(hooks) = self.hook_policy.as_ref() else {
            return Ok(None);
        };
        let Some(handle) = hook_runtime else {
            return Ok(None);
        };
        let Some(request) = hook_request.as_mut() else {
            return Ok(None);
        };
        if !chat_skippy_hooks_enabled(request) {
            return Ok(None);
        }

        if !*post_prefill_hook_checked {
            *post_prefill_hook_checked = true;
            if let Some(signal) = token_signal {
                let signals = PrefillHookSignals {
                    first_token_entropy: f64::from(signal.entropy),
                    first_token_margin: f64::from(signal.margin),
                };
                let outcome = handle.block_on(hooks.after_prefill(request, signals))?;
                apply_chat_hook_outcome(request, &outcome);
                if let Some(text) = hook_injected_text(&outcome) {
                    return self.inject_hook_text_into_session(session_id, &text);
                }
            }
        }

        let Some(window) = signal_window else {
            return Ok(None);
        };
        if !mid_generation_window_should_fire(decoded_tokens, last_mid_generation_hook_at, &window)
        {
            return Ok(None);
        }

        let signals = GenerationHookSignals {
            n_decoded: i64::try_from(decoded_tokens).unwrap_or(i64::MAX),
            window_tokens: window.token_count,
            mean_entropy: f64::from(window.mean_entropy),
            max_entropy: f64::from(window.max_entropy),
            mean_margin: f64::from(window.mean_margin),
            min_margin: f64::from(window.min_margin),
            high_entropy_count: window.high_entropy_count,
            repetition_count: window.repetition_count,
        };
        let outcome = handle.block_on(hooks.mid_generation(request, signals))?;
        *last_mid_generation_hook_at = Some(decoded_tokens);
        apply_chat_hook_outcome(request, &outcome);
        if let Some(text) = hook_injected_text(&outcome) {
            return self.inject_hook_text_into_session(session_id, &text);
        }
        Ok(None)
    }

    pub(super) fn generation_hooks_active(
        &self,
        hook_request: &Option<GenerationRequest>,
        hook_runtime: Option<&tokio::runtime::Handle>,
    ) -> bool {
        self.hook_policy.is_some()
            && hook_runtime.is_some()
            && hook_request.as_ref().is_some_and(chat_skippy_hooks_enabled)
    }
}

/// Parses generated text produced under tool-call emulation into a
/// [`ParsedChatMessage`]. During streaming (`is_partial`) the trailing
/// incomplete line is held back and tool calls are withheld, so a half-formed
/// `TOOL_CALL` marker is never streamed as content and calls are emitted once,
/// on finalization.
///
/// Unlike the native path, emulated tool calls are therefore *not* streamed
/// incrementally. The emulated parser only recognises a call once its JSON
/// object closes (`tool_emulation::parse_emulated_tool_calls`), so it has no
/// representation of a partially generated argument object to stream from;
/// streaming it needs a partial-JSON scanner, not a change to this gate.
pub(super) fn parse_emulated_chat_output(
    text: &str,
    request: &GenerationRequest,
    is_partial: bool,
) -> Option<ParsedChatMessage> {
    let allowed = request_allowed_tool_names(request);
    let partial_scan = is_partial.then(|| tool_emulation::partial_emulation_text(text));
    let scan_text = partial_scan.as_deref().unwrap_or(text);
    let parse = tool_emulation::parse_emulated_tool_calls(scan_text, &allowed);
    let tool_calls = if is_partial || parse.tool_calls.is_empty() {
        None
    } else {
        let mut calls = parse.tool_calls;
        if request.parallel_tool_calls == Some(false) {
            calls.truncate(1);
        }
        skippy_inference_api::ensure_tool_call_ids(&mut calls);
        Some(Value::Array(calls))
    };
    Some(ParsedChatMessage {
        content: parse.content,
        reasoning_content: None,
        tool_calls,
    })
}

/// The leading system and developer messages and an empty user turn, when the
/// conversation has both. Every request with those system messages renders the
/// same tokens up to where the user's text begins.
fn system_prefix_messages(messages: &[ChatMessage]) -> Option<Vec<ChatMessage>> {
    let system = messages
        .iter()
        .take_while(|message| matches!(message.role.as_str(), "system" | "developer"))
        .count();
    (system > 0 && system < messages.len()).then(|| {
        let user = ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(String::new())),
            extra: BTreeMap::new(),
        };
        messages[..system]
            .iter()
            .cloned()
            .chain(std::iter::once(user))
            .collect()
    })
}

#[cfg(test)]
mod system_prefix_tests {
    use super::*;

    fn message(role: &str, text: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(MessageContent::Text(text.to_string())),
            extra: BTreeMap::new(),
        }
    }

    #[test]
    fn system_messages_are_followed_by_an_empty_user_turn() {
        let conversation = [
            message("system", "rules"),
            message("developer", "tools"),
            message("user", "first question"),
            message("assistant", "answer"),
            message("user", "second question"),
        ];
        assert_eq!(
            system_prefix_messages(&conversation),
            Some(vec![
                message("system", "rules"),
                message("developer", "tools"),
                message("user", ""),
            ])
        );
    }

    #[test]
    fn conversations_without_both_parts_have_no_system_prefix() {
        assert_eq!(system_prefix_messages(&[message("user", "question")]), None);
        assert_eq!(system_prefix_messages(&[message("system", "rules")]), None);
        assert_eq!(system_prefix_messages(&[]), None);
    }
}
