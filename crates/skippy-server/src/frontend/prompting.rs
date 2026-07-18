use crate::frontend::generation::ParsedChatMessage;
use crate::frontend::generation::PreparedGenerationPrompt;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::chat_message_generation_value;
use crate::frontend::generation::hook_injected_text;
use crate::frontend::generation::mid_generation_window_should_fire;
use crate::frontend::generation::parsed_chat_message_from_json;
use crate::frontend::generation::request_allowed_tool_names;
use crate::frontend::generation::tool_calls_requested;
use crate::frontend::tool_emulation;
use crate::frontend::util::openai_backend_error;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::GenerationHookSignals;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use openai_frontend::PrefillHookSignals;
use openai_frontend::apply_chat_hook_outcome;
use openai_frontend::chat_mesh_hooks_enabled;
use serde_json::Value;
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
    pub(super) fn prepare_chat_prompt(
        &self,
        request: &ChatCompletionRequest,
        options: ChatTemplateOptions,
    ) -> OpenAiResult<PreparedGenerationPrompt> {
        let marker = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
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
            return Ok(PreparedGenerationPrompt {
                text: emulated.prompt,
                media: emulated.media,
                chat_parse_metadata: Some(emulated.metadata_json),
            });
        }

        Ok(PreparedGenerationPrompt {
            text: native.prompt,
            media: native.media,
            chat_parse_metadata: Some(native.metadata_json),
        })
    }

    /// Applies the chat template for the given messages. When `messages` is
    /// `None`, the request's own messages are used. When `include_tools` is
    /// false, `tools`/`tool_choice` are omitted (used by the emulation path).
    fn render_chat_prompt(
        &self,
        request: &ChatCompletionRequest,
        options: &ChatTemplateOptions,
        marker: &str,
        messages: Option<&[openai_frontend::ChatMessage]>,
        include_tools: bool,
    ) -> OpenAiResult<RenderedChatPrompt> {
        let source_messages = messages.unwrap_or(&request.messages);
        let mut media = Vec::new();
        let template_messages = source_messages
            .iter()
            .map(|message| chat_message_generation_value(message, marker, &mut media))
            .collect::<OpenAiResult<Vec<_>>>()?;
        let messages_json = serde_json::to_string(&template_messages).map_err(|error| {
            OpenAiError::invalid_request(format!("serialize messages: {error}"))
        })?;
        let tools_json = if include_tools {
            request
                .tools
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .map_err(|error| {
                    OpenAiError::invalid_request(format!("serialize tools: {error}"))
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
                    OpenAiError::invalid_request(format!("serialize tool_choice: {error}"))
                })?
        } else {
            None
        };
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let result = runtime
            .model
            .apply_chat_template_json(
                &messages_json,
                ChatTemplateJsonOptions {
                    add_assistant: options.add_assistant,
                    enable_thinking: options.enable_thinking,
                    reasoning_format: options.reasoning_format,
                    tools_json,
                    tool_choice_json,
                    parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
                },
            )
            .map_err(openai_backend_error)?;
        Ok(RenderedChatPrompt {
            prompt: result.prompt,
            media,
            metadata_json: result.metadata_json,
        })
    }

    pub(super) fn parse_chat_output(
        &self,
        text: &str,
        request: &ChatCompletionRequest,
        metadata: Option<&str>,
        is_partial: bool,
    ) -> OpenAiResult<Option<ParsedChatMessage>> {
        let Some(metadata) = metadata else {
            return Ok(None);
        };

        // Emulation path: when tools were requested but the prompt was rendered
        // without native tool support, the model emitted TOOL_CALL text lines.
        // Parse those into OpenAI tool_calls instead of using the native parser.
        if tool_calls_requested(request)
            && !tool_emulation::template_supports_native_tool_calls(metadata)
        {
            return Ok(parse_emulated_chat_output(text, request, is_partial));
        }

        let parsed_json = {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .model
                .parse_chat_response_json(text, metadata, is_partial)
                .map_err(openai_backend_error)?
        };
        Ok(parsed_chat_message_from_json(&parsed_json, request))
    }

    pub(super) fn tokenize(&self, prompt: &str) -> OpenAiResult<Vec<i32>> {
        self.tokenize_with_options(prompt, true)
    }

    pub(super) fn tokenize_continuation(&self, text: &str) -> OpenAiResult<Vec<i32>> {
        self.tokenize_with_options(text, false)
    }

    pub(super) fn tokenize_with_options(
        &self,
        text: &str,
        add_special: bool,
    ) -> OpenAiResult<Vec<i32>> {
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        runtime
            .model
            .tokenize(text, add_special)
            .map_err(openai_backend_error)
    }

    pub(super) fn inject_hook_text_into_session(
        &self,
        session_id: &str,
        text: &str,
    ) -> OpenAiResult<Option<i32>> {
        let token_ids = self.tokenize_continuation(text)?;
        if token_ids.is_empty() {
            return Ok(None);
        }
        if token_ids.len() > 1 {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
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
        hook_request: &mut Option<ChatCompletionRequest>,
        hook_runtime: Option<&tokio::runtime::Handle>,
        decoded_tokens: usize,
        post_prefill_hook_checked: &mut bool,
        last_mid_generation_hook_at: &mut Option<usize>,
        token_signal: Option<TokenSignal>,
        signal_window: Option<GenerationSignalWindow>,
    ) -> OpenAiResult<Option<i32>> {
        let Some(hooks) = self.hook_policy.as_ref() else {
            return Ok(None);
        };
        let Some(handle) = hook_runtime else {
            return Ok(None);
        };
        let Some(request) = hook_request.as_mut() else {
            return Ok(None);
        };
        if !chat_mesh_hooks_enabled(request) {
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
        hook_request: &Option<ChatCompletionRequest>,
        hook_runtime: Option<&tokio::runtime::Handle>,
    ) -> bool {
        self.hook_policy.is_some()
            && hook_runtime.is_some()
            && hook_request.as_ref().is_some_and(chat_mesh_hooks_enabled)
    }
}

/// Parses generated text produced under tool-call emulation into a
/// [`ParsedChatMessage`]. During streaming (`is_partial`) the trailing
/// incomplete line is held back and tool calls are withheld, so a half-formed
/// `TOOL_CALL` marker is never streamed as content and calls are emitted once,
/// on finalization — matching native tool-call streaming semantics.
pub(super) fn parse_emulated_chat_output(
    text: &str,
    request: &ChatCompletionRequest,
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
        Some(Value::Array(calls))
    };
    Some(ParsedChatMessage {
        content: parse.content,
        reasoning_content: None,
        tool_calls,
    })
}
