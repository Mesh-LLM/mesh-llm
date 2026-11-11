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
use crate::kv_integration::StagePrefixCachePayload;
use crate::runtime_state::panic_recovery::lock_runtime;
use serde_json::Value;
use skippy_inference_api::GenerationHookSignals;
use skippy_inference_api::InferenceError;
use skippy_inference_api::InferenceRequestContext;
use skippy_inference_api::InferenceResult;
use skippy_inference_api::PrefillHookSignals;
use skippy_inference_api::apply_chat_hook_outcome;
use skippy_inference_api::chat_skippy_hooks_enabled;
use skippy_inference_api::generation::GenerationRequest;
use skippy_inference_api::generation::MessageContent;
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
        // A prefilled assistant message is continued, not rendered as a turn:
        // templates close every assistant turn, so the model would see a
        // finished answer and stop. Render the history before it with the
        // generation prompt and append its text, as llama.cpp's server does.
        // A generation prompt that leaves a reasoning block open would put the
        // text inside it, so that case keeps the ordinary rendering.
        if let Some(continued) = assistant_continuation(request) {
            let mut history = request.clone();
            history.messages.pop();
            history.extra.remove("prefill_assistant");
            let continuation_options = ChatTemplateOptions {
                add_assistant: true,
                ..options.clone()
            };
            let mut prepared = self.prepare_chat_prompt(&history, continuation_options)?;
            if !generation_prompt_opens_reasoning(prepared.chat_parse_metadata.as_deref()) {
                prepared.text.push_str(&continued);
                return Ok(prepared);
            }
        }

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
            let recurrent_cache_prefix_text = if self
                .kv
                .as_ref()
                .is_some_and(|kv| kv.payload == StagePrefixCachePayload::KvRecurrent)
                && emulated.media.is_empty()
                && options.add_assistant
            {
                let mut prefix_options = options.clone();
                prefix_options.add_assistant = false;
                self.render_chat_prompt(request, &prefix_options, &marker, Some(&rewritten), false)
                    .ok()
                    .map(|rendered| rendered.prompt)
            } else {
                None
            };
            return Ok(PreparedGenerationPrompt {
                text: emulated.prompt,
                media: emulated.media,
                chat_parse_metadata: Some(emulated.metadata_json),
                recurrent_cache_prefix_text,
            });
        }

        let recurrent_cache_prefix_text = if self
            .kv
            .as_ref()
            .is_some_and(|kv| kv.payload == StagePrefixCachePayload::KvRecurrent)
            && native.media.is_empty()
            && options.add_assistant
        {
            let mut prefix_options = options.clone();
            prefix_options.add_assistant = false;
            self.render_chat_prompt(request, &prefix_options, &marker, None, true)
                .ok()
                .map(|rendered| rendered.prompt)
        } else {
            None
        };
        Ok(PreparedGenerationPrompt {
            text: native.prompt,
            media: native.media,
            chat_parse_metadata: Some(native.metadata_json),
            recurrent_cache_prefix_text,
        })
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
        // Short-lock reader pattern: clone the immutable reader and run the
        // template FFI on it, so template rendering never waits behind
        // decode's hold of the runtime mutex (and vice versa).
        let reader = lock_runtime(&self.runtime).model.reader();
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
        // Hold the inference mutex only long enough to clone the immutable
        // reader; the tokenizer FFI runs on the reader, so decode can hold
        // the runtime lock concurrently without serializing tokenization
        // behind it (see render_chat_prompt for the same pattern).
        let reader = lock_runtime(&self.runtime).model.reader();
        reader
            .tokenize(text, add_special)
            .map_err(openai_backend_error)
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

/// The text a `prefill_assistant` request asks the model to continue: the
/// last message, when the request set `prefill_assistant` and that message is
/// an assistant one with text-only content and no other field. `None` leaves
/// the request to the ordinary template rendering, which sees every field.
fn assistant_continuation(request: &GenerationRequest) -> Option<String> {
    request
        .extra
        .get("prefill_assistant")
        .filter(|value| !value.is_null())?;
    let last = request
        .messages
        .last()
        .filter(|message| message.role == "assistant" && message.extra.is_empty())?;
    let content = last.content.as_ref()?;
    let text_only = match content {
        MessageContent::Text(_) => true,
        MessageContent::Parts(parts) => parts.iter().all(|part| part.content_type == "text"),
        MessageContent::Other(_) => false,
    };
    if !text_only {
        return None;
    }
    // The same text the template would have rendered: parts joined by "\n".
    crate::frontend::message_content_to_generation_text(content, "", &mut Vec::new()).ok()
}

/// Whether the rendered generation prompt ends inside a reasoning block: it
/// contains the template's thinking start tag and no end tag after it.
fn generation_prompt_opens_reasoning(metadata_json: Option<&str>) -> bool {
    let Some(metadata) = metadata_json.and_then(|json| serde_json::from_str::<Value>(json).ok())
    else {
        return false;
    };
    let prompt = metadata
        .get("generation_prompt")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let Some(start_tag) = metadata
        .get("thinking_start_tag")
        .and_then(Value::as_str)
        .filter(|tag| !tag.is_empty())
    else {
        return false;
    };
    let Some(opened) = prompt.rfind(start_tag) else {
        return false;
    };
    let after = &prompt[opened + start_tag.len()..];
    !metadata
        .get("thinking_end_tags")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|tag| !tag.is_empty())
        .any(|tag| after.contains(tag))
}

#[cfg(test)]
mod continuation_tests {
    use super::*;
    use skippy_inference_api::generation::GenerationMessage;

    fn request(messages: Vec<(&str, &str)>, prefill: bool) -> GenerationRequest {
        let mut request = GenerationRequest {
            messages: messages
                .into_iter()
                .map(|(role, content)| GenerationMessage {
                    role: role.into(),
                    content: Some(MessageContent::Text(content.into())),
                    extra: Default::default(),
                })
                .collect(),
            ..GenerationRequest::default()
        };
        if prefill {
            request
                .extra
                .insert("prefill_assistant".into(), Value::String("unused".into()));
        }
        request
    }

    #[test]
    fn a_prefilled_assistant_message_is_continued() {
        let request = request(vec![("user", "Count."), ("assistant", "1\n2\n")], true);
        assert_eq!(assistant_continuation(&request).as_deref(), Some("1\n2\n"));
    }

    #[test]
    fn without_prefill_or_an_assistant_tail_nothing_is_continued() {
        let history = request(vec![("user", "Hi."), ("assistant", "Hello.")], false);
        assert_eq!(assistant_continuation(&history), None);
        let user_last = request(vec![("user", "Hi.")], true);
        assert_eq!(assistant_continuation(&user_last), None);
    }

    #[test]
    fn text_parts_are_joined_as_the_template_would_and_other_parts_are_left_to_it() {
        let mut parts = request(vec![("user", "Count.")], true);
        parts.messages.push(GenerationMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Parts(
                serde_json::from_value(serde_json::json!([
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"}
                ]))
                .unwrap(),
            )),
            extra: Default::default(),
        });
        assert_eq!(
            assistant_continuation(&parts).as_deref(),
            Some("first\nsecond")
        );
        if let Some(MessageContent::Parts(list)) = &mut parts.messages[1].content {
            list[1].content_type = "image_url".into();
        }
        assert_eq!(assistant_continuation(&parts), None);
    }

    #[test]
    fn a_prefill_with_other_fields_is_left_to_the_template() {
        let mut with_tool_calls = request(vec![("user", "Run it."), ("assistant", "Sure.")], true);
        with_tool_calls.messages[1].extra.insert(
            "tool_calls".into(),
            serde_json::json!([{"id": "call_1", "type": "function"}]),
        );
        assert_eq!(assistant_continuation(&with_tool_calls), None);
    }

    fn metadata(generation_prompt: &str) -> String {
        serde_json::json!({
            "generation_prompt": generation_prompt,
            "thinking_start_tag": "<think>",
            "thinking_end_tags": ["</think>"],
        })
        .to_string()
    }

    #[test]
    fn an_open_reasoning_block_in_the_generation_prompt_is_detected() {
        assert!(generation_prompt_opens_reasoning(Some(&metadata(
            "<|im_start|>assistant\n<think>\n"
        ))));
        assert!(!generation_prompt_opens_reasoning(Some(&metadata(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ))));
        assert!(!generation_prompt_opens_reasoning(Some(&metadata(
            "<|im_start|>assistant\n"
        ))));
        assert!(!generation_prompt_opens_reasoning(None));
    }
}
