//! Translation between the Anthropic Messages protocol and the internal
//! OpenAI-shaped chat types.
//!
//! The chat types are the pipeline's common denominator: every backend
//! (local runtime, mesh routing, MoA) consumes [`ChatCompletionRequest`] and
//! produces [`ChatCompletionResponse`]/[`ChatCompletionChunk`]. Keeping the
//! protocol boundary in this module (plus [`super::protocol`]) makes parity
//! reviewable field by field.
//!
//! Anthropic carries tool results inside the *following user turn* (as
//! `tool_result` blocks); the OpenAI pipeline carries them as distinct
//! `role: "tool"` messages with a `tool_call_id`. One Anthropic message can
//! therefore expand into several chat messages, so the request translation
//! returns a flat `Vec<ChatMessage>`.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::anthropic::protocol::{
    AnthropicContentBlock, AnthropicDelta, AnthropicMessageContent, AnthropicMessageDelta,
    AnthropicMessageStart, AnthropicMessagesRequest, AnthropicMessagesResponse,
    AnthropicMessagesStreamEvent, AnthropicResponseBlock, AnthropicSystemPrompt, AnthropicUsage,
};
use crate::chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent,
};
use crate::common::{FinishReason, Usage};
use crate::errors::OpenAiError;

/// Anthropic stop reasons. `refusal` maps from the OpenAI content-filter
/// finish reason; the `error` stop reason is produced only by in-band
/// streaming failures, matching the shape-honest failure semantics of #1905.
pub const STOP_REASON_END_TURN: &str = "end_turn";
pub const STOP_REASON_MAX_TOKENS: &str = "max_tokens";
pub const STOP_REASON_TOOL_USE: &str = "tool_use";
pub const STOP_REASON_REFUSAL: &str = "refusal";

/// Translate a Messages request into the internal chat request.
pub fn messages_request_to_chat_request(
    request: AnthropicMessagesRequest,
) -> Result<ChatCompletionRequest, OpenAiError> {
    if request.model.trim().is_empty() {
        return Err(OpenAiError::invalid_request("model is required"));
    }
    if request.messages.is_empty() {
        return Err(OpenAiError::invalid_request(
            "messages: at least one message is required",
        ));
    }

    let mut messages = Vec::new();
    if let Some(system) = request.system.as_ref() {
        let text = system_text(system);
        if !text.is_empty() {
            messages.push(text_message("system", &text));
        }
    }
    for (turn_index, message) in request.messages.iter().enumerate() {
        let role = message.role.as_str();
        if role != "user" && role != "assistant" {
            return Err(OpenAiError::invalid_request(format!(
                "messages[{turn_index}]: role must be `user` or `assistant`, got `{role}`"
            )));
        }
        expand_message(role, &message.content, &mut messages)?;
    }

    let tools = match request.tools.as_ref() {
        Some(tools) => Some(translate_tools(tools)?),
        None => None,
    };
    let tool_choice = request
        .tool_choice
        .as_ref()
        .map(translate_tool_choice)
        .transpose()?;

    Ok(ChatCompletionRequest {
        model: request.model,
        messages,
        stream: request.stream,
        max_tokens: None,
        max_completion_tokens: Some(request.max_tokens),
        temperature: request.temperature,
        top_p: request.top_p,
        n: None,
        logprobs: None,
        top_logprobs: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        response_format: None,
        tools,
        tool_choice,
        parallel_tool_calls: None,
        user: request.metadata.as_ref().and_then(|m| m.user_id.clone()),
        stop: request.stop_sequences.as_ref().map(|values| {
            if values.len() == 1 {
                crate::common::StopSequence::One(values[0].clone())
            } else {
                crate::common::StopSequence::Many(values.clone())
            }
        }),
        seed: None,
        reasoning: None,
        reasoning_effort: None,
        prompt_cache_key: None,
        prompt_cache_retention: None,
        stream_options: None,
        extra: Default::default(),
    })
}

fn system_text(system: &AnthropicSystemPrompt) -> String {
    match system {
        AnthropicSystemPrompt::Text(text) => text.clone(),
        AnthropicSystemPrompt::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| block.text.clone())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Expand one Anthropic message into zero or more chat messages.
fn expand_message(
    role: &str,
    content: &AnthropicMessageContent,
    out: &mut Vec<ChatMessage>,
) -> Result<(), OpenAiError> {
    match content {
        AnthropicMessageContent::Text(text) => {
            if !text.is_empty() {
                out.push(text_message(role, text));
            }
            Ok(())
        }
        AnthropicMessageContent::Blocks(blocks) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<Value> = Vec::new();
            let mut tool_results: Vec<ChatMessage> = Vec::new();
            for block in blocks {
                match block {
                    AnthropicContentBlock::Text(text) => {
                        if let Some(text) = text.text.as_deref()
                            && !text.is_empty()
                        {
                            text_parts.push(text.to_string());
                        }
                    }
                    AnthropicContentBlock::ToolUse(tool_use) => {
                        tool_calls.push(json!({
                            "id": tool_use.id,
                            "type": "function",
                            "function": {
                                "name": tool_use.name,
                                "arguments": serde_json::to_string(&tool_use.input)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            },
                        }));
                    }
                    AnthropicContentBlock::ToolResult(tool_result) => {
                        let content = if tool_result.is_error {
                            format!(
                                "tool error: {}",
                                tool_result_content_text(&tool_result.content)
                            )
                        } else {
                            tool_result_content_text(&tool_result.content)
                        };
                        tool_results.push(ChatMessage {
                            role: "tool".to_string(),
                            content: Some(MessageContent::Text(content)),
                            extra: BTreeMap::from([(
                                "tool_call_id".to_string(),
                                json!(tool_result.tool_use_id),
                            )]),
                        });
                    }
                    AnthropicContentBlock::Other(value) => {
                        // Unrecognized block: preserve it so media detection
                        // (media_url/media_data) can still see containers it
                        // understands instead of silently dropping client
                        // content.
                        text_parts.push(value.to_string());
                    }
                }
            }

            match role {
                "user" => {
                    // Anthropic orders a user turn's non-tool content before
                    // its tool results; both belong to the same turn.
                    let has_tool_results = !tool_results.is_empty();
                    if !text_parts.is_empty() {
                        out.push(text_message("user", &text_parts.join("\n")));
                    }
                    out.extend(tool_results);
                    if text_parts.is_empty() && !has_tool_results {
                        return Err(OpenAiError::invalid_request(
                            "messages: user message has no content",
                        ));
                    }
                }
                _ => {
                    let mut extra = BTreeMap::new();
                    if !tool_calls.is_empty() {
                        extra.insert("tool_calls".to_string(), Value::Array(tool_calls));
                    }
                    out.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: if text_parts.is_empty() {
                            None
                        } else {
                            Some(MessageContent::Text(text_parts.join("\n")))
                        },
                        extra,
                    });
                }
            }
            Ok(())
        }
    }
}

fn text_message(role: &str, text: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: Some(MessageContent::Text(text.to_string())),
        extra: Default::default(),
    }
}

fn tool_result_content_text(content: &Option<Value>) -> String {
    match content {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => other.to_string(),
    }
}

fn translate_tools(
    tools: &[crate::anthropic::protocol::AnthropicToolDefinition],
) -> Result<Value, OpenAiError> {
    let mut translated = Vec::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        if tool.name.trim().is_empty() {
            return Err(OpenAiError::invalid_request(format!(
                "tools[{index}].name must be a non-empty string"
            )));
        }
        let mut function = serde_json::Map::new();
        function.insert("name".to_string(), json!(tool.name));
        if let Some(description) = tool.description.as_ref() {
            function.insert("description".to_string(), json!(description));
        }
        function.insert(
            "parameters".to_string(),
            tool.input_schema.clone().unwrap_or_else(|| json!({})),
        );
        translated.push(json!({
            "type": "function",
            "function": Value::Object(function),
        }));
    }
    Ok(Value::Array(translated))
}

fn translate_tool_choice(
    choice: &crate::anthropic::protocol::AnthropicToolChoice,
) -> Result<Value, OpenAiError> {
    match choice.kind.as_str() {
        "auto" => Ok(json!("auto")),
        "any" => Ok(json!("required")),
        "tool" => {
            let Some(name) = choice.name.as_deref() else {
                return Err(OpenAiError::invalid_request(
                    "tool_choice.type `tool` requires tool_choice.name",
                ));
            };
            Ok(json!({
                "type": "function",
                "function": { "name": name },
            }))
        }
        other => Err(OpenAiError::invalid_request(format!(
            "tool_choice.type must be `auto`, `any`, or `tool`; got `{other}`"
        ))),
    }
}

/// Build the non-streaming Messages response from a chat completion.
pub fn messages_response_from_chat_response(
    response: &ChatCompletionResponse,
) -> Result<AnthropicMessagesResponse, OpenAiError> {
    let Some(choice) = response.choices.first() else {
        return Err(OpenAiError::internal(
            "chat completion response had no choices",
        ));
    };
    let message = &choice.message;
    let mut content = Vec::new();
    if let Some(text) = message.content.as_deref()
        && !text.is_empty()
    {
        content.push(AnthropicResponseBlock::Text {
            text: text.to_string(),
        });
    }
    if let Some(tool_calls) = message.tool_calls.as_ref()
        && let Some(calls) = tool_calls.as_array()
        && !calls.is_empty()
    {
        content.extend(tool_use_blocks(calls)?);
    }
    if content.is_empty() {
        content.push(AnthropicResponseBlock::Text {
            text: String::new(),
        });
    }

    Ok(AnthropicMessagesResponse {
        id: format!("msg_{}", response.id),
        kind: "message",
        role: "assistant",
        model: response.model.clone(),
        content,
        stop_reason: Some(stop_reason_from_finish(choice.finish_reason)),
        stop_sequence: None,
        usage: anthropic_usage(&response.usage),
    })
}

fn tool_use_blocks(tool_calls: &[Value]) -> Result<Vec<AnthropicResponseBlock>, OpenAiError> {
    let mut blocks = Vec::with_capacity(tool_calls.len());
    for (index, tool_call) in tool_calls.iter().enumerate() {
        let Some(function) = tool_call.get("function") else {
            return Err(OpenAiError::internal(format!(
                "tool_calls[{index}] is missing its function object"
            )));
        };
        let Some(name) = function.get("name").and_then(Value::as_str) else {
            return Err(OpenAiError::internal(format!(
                "tool_calls[{index}].function.name is missing"
            )));
        };
        let id = tool_call
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("toolu_{index}"));
        let arguments = function
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!("{}"));
        let input = match arguments {
            Value::String(text) => serde_json::from_str(&text).unwrap_or_else(|_| json!({})),
            Value::Object(_) => arguments,
            _ => json!({}),
        };
        blocks.push(AnthropicResponseBlock::ToolUse {
            id,
            name: name.to_string(),
            input,
        });
    }
    Ok(blocks)
}

pub fn stop_reason_from_finish(finish: Option<FinishReason>) -> &'static str {
    match finish {
        Some(FinishReason::Stop) | None => STOP_REASON_END_TURN,
        Some(FinishReason::Length) => STOP_REASON_MAX_TOKENS,
        Some(FinishReason::ToolCalls) => STOP_REASON_TOOL_USE,
        Some(FinishReason::ContentFilter) => STOP_REASON_REFUSAL,
    }
}

fn anthropic_usage(usage: &Usage) -> AnthropicUsage {
    AnthropicUsage {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
    }
}

/// Build the `message_start` event from the request identity. Input tokens
/// are reported at `message_delta` time, when the backend reports usage.
pub fn message_start_event(id: &str, model: &str) -> AnthropicMessagesStreamEvent {
    AnthropicMessagesStreamEvent::MessageStart {
        message: AnthropicMessageStart {
            id: format!("msg_{id}"),
            kind: "message",
            role: "assistant",
            model: model.to_string(),
            content: Vec::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 0,
                output_tokens: 0,
            },
        },
    }
}

pub fn text_block_start(index: usize) -> AnthropicMessagesStreamEvent {
    AnthropicMessagesStreamEvent::ContentBlockStart {
        index,
        content_block: AnthropicResponseBlock::Text {
            text: String::new(),
        },
    }
}

pub fn tool_use_block_start(index: usize, id: &str, name: &str) -> AnthropicMessagesStreamEvent {
    AnthropicMessagesStreamEvent::ContentBlockStart {
        index,
        content_block: AnthropicResponseBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input: json!({}),
        },
    }
}

pub fn text_block_delta(index: usize, text: &str) -> AnthropicMessagesStreamEvent {
    AnthropicMessagesStreamEvent::ContentBlockDelta {
        index,
        delta: AnthropicDelta::Text {
            text: text.to_string(),
        },
    }
}

pub fn json_block_delta(index: usize, partial_json: &str) -> AnthropicMessagesStreamEvent {
    AnthropicMessagesStreamEvent::ContentBlockDelta {
        index,
        delta: AnthropicDelta::InputJson {
            partial_json: partial_json.to_string(),
        },
    }
}

pub fn content_block_stop(index: usize) -> AnthropicMessagesStreamEvent {
    AnthropicMessagesStreamEvent::ContentBlockStop { index }
}

pub fn message_delta_event(
    stop_reason: Option<&'static str>,
    usage: &Usage,
) -> AnthropicMessagesStreamEvent {
    AnthropicMessagesStreamEvent::MessageDelta {
        delta: AnthropicMessageDelta {
            stop_reason,
            stop_sequence: None,
        },
        usage: anthropic_usage(usage),
    }
}

/// Map an OpenAI error body (the chat-completion error envelope) into an
/// Anthropic `error` stream event.
pub fn translate_stream_error_body(body: &Value) -> AnthropicMessagesStreamEvent {
    let error = body.get("error");
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("upstream request failed")
        .to_string();
    let kind = error
        .and_then(|error| error.get("type"))
        .and_then(Value::as_str)
        .map(|kind| match kind {
            "invalid_request_error" => "invalid_request_error",
            "rate_limit_error" => "rate_limit_error",
            "overloaded_error" => "overloaded_error",
            "timeout_error" => "timeout_error",
            _ => "api_error",
        })
        .unwrap_or("api_error");
    AnthropicMessagesStreamEvent::Error {
        error: crate::anthropic::protocol::AnthropicErrorMessage { kind, message },
    }
}

/// Assistant-side accumulator: turns chat completion chunks into the
/// Anthropic content-block event sequence.
///
/// Chat chunks carry text deltas directly and tool-call fragments in the
/// OpenAI `tool_calls` array shape. Tool fragments accumulate into a single
/// OpenAI call; for the wire we mirror that accumulation: text streams into
/// block 0 as it arrives, tool JSON streams into block 1 as fragments
/// arrive, and [`Self::finish`] closes open blocks and emits the terminal
/// `message_delta`/`message_stop` pair.
#[derive(Debug, Default)]
pub struct MessagesStreamAssembler {
    text_block_open: bool,
    tool_block_open: bool,
    text_index: usize,
    tool_index: usize,
    next_index: usize,
    tool_id: Option<String>,
    tool_name: Option<String>,
    last_usage: Option<Usage>,
    finished: bool,
}

impl MessagesStreamAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one chat chunk; returns the Anthropic events it produces.
    pub fn absorb(&mut self, chunk: &ChatCompletionChunk) -> Vec<AnthropicMessagesStreamEvent> {
        let mut events = Vec::new();
        if let Some(usage) = chunk.usage.as_ref() {
            self.last_usage = Some(usage.clone());
        }
        let Some(choice) = chunk.choices.first() else {
            return events;
        };
        if let Some(text) = choice.delta.content.as_deref()
            && !text.is_empty()
        {
            if !self.text_block_open {
                events.push(text_block_start(self.next_index));
                self.text_index = self.next_index;
                self.next_index += 1;
                self.text_block_open = true;
            }
            events.push(text_block_delta(self.text_index, text));
        }
        if let Some(fragments) = choice.delta.tool_calls.as_ref().and_then(Value::as_array)
            && !fragments.is_empty()
        {
            for fragment in fragments {
                if let Some(id) = fragment.get("id").and_then(Value::as_str) {
                    self.tool_id = Some(id.to_string());
                }
                if let Some(name) = fragment
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                {
                    self.tool_name = Some(name.to_string());
                }
                let arguments = fragment
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if arguments.is_empty() {
                    continue;
                }
                if !self.tool_block_open {
                    events.push(tool_use_block_start(
                        self.next_index,
                        self.tool_id.as_deref().unwrap_or("toolu_stream"),
                        self.tool_name.as_deref().unwrap_or("tool"),
                    ));
                    self.tool_index = self.next_index;
                    self.next_index += 1;
                    self.tool_block_open = true;
                }
                events.push(json_block_delta(self.tool_index, arguments));
            }
        }
        events
    }

    /// Emit the terminal events once the backend stream has ended.
    ///
    /// Idempotent: a stream whose final chunk already carried
    /// `finish_reason` has closed the protocol, and a repeated `finish`
    /// (epilogue after finish-reason chunk, or a backend that just ends)
    /// produces no further events.
    pub fn finish(
        &mut self,
        finish_reason: Option<FinishReason>,
    ) -> Vec<AnthropicMessagesStreamEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut events = Vec::new();
        if self.tool_block_open {
            events.push(content_block_stop(self.tool_index));
        }
        if self.text_block_open {
            events.push(content_block_stop(self.text_index));
        }
        let stop_reason = stop_reason_from_finish(finish_reason);
        let usage = self.last_usage.clone().unwrap_or_default();
        events.push(message_delta_event(Some(stop_reason), &usage));
        events.push(AnthropicMessagesStreamEvent::MessageStop {});
        events
    }
}

#[cfg(test)]
#[path = "translate_tests.rs"]
mod tests;
